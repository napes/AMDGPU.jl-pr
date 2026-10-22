using Test
using AMDGPU
using AMDGPU: Device
using InteractiveUtils

# Compile every concrete method defined in the package sources through the host LLVM
# pipeline. This catches target-only symbols escaping the device overlay without asking the
# CPU JIT to link AMDGPU runtime entry points that only exist in a GPU compilation context.
const HOST_AOT_FORBIDDEN_PATTERNS = (
    r"\bllvm\.amdgcn\.",
    r"\b__ockl_",
    r"\b__ocml_",
)

# Every package source file counts, not only `src/device`: a plain function that calls a
# target intrinsic (like `ROCKernels.hardware_lane`) is compiled for the host by a sysimage
# build just as a device-method leak is. Methods are found by the file that defines them,
# in every loaded module, so extensions of Base and KernelInterface functions are included
# (named functions and constructors; closures defined inside a function are not reached,
# but they are compiled with it). `Core.methodtable` doesn't exist before Julia 1.12, so
# the method table isn't walked directly.
# Device-only methods live in `AMDGPU.method_table` and are not seen here, only their
# `@device_function` CPU stubs are.
const PACKAGE_SOURCE_DIRS = map(dir -> normpath(joinpath(pkgdir(AMDGPU), dir)), ("src", "ext"))

function in_package_sources(method::Method)
    file = normpath(String(method.file))
    return any(dir -> startswith(file, dir * Base.Filesystem.path_separator), PACKAGE_SOURCE_DIRS)
end

# every loaded module and its submodules
function loaded_modules()
    modules = Module[]
    seen = Set{Module}()
    todo = collect(Module, values(Base.loaded_modules))
    while !isempty(todo)
        mod = pop!(todo)
        mod in seen && continue
        push!(seen, mod); push!(modules, mod)
        for name in names(mod; all=true)
            isdefined(mod, name) && !Base.isdeprecated(mod, name) || continue
            value = getfield(mod, name)
            value isa Module && push!(todo, value)
        end
    end
    return modules
end

# the function a method belongs to, when it can be called without an instance: `nothing`
# for closures and callable structs
function method_function(method::Method)
    ft = method.sig.parameters[1]
    if ft <: Type
        T = ft.parameters[1]
        return T isa Type ? T : nothing
    end
    return Base.issingletontype(ft) ? ft.instance : nothing
end

function concrete_package_methods()
    seen = Set{Method}()
    rows = NamedTuple[]
    for mod in loaded_modules(), name in names(mod; all=true, imported=true)
        isdefined(mod, name) && !Base.isdeprecated(mod, name) || continue
        callable = getfield(mod, name)
        callable isa Union{Function,Type} || continue
        for method in methods(callable)
            method in seen && continue
            push!(seen, method)
            in_package_sources(method) || continue
            method.sig isa DataType && Base.isdispatchtuple(method.sig) || continue
            f = method_function(method)
            f === nothing && continue
            signature = Tuple{method.sig.parameters[2:end]...}
            label = string(method.module, ".", method.name, " :: ", signature)
            push!(rows, (; label, f, signature, method))
        end
    end
    sort!(rows; by=row -> row.label)
    return rows
end

function host_llvm(f, signature)
    io = IOBuffer()
    code_llvm(io, f, signature; raw=true, dump_module=true, optimize=false)
    return String(take!(io))
end

function forbidden_host_symbols(llvm)
    return filter(pattern -> occursin(pattern, llvm), HOST_AOT_FORBIDDEN_PATTERNS)
end

function is_cpu_stub(method::Method)
    code = Base.uncompressed_ast(method).code
    return any(code) do statement
        Meta.isexpr(statement, :call) || return false
        any(statement.args) do argument
            argument isa String && occursin("not intended for use on the CPU", argument)
        end
    end
end

# Deliberately leave this function outside `@device_function`. It proves that host
# reflection exposes the exact target-only symbol class this regression must detect.
Base.@noinline unprotected_amdgcn_negative_control() =
    ccall("llvm.amdgcn.s.memtime", llvmcall, UInt64, ())

@testset "device methods are isolated from host AOT" begin
    negative_llvm = host_llvm(unprotected_amdgcn_negative_control, Tuple{})
    @test !isempty(forbidden_host_symbols(negative_llvm))

    rows = concrete_package_methods()
    # This is a lower-bound sanity check, not a frozen inventory: the rows above are
    # derived from the loaded method table and grow automatically with the package.
    @test length(rows) > 100

    # The walk must reach beyond `src/device`, or a leak outside it passes unseen.
    device_dir = normpath(joinpath(pkgdir(AMDGPU), "src", "device"))
    @test any(row -> startswith(normpath(String(row.method.file)), device_dir), rows)
    @test any(row -> !startswith(normpath(String(row.method.file)), device_dir), rows)
    @test any(row -> basename(String(row.method.file)) == "ROCKernels.jl", rows)

    failures = Pair{String,Any}[]
    unreflectable = String[]
    for row in rows
        llvm = try
            host_llvm(row.f, row.signature)
        catch err
            # a method that cannot be reflected (e.g. it needs a GPU context to infer) is
            # reported, not failed; one whose error names a target symbol is a leak
            detail = sprint(showerror, err)
            hits = forbidden_host_symbols(detail)
            isempty(hits) ? push!(unreflectable, row.label) : push!(failures, row.label => hits)
            continue
        end
        isempty(llvm) && push!(failures, row.label => "empty host IR")
        hits = forbidden_host_symbols(llvm)
        isempty(hits) || push!(failures, row.label => hits)
    end
    @info "Host-AOT scan" methods=length(rows) unreflectable=length(unreflectable)
    for (label, detail) in failures
        @error "Host-AOT device symbol leak" label detail
    end
    @test isempty(failures)

    # Pin representative CPU stubs for every family expanded by #1022. The broad IR
    # walk above catches newly-added concrete methods; these assertions ensure that the
    # walk still reaches indexing, synchronization, execution control, hostcall/timing,
    # wavefront, and both WMMA modules rather than passing on an unrelated subset.
    representatives = (
        "indexing" => Device.workitemIdx_x,
        "synchronization" => Device.sync_grid,
        "execution control" => Device.endpgm,
        "hostcall signal" => Device.device_signal_cas!,
        "timing" => Device.memtime,
        "wavefront" => Device.ballot,
        "WMMA RDNA3" => Device.WMMA_RDNA3._f32_16x16x16_f16,
        "WMMA RDNA4" => Device.WMMA_RDNA4._f32_16x16x16_f16,
        "sub-group lane (ROCKernels)" => AMDGPU.ROCKernels.hardware_lane,
    )
    for (family, f) in representatives
        family_rows = filter(row -> row.f === f, rows)
        @testset "$family family" begin
            @test !isempty(family_rows)
            @test any(row -> is_cpu_stub(row.method), family_rows)
        end
    end
end
