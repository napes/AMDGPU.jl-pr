using Test
using AMDGPU
using AMDGPU: Device
using InteractiveUtils

# Compile every concrete device method through the host LLVM pipeline. This catches
# target-only symbols escaping the device overlay without asking the CPU JIT to link
# AMDGPU runtime entry points that only exist in a GPU compilation context.
const HOST_AOT_FORBIDDEN_PATTERNS = (
    r"\bllvm\.amdgcn\.",
    r"\b__ockl_",
    r"\b__ocml_",
)

function device_modules()
    modules = Module[Device]
    i = 1
    while i <= length(modules)
        mod = modules[i]
        for name in names(mod; all=true)
            isdefined(mod, name) || continue
            value = getfield(mod, name)
            value isa Module || continue
            value === mod && continue
            parentmodule(value) === mod || continue
            value in modules || push!(modules, value)
        end
        i += 1
    end
    return modules
end

function concrete_device_methods()
    root = normpath(joinpath(pkgdir(AMDGPU), "src", "device"))
    modules = device_modules()
    seen = Set{Method}()
    rows = NamedTuple[]

    for mod in modules, name in names(mod; all=true)
        isdefined(mod, name) || continue
        f = getfield(mod, name)
        f isa Function || continue
        for method in methods(f)
            method in seen && continue
            method.module in modules || continue
            startswith(normpath(String(method.file)), root) || continue
            method.sig isa DataType && Base.isdispatchtuple(method.sig) || continue

            push!(seen, method)
            signature = Tuple{method.sig.parameters[2:end]...}
            label = string(mod, ".", name, " :: ", signature)
            push!(rows, (; label, f, signature, method))
        end
    end
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

    rows = concrete_device_methods()
    # This is a lower-bound sanity check, not a frozen inventory: the rows above are
    # derived from the loaded method table and grow automatically with the device API.
    @test length(rows) > 100

    failures = Pair{String,Any}[]
    for row in rows
        llvm = host_llvm(row.f, row.signature)
        isempty(llvm) && push!(failures, row.label => "empty host IR")
        hits = forbidden_host_symbols(llvm)
        isempty(hits) || push!(failures, row.label => hits)
    end
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
    )
    for (family, f) in representatives
        family_rows = filter(row -> row.f === f, rows)
        @testset "$family family" begin
            @test !isempty(family_rows)
            @test any(row -> is_cpu_stub(row.method), family_rows)
        end
    end
end
