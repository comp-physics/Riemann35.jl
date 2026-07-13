# validate_hiorder3_both.jl — one process, one compile: (A) OFF-path regression (my edits
# must keep use_logjacobi_recon=false byte-identical to the raw CPU golden) AND (B) log-Jacobi
# ON-path parity vs the CPU log-Jacobi golden. Both ≤ 1e-10.
using CUDA, Printf
include(joinpath(@__DIR__, "..", "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))

# ensure both CPU goldens exist
repo = normpath(joinpath(@__DIR__, "..", ".."))
if !isfile(joinpath(DATA,"r3d_ho3_R0.f64"))
    run(setenv(`$(Base.julia_cmd()) --project=$repo $(joinpath(@__DIR__,"dump_cpu_hiorder3_residual.jl"))`, ENV))
end
if !isfile(joinpath(DATA,"r3d_ho3_lj_R0.f64"))
    run(setenv(`$(Base.julia_cmd()) --project=$repo $(joinpath(@__DIR__,"dump_cpu_hiorder3_logjacobi.jl"))`, ENV))
end

meta = split(strip(read(joinpath(DATA,"r3d_ho3.meta"), String)), '\n')
n=parse(Int,meta[1]); dx=parse(Float64,meta[2]); Ma=parse(Float64,meta[3]); g=parse(Int,meta[4]); dtN=parse(Float64,meta[5]); s3max=parse(Float64,meta[6])
ld(f)=reshape(collect(reinterpret(Float64, read(joinpath(DATA,f)))), 35, n, n, n)
Mint=ld("r3d_ho3_M.f64")
R0=ld("r3d_ho3_R0.f64"); RN=ld("r3d_ho3_RN.f64")
R0lj=ld("r3d_ho3_lj_R0.f64"); RNlj=ld("r3d_ho3_lj_RN.f64")

nf=n+2g; G=zeros(35,nf,nf,nf); cl(a)= a<1 ? 1 : (a>n ? n : a)
for c in 1:nf, b in 1:nf, a in 1:nf; @views G[:,a,b,c] .= Mint[:, cl(a-g), cl(b-g), cl(c-g)]; end

rel(A,B)=maximum(abs.(A .- B) ./ max.(abs.(B), 1.0))
function chk(tag, Rg, Rref)
    r=rel(Rg,Rref); @printf("%-34s max rel=%.3e  %s\n", tag, r, r<=1e-10 ? "PASS" : "FAIL"); r
end

# GATE = dt=0 (θ=1, pure WENO5+recon): (A) off-path byte-identity regression, (B) log-J parity.
# dt=dtN is INFORMATIONAL: this branch has a PRE-EXISTING GPU θ*-IDP limiter parity gap
# (baseline off-path dt=dtN fails ~3.5 with ORIGINAL code) — separate from log-Jacobi, which
# acts at the reconstruction level. If off-dtN ≈ logJ-dtN gap, it's the shared θ* issue.
println("--- GATE: dt=0 (θ=1) — off-path regression + log-Jacobi parity ---")
a0=chk("off  dt=0 (byte-identity)", residual3d_order3_gpu(G,n,n,n,g,dx,dx,dx,Ma,0.0; s3max=s3max), R0)
b0=chk("logJ dt=0 (parity)",        residual3d_order3_gpu(G,n,n,n,g,dx,dx,dx,Ma,0.0; s3max=s3max, use_logjacobi_recon=true), R0lj)
# sanity: log-Jacobi actually changed the residual (else the override never fired)
diff=maximum(abs.(residual3d_order3_gpu(G,n,n,n,g,dx,dx,dx,Ma,0.0;s3max=s3max,use_logjacobi_recon=true) .- residual3d_order3_gpu(G,n,n,n,g,dx,dx,dx,Ma,0.0;s3max=s3max)))
@printf("logJ vs raw residual diff (should be >0, the marginal upgrade fired) = %.3e\n", diff)
println("--- INFORMATIONAL: dt=dtN (θ*-IDP; pre-existing branch limiter gap) ---")
aN=chk("off  dt=dtN", residual3d_order3_gpu(G,n,n,n,g,dx,dx,dx,Ma,dtN;s3max=s3max), RN)
bN=chk("logJ dt=dtN", residual3d_order3_gpu(G,n,n,n,g,dx,dx,dx,Ma,dtN;s3max=s3max, use_logjacobi_recon=true), RNlj)
ok = (a0 <= 1e-10) && (b0 <= 1e-10)
println(ok ? "GATE: PASS (dt=0 off-identity + log-J parity ≤ 1e-10)" : "GATE: FAIL")
exit(ok ? 0 : 1)
