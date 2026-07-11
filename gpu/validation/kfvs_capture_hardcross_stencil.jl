# kfvs_capture_hardcross_stencil.jl — Phase I.2: reproduce the 170 bad stage-2 anchor
# cells on the HARD Ma=100 crossing and write the FIRST failing stencil.
#
# Runs ONE CPU order-3 F3 step with ENV["KFVS_CAPTURE"]=1; the residual records the first
# interior cell whose pure θ=0 KFVS anchor update Mlo leaves the cone, plus its 7-cell face
# stencil (raw moments) and the update params (λ, Ma, s3max). Saved raw to
# kfvs_hardcross_stencil_raw.jld2 for offline U^Q/D/cone-spectrum analysis.
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"; ENV["KFVS_CAPTURE"]="1"
using Riemann35, MPI, Printf, JLD2
MPI.Initialized() || MPI.Init()

DATA = joinpath(@__DIR__, "..", "..", "data")
cross = reshape(collect(reinterpret(Float64, read(joinpath(DATA,"r3d_cross_ma100.f64")))), 35, 3)
bg=cross[:,1]; Mt=cross[:,2]; Mb=cross[:,3]; Ma=100.0
N=32; h=4; dx=1.0/N; s3max=max(40.0,4.0+Ma/2); dt=0.12*dx/(Ma/2+5)
M=zeros(Float64,N+2h,N+2h,N,35)
Cs=floor(Int,0.1*N); lo=div(N,2)-Cs;hi=div(N,2);lo2=div(N,2)+1;hi2=div(N,2)+1+Cs
for k in 1:N,j in 1:N,i in 1:N
  v=bg; (lo<=i<=hi&&lo<=j<=hi&&lo<=k<=hi)&&(v=Mb); (lo2<=i<=hi2&&lo2<=j<=hi2&&lo2<=k<=hi2)&&(v=Mt)
  @views M[i+h,j+h,k,:].=v
end
comm=MPI.COMM_WORLD; decomp=setup_mpi_cartesian_3d(N,N,N,h,comm); bc=:copy
halo_exchange_3d!(M,decomp,bc)

println("=== F3 one step, KFVS_CAPTURE on ===")
try
    step_highorder_3d!(M,dt,decomp,bc,N,N,N,h,dx,dx,dx,Ma; order=3,s3max=s3max,use_kfvs_anchor=true)
catch e
    @printf("(step threw %s after capture — expected; the capture happens pre-crash)\n", typeof(e))
end

caps = Riemann35._KFVS_CAPTURE[]
if caps === nothing || isempty(caps)
    error("no anchor-exit captured — margin threshold not hit (unexpected)")
end
@printf("captured %d anchor-exit cells; margin range [%.3e, %.3e]\n",
        length(caps), minimum(c.margin for c in caps), maximum(c.margin for c in caps))
# flatten to arrays: each field a Vector over captured cells
ncap = length(caps)
cells = reduce(hcat, [collect(c.cell) for c in caps])          # 3 x ncap
margins = [c.margin for c in caps]
Mlos = reduce(hcat, [c.Mlo for c in caps])                     # 35 x ncap
stk(sym) = reduce(hcat, [getfield(c, sym) for c in caps])      # 35 x ncap per stencil slot
out = joinpath(@__DIR__, "kfvs_hardcross_stencil_raw.jld2")
jldsave(out; ncap=ncap, cells=cells, margins=margins, Mlos=Mlos,
        lam=collect(caps[1].lam), Ma=caps[1].Ma, s3max=caps[1].s3max, dt=dt, dx=dx, halo=h, N=N,
        C=stk(:C), Lx=stk(:Lx), Rx=stk(:Rx), Ly=stk(:Ly), Ry=stk(:Ry), Lz=stk(:Lz), Rz=stk(:Rz),
        ic_bg=bg, ic_Mt=Mt, ic_Mb=Mb)
println("wrote $out  ($ncap cells)")
