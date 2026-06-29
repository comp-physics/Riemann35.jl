# dump_cpu_limiter_residual.jl — CPU residual_ho_3d!(order=2, use_limiter=true) reference
# for the GPU ho_realizability_limiter validation. MAIN env. The limiter path
# (face_states_lim) builds realizable-by-construction faces and does NOT use the vacuum
# floor; HO_VACUUM_FLOOR[] is set to match the GPU call but is unused on this path.
# Reads r3d.meta / r3d_M.f64 from $RIEMANN35_DATA, writes r3d_Rlim.f64.
using Riemann35
Riemann35.HO_VACUUM_FLOOR[] = 0.001
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
meta = split(strip(read(joinpath(DATA,"r3d.meta"),String)),'\n')
n=parse(Int,meta[1]); dx=parse(Float64,meta[2]); Ma=parse(Float64,meta[3]); g=parse(Int,meta[4])
Mint = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_M.f64")))),35,n,n,n)
Mc = zeros(n+2g,n+2g,n,35)
for k in 1:n, j in 1:n, i in 1:n; @views Mc[i+g,j+g,k,:].=Mint[:,i,j,k]; end
for k in 1:n
  for j in 1:n+2g, hh in 1:g; @views Mc[hh,j,k,:].=Mc[g+1,j,k,:]; @views Mc[n+g+hh,j,k,:].=Mc[n+g,j,k,:]; end
  for i in 1:n+2g, hh in 1:g; @views Mc[i,hh,k,:].=Mc[i,g+1,k,:]; @views Mc[i,n+g+hh,k,:].=Mc[i,n+g,k,:]; end
end
R=zeros(n+2g,n+2g,n,35); residual_ho_3d!(R,Mc,n,n,n,g,dx,dx,dx,Ma; order=2, use_limiter=true)
write(joinpath(DATA,"r3d_Rlim.f64"), reinterpret(UInt8, vec(Float64.(permutedims(R[g+1:g+n,g+1:g+n,:,:],(4,1,2,3))))))
println("wrote r3d_Rlim.f64 (CPU order=2 use_limiter)")
