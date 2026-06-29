# dump_cpu_order1_residual.jl — CPU residual_ho_3d!(order=1) reference for the GPU order-1
# validation. Run in the MAIN package env. Reads r3d.meta / r3d_M.f64 from $RIEMANN35_DATA,
# writes r3d_R1.f64 (interior, (35,n,n,n) GPU layout).
using Riemann35
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
meta = split(strip(read(joinpath(DATA,"r3d.meta"),String)),'\n')
n=parse(Int,meta[1]); dx=parse(Float64,meta[2]); Ma=parse(Float64,meta[3]); g=parse(Int,meta[4])
Mint = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_M.f64")))),35,n,n,n)
Mcpu = zeros(n+2g,n+2g,n,35)
for k in 1:n, j in 1:n, i in 1:n; @views Mcpu[i+g,j+g,k,:].=Mint[:,i,j,k]; end
for k in 1:n
  for j in 1:n+2g, hh in 1:g; @views Mcpu[hh,j,k,:].=Mcpu[g+1,j,k,:]; @views Mcpu[n+g+hh,j,k,:].=Mcpu[n+g,j,k,:]; end
  for i in 1:n+2g, hh in 1:g; @views Mcpu[i,hh,k,:].=Mcpu[i,g+1,k,:]; @views Mcpu[i,n+g+hh,k,:].=Mcpu[i,n+g,k,:]; end
end
R=zeros(n+2g,n+2g,n,35); residual_ho_3d!(R,Mcpu,n,n,n,g,dx,dx,dx,Ma; order=1)
write(joinpath(DATA,"r3d_R1.f64"), reinterpret(UInt8, vec(Float64.(permutedims(R[g+1:g+n,g+1:g+n,:,:],(4,1,2,3))))))
println("wrote r3d_R1.f64 (CPU order=1, n=$n Ma=$Ma)")
