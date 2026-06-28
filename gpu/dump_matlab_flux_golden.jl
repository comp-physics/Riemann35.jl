using MAT
GD=joinpath(@__DIR__,"..","test","goldenfiles")
DATA=get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "data"))
data=matread(joinpath(GD,"test_flux_eigenvalues_golden.mat"))["golden_data"]["tests"]
fb=data["flux_basic"]; input=vec(Float64.(fb["input"])); ex=fb["output"]
write("$DATA/flxg_in.f64", reinterpret(UInt8, input))
write("$DATA/flxg_out.f64", reinterpret(UInt8, vcat(vec(Float64.(ex["Fx"])),vec(Float64.(ex["Fy"])),vec(Float64.(ex["Fz"])),vec(Float64.(ex["M_real"])))))
println("flux golden dumped: input len=", length(input), "  keys=", keys(ex))
