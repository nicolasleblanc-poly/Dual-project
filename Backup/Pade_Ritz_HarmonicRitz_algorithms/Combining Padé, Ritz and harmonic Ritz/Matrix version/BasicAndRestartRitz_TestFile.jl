module RestartHarmonicRitz_TestFile
using LinearAlgebra, Random, Base.Threads, Plots, Restart_Ritz_jacobiDavidson

sz = 256
tol_MGS = 1e-9
tol_bicgstab = 1e-4
tol_conv = 1e-4
tol_cg = 1e-4
tol_eigval = 1e-6
max_nb_it = 1000
opt = Array{ComplexF64}(undef,sz,sz)
rand!(opt)
opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
trueEigSys = eigen(opt)
minEigPos = argmax(abs.(trueEigSys.values))
minEig = trueEigSys.values[minEigPos]
maxEigPos = argmax(abs.(trueEigSys.values))
maxEig = trueEigSys.values[maxEigPos]
extremeEig = 0.0
if abs(maxEig) > abs(minEig)
	extremeEig = maxEig
else # Essentially if abs(maxEig) < abs(minEig)
	extremeEig = minEig
end

dims = size(opt)
innerLoopDim = 50
restartDim = 5

# Basic Ritz
val_basic = jacDavRitz_basic(opt,innerLoopDim,tol_MGS,tol_conv,tol_eigval,
	tol_bicgstab)
# Restart Ritz
val_restart = jacDavRitz_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
	tol_bicgstab)

println("The Julia eigenvalues are ",trueEigSys.values,".")
print("Basic Ritz eigenvalue closest to 0 is ", val_restart, "\n")
print("Restart Ritz eigenvalue closest to 0 is ", val_restart, "\n")
println("The extreme Julia eigenvalue is ", extremeEig,".")

end