module RestartHarmonicRitz_TestFile
using LinearAlgebra, Random, Base.Threads, Plots, Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson

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
minEigPos = argmin(abs.(trueEigSys.values))
minEig = trueEigSys.values[minEigPos]
dims = size(opt)
innerLoopDim = 50
restartDim = 5

# Basic Ritz
val_basic = jacDavRitzHarm_basic(opt,innerLoopDim,tol_MGS,tol_conv,tol_eigval,
	tol_bicgstab)
# Restart Ritz
val_restart = jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
	tol_bicgstab)

println("The Julia eigenvalues are ",trueEigSys.values,".")
print("Basic harmonic Ritz eigenvalue closest to 0 is ", val_restart, "\n")
print("Restart harmonic Ritz eigenvalue closest to 0 is ", val_restart, "\n")
println("The Julia smallest eigenvalue is ", minEig,".")

end