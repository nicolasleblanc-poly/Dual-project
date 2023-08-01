module phys_setup 
export G_create, ei_create
using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, bfgs_power_iteration_asym_only, dual_asym_only, gmres, JLD2, FileIO

function G_create(cellsA,cellsB,scaleA,scaleB,coordA,coordB)
	### Prepare Green functions
	println("Green function construction started.")
	# Create domains
	assemblyInfo = MaxGAssemblyOpts()
	aDom = MaxGDom(cellsA, scaleA, coordA) 
	bDom = MaxGDom(cellsB, scaleB, coordB)
	## Prepare Green function operators. 
	# First number is adjoint mode, set to zero for standard operation. 
	gMemSlfN = MaxGOprGenSlf(0, assemblyInfo, aDom)
	gMemSlfA = MaxGOprGenSlf(1, assemblyInfo, aDom)
	gMemExtN = MaxGOprGenExt(0, assemblyInfo, aDom, bDom)
	# # Operator shorthands
	# greenActAA! = () -> grnOpr!(gMemSlfN) 
	# greenAdjActAA! = () -> grnOpr!(gMemSlfA)
	# greenActAB! = () -> grnOpr!(gMemExtN)

	"""
	When the different parts of the Green function operator for a chosen size 
		haven't been created. 
	"""
	println("Green function construction completed.")
	# # New Green function code stop 
	return gMemSlfN,gMemSlfA,gMemExtN
	"""
	When the different parts of the Green function operator for a chosen size 
		have been created. 
	Instead of returning something, this function will write the outputs to a 
		serialized file. 
	"""
	# # Create a file for gMemSlfN
	# gMemSlfN_file = File(format"JLD2", "/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/Operator/gMemSlfN.jld2")
	# # Save data into the file
	# save(gMemSlfN_file, gMemSlfN)

	# # Create a file for gMemSlfN
	# gMemSlfA_file = File(format"JLD2", "/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/Operator/gMemSlfA.jld2")
	# # Save data into the file
	# save(gMemSlfA_file, gMemSlfA)

	# # Create a file for gMemSlfN
	# gMemExtN_file = File(format"JLD2", "/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/Operator/gMemExtN.jld2")
	# # Save data into the file
	# save(gMemExtN_file, gMemExtN)
end 

function ei_create(gMemExtN, cellsA, cellsB)
	# Source current memory
	currSrcAB = Array{ComplexF64}(undef, cellsB[1], cellsB[2], cellsB[3], 3)
	# Good ei code start 
	ei = Gv_AB(gMemExtN, cellsA, cellsB, currSrcAB) # This is a vector, so it is already reshaped
	# print("ei ", ei, "\n")
	# Good ei code end 
	return ei 
end
end 