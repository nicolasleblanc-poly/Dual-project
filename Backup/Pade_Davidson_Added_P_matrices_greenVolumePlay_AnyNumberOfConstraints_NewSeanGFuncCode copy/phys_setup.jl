module phys_setup 
export G_create, ei_create
using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, bfgs_power_iteration_asym_only, dual_asym_only, gmres

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
	println("Green function construction completed.")
	# New Green function code stop 
	return gMemSlfN,gMemSlfA,gMemExtN
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