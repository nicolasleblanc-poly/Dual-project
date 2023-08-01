using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, bfgs_power_iteration_asym_only, dual_asym_only, gmres, item_asym_only, cg_asym_only

# The following latex files explains the different functions of the program
# https://www.overleaf.com/read/yrdmwzjhqqqs

threads = nthreads()
# Set the number of BLAS threads. The number of Julia threads is set as an 
# environment variable. The total number of threads is Julia threads + BLAS 
# threads. VICu is does not call BLAS libraries during threaded operations, 
# so both thread counts can be set near the available number of cores. 
BLAS.set_num_threads(threads)
# Analogous comments apply to FFTW threads. 
FFTW.set_num_threads(threads)
# Confirm thread counts
blasThreads = BLAS.get_num_threads()
fftwThreads = FFTW.get_num_threads()
println("MaxGTests initialized with ", nthreads(), 
	" Julia threads, $blasThreads BLAS threads, and $fftwThreads FFTW threads.")


# # Old Green function code start
# # Setup for the creation of the total Green function
# # Start 
# # Define test volume, all lengths are defined relative to the wavelength. 
# # Number of cells in the volume. 
# # cellsA = [8, 8, 8]
# cellsA = [2, 2, 2]
# cellsB = [1, 1, 1]
# # Edge lengths of a cell relative to the wavelength. 
# scaleA = (0.02, 0.02, 0.02)
# scaleB = (0.02, 0.02, 0.02)
# # Center position of the volume. 
# coordA = (0.0, 0.0, 0.0)
# coordB = (-0.3, 0.0, 0.0)
# # Create MaxG volumes.
# volA = genMaxGVol(MaxGDom(cellsA, scaleA, coordA))
# volB = genMaxGVol(MaxGDom(cellsB, scaleB, coordB))
# # Information for Green function construction. 
# # Complex frequency ratio. 
# freqPhase = 1.0 + im * 0.0
# # Gauss-Legendre approximation orders. 
# ordGLIntFar = 2
# ordGLIntMed = 8
# ordGLIntNear = 16
# # Cross over points for Gauss-Legendre approximation.
# crossMedFar = 16
# crossNearMed = 8
# assemblyInfo = MaxGAssemblyOpts(freqPhase, ordGLIntFar, ordGLIntMed, 
# 	ordGLIntNear, crossMedFar, crossNearMed)
# # Pre-allocate memory for circulant green function vector. 
# # Let's say we are only considering the AA case for simplicity
# greenCircAA = Array{ComplexF64}(undef, 3, 3, 2 * cellsA[1], 2 * cellsA[2], 
# 	2 * cellsA[3])
# greenCircAB = Array{ComplexF64}(undef, 3, 3, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])
# # End 

# CPU computation of Green function
# Start 
# The first index is the target and the second is the source
# # For the AA case 
# genGreenSlf!(greenCircAA, volA, assemblyInfo)
# # For the AB case 
# genGreenExt!(greenCircAB, volA, volB, assemblyInfo)
# # End 
# # Old Green function code end


# New Green function code start 
# Define test volume, all lengths are defined relative to the wavelength. 
# Number of cells in the volume. 
cellsA = [2, 2, 2]
cellsB = [1, 1, 1]
# Edge lengths of a cell relative to the wavelength. 
scaleA = (0.1, 0.1, 0.1)
scaleB = (0.2, 0.2, 0.2)
# Center position of the volume. 
coordA = (0.0, 0.0, 0.0)
coordB = (0.0, 0.0, 1.0)
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

# New Green function test code 
# Start 
# testvect = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
# rand!(testvect)

# print("testvect ",test(gMemSlfN, testvect), "\n")

# grnOpr!(gMemSlfN)

# copyto!(gMemSlfN.srcVec, testvect)

# # print("gMemSlfN.srcVec ",gMemSlfN.srcVec, "\n")

# # greenActAA!()

# print("gMemSlfN.trgVec ",gMemSlfN.trgVec, "\n")
# End 


# Source current memory
currSrcAB = Array{ComplexF64}(undef, cellsB[1], cellsB[2], cellsB[3], 3)

# Calculate ei electric field vector 
# elecIn = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3) 
# elecIn_vect = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
# currSrcAB[1,1,1,3] = 10.0 + 0.0im

# Good ei code start 
ei = Gv_AB(gMemExtN, cellsA, cellsB, currSrcAB) # This is a vector, so it is already reshaped
print("ei ", ei, "\n")
# Good ei code end 

# copy!(elecIn,GAB_curr);
# elecIn_reshaped = reshape(elecIn, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
# print(reshape(elecIn, (cellsA[1]*cellsA[2]*cellsA[3]*3)))
# print(size(reshape(elecIn, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))))
# print(ei)

# End 

# Let's define some values used throughout the program.
# chi coefficient
chi_coeff = 3.0 + 0.01im
# inverse chi coefficient
chi_inv_coeff = 1/chi_coeff 
chi_inv_coeff_dag = conj(chi_inv_coeff)
# define the projection operators

# What is the length of P? How do we fill in P precisely?

P = I # this is the real version of the identity matrix since we are considering 
# the symmetric and ansymmetric parts of some later calculations. 
# If we were only considering the symmetric parts of some latter calculations,
# we would need to use the imaginary version of the identity matrix. 
# Pdag = conj.(transpose(P)) 
# we could do the code above for a P other than the identity matrix
Pdag = P
# let's get the initial b vector (aka using the initial Lagrange multipliers). Done for test purposes
# b = bv(ei, l,P)
l = [0.75] # initial Lagrange multipliers

# BFGS call code using gmres 
# This is the code for the main function call using BFGS with the power iteration
# method to solve for the Lagrange multiplier and gmres to solve for |T>.
# # Start 
bfgs = BFGS_fakeS_with_restart_pi(gMemSlfN,gMemSlfA,l,dual,P,chi_inv_coeff,ei,
cellsA,validityfunc,power_iteration_second_evaluation)
# The BFGS_fakeS_with_restart_pi function can be found in the bfgs_power_iteration_asym_only file
dof = bfgs[1]
grad = bfgs[2]
dualval = bfgs[3]
objval = bfgs[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")
# End 

# ITEM call code 
# item = ITEM(gMemSlfN, gMemSlfA,l,dual,P,chi_inv_coeff,ei,cellsA,validityfunc) 
# print("dof ", item[1], "\n")
# print("grad ", item[2], "\n")
# print("dualval ", item[3], "\n")
# print("objval ", item[4], "\n")


# Test code 
# testvect = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3,1)
# rand!(testvect)
# print("Gv_AA(gMemSlfN, cellsA, vec) ", Gv_AA(gMemSlfN, cellsA, testvect), "\n")
# print("l[1]*asym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,testvect)", l[1]*asym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,testvect)
# , "\n")



