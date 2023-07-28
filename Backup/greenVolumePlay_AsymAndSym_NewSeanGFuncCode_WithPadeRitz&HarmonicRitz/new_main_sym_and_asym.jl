using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product_sym_and_asym, bfgs_pade_sym_and_asym, dual_asym_only, gmres, 
approx_xi_solve_Ritz_HarmonicRitz_GreenOperator_sym_and_asym, PadeForRitz_GreenOperator_Code_sym_and_asym,
Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson_operator_sym_and_asym,
Restart_Ritz_jacobiDavidson_operator_sym_and_asym, b_sym_and_asym

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
# 
# P = I # this is the real version of the identity matrix since we are considering 
# # the symmetric and ansymmetric parts of some later calculations. 
# # If we were only considering the symmetric parts of some latter calculations,
# # we would need to use the imaginary version of the identity matrix. 
# # Pdag = conj.(transpose(P)) 
# # we could do the code above for a P other than the identity matrix
# Pdag = P

# Define the projection operators
# The first P is always the asym only constraint and it is
# the complex identity matrix. 
M = ones(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
M[:, :, :,:] .= 1.0im
N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
P0 = Diagonal(N)
print("P0 ", P0, "\n")

# First baby cube [1:cellsA[1]/2, 1:cellsA[2]/2, cellsA[3]/2:end]
M = zeros(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
M[1:Int(cellsA[1]/2), 1:Int(cellsA[2]/2), 1:Int(cellsA[3]/2),:] .= 1.0im
N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
P1 = Diagonal(N)
print("P1 ", P1, "\n")
P = [P0,P1]


xi = 0.25 # Initial Lagrangr multiplier for Asym constraint
l = [0.75] # Initial Lagrange multiplier for Sym constraint 
# Let's get the initial b vector (aka using the initial Lagrange multipliers). Done for test purposes
b = bv_sym_and_asym(ei, xi, l, P) 

xi = 0.5 # Temporary value
innerLoopDim = 3
restartDim = 1
tol_MGS = 1.0e-12
tol_conv = 1.0e-3
tol_eigval = 1.0e-3
tol_bicgstab = 1e-3
tol_bissection = 1e-4

# This is the code for the main function call using bfgs with the power iteration
# method to solve for the Lagrange multiplier and gmres to solve for |T>.
# # Start 
bfgs = BFGS_fakeS_with_restart_pade(gMemSlfN, gMemSlfA,xi,l,innerLoopDim,
	restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,tol_bissection,dual,P,
		chi_inv_coeff,ei,b,cellsA,root_solve,jacDavRitzHarm_restart)
# the BFGS_fakeS_with_restart_pi function can be found in the bfgs_power_iteration_asym_only file
dof = bfgs[1]
grad = bfgs[2]
dualval = bfgs[3]
objval = bfgs[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")
# End 

# # ITEM call code 
# item = ITEM(gMemSlfN, gMemSlfA,l,dual,P,chi_inv_coeff,ei,cellsA,validityfunc) 
# print("dof ", item[1], "\n")
# print("grad ", item[2], "\n")
# print("dualval ", item[3], "\n")
# print("objval ", item[4], "\n")



# testvect = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3,1)
# rand!(testvect)
# print("Gv_AA(gMemSlfN, cellsA, vec) ", Gv_AA(gMemSlfN, cellsA, testvect), "\n")
# print("l[1]*asym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,testvect)", l[1]*asym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,testvect)
# , "\n")
