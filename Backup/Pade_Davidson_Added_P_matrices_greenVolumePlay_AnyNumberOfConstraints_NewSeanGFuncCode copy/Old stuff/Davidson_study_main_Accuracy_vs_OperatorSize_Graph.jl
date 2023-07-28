using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, bfgs_power_iteration_asym_only, dual_asym_only, gmres,
phys_setup, opt_setup, Davidson_Operator_HarmonicRitz_module_May16

"""
Before jumping to larger system with the Davidson iteration method, using 
Harmonic Ritz vectors, we want to study its behavior when we use it on an 
operator, which is the Green function in our case, instead of a matrix, where 
it works very well.

We want to find the eigenvalues of A = Sym(P*G)+alpha*Asym(G-chi^-1) 
with alpha as a parameter to test, a random complex vector for the diagonal of 
P, real(chi) = 3 and imag(chi) = 10e-3
"""

# Setup
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
cellsA = [2,2,2]
cellsB = [1, 1, 1]
# Edge lengths of a cell relative to the wavelength. 
scaleA = (0.1, 0.1, 0.1)
scaleB = (0.2, 0.2, 0.2)
# Center position of the volume. 
coordA = (0.0, 0.0, 0.0)
coordB = (0.0, 0.0, 1.0)

# # Green function creation 
G_call = G_create(cellsA,cellsB,scaleA,scaleB,coordA,coordB)
gMemSlfN = G_call[1]
gMemSlfA = G_call[2]
gMemExtN = G_call[3]

# # P matrix creation 
# M = ones(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
# M[:, :, :,:] .= 1.0im
# N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
# P0 = Diagonal(N)
# print("P0 ", P0, "\n")

diagonal = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3)
rand!(diagonal)
P = Diagonal(diagonal)
print("P ", P, "\n")

# Let's define some values used throughout the program.
# chi coefficient
chi_coeff = 3.0 + 0.001im
# inverse chi coefficient
chi_inv_coeff = 1/chi_coeff 
chi_inv_coeff_dag = conj(chi_inv_coeff)

# alpha is a parameter that we will play with to get a positive smallest 
# eigenvalue of our system 
alpha = 1e-3


# tol = 1e-3 # Loose tolerance 
# tol = 1e-6 # Tight tolerance 

tol_MGS = 1.0e-6
tol_conv = 1.0e-6

# (cellsA[1]*cellsA[2]*cellsA[3]*3,1)
# dims = size(opt)

restartLoopDim = 2
innerLoopDim = 24
# numberRestartVals = 1
dims = Int(cellsA[1]*cellsA[2]*cellsA[3]*3)


"""
Basic function call

Here's the function call:
jacDavRitzHarm_basic(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,alpha,
	innerLoopDim::Integer,dims::Integer,tol_MGS::Float64,tol_conv::Float64)::Float64
"""
fct_call_basic = jacDavRitzHarm_basic(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,alpha,
	innerLoopDim,dims,tol_MGS,tol_conv)

"""
Restart function call

Here's the function call:
jacDavRitzHarm_restart(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
	P,alpha,innerLoopDim::Integer,restartDim::Integer,dims::Integer,
	tol_MGS::Float64,tol_conv::Float64)::Float64
"""

# fct_call_restart = jacDavRitzHarm_restart(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
# 	P,alpha,restartLoopDim,innerLoopDim,dims,tol_MGS,tol_conv)

print("The smallest eigenvalue for the basic program is ", fct_call_basic, "\n")


# trgBasis = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 
# 	cellsA[1]*cellsA[2]*cellsA[3]*3)
# srcBasis = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 
# 	cellsA[1]*cellsA[2]*cellsA[3]*3)
# kMat = zeros(ComplexF64, cellsA[1]*cellsA[2]*cellsA[3]*3, 
# 	cellsA[1]*cellsA[2]*cellsA[3]*3)

# vecDim = cellsA[1]*cellsA[2]*cellsA[3]*3
# repDim = cellsA[1]*cellsA[2]*cellsA[3]*3


# trgBasis = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 
# 	cellsA[1]*cellsA[2]*cellsA[3]*3)
# srcBasis = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 
# 	cellsA[1]*cellsA[2]*cellsA[3]*3)
# kMat = zeros(ComplexF64, cellsA[1]*cellsA[2]*cellsA[3]*3, 
# 	cellsA[1]*cellsA[2]*cellsA[3]*3)
# vecDim = cellsA[1]*cellsA[2]*cellsA[3]*3
# repDim = cellsA[1]*cellsA[2]*cellsA[3]*3



# fct_call = jacDavRitzHarm_restart(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
# 	P,alpha,trgBasis,srcBasis, kMat, restartLoopDim,innerLoopDim,
# 	numberRestartVals,tol)

