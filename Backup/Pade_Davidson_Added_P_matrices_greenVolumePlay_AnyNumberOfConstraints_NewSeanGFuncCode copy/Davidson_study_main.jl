using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, bfgs_power_iteration_asym_only, dual_asym_only, gmres,
phys_setup, opt_setup, Davidson_Operator_HarmonicRitz_module_May16, Plots

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
# cellsA = [2,2,2]
"""
cellsA is defined below and will be varied so that the operator size is varied.
"""
cellsB = [1, 1, 1]
# Edge lengths of a cell relative to the wavelength. 
scaleA = (0.1, 0.1, 0.1)
scaleB = (0.2, 0.2, 0.2)
# Center position of the volume. 
coordA = (0.0, 0.0, 0.0)
coordB = (0.0, 0.0, 1.0)

# Let's define some values used throughout the program.
# chi coefficient
chi_coeff = 3.0 + 0.001im
# inverse chi coefficient
chi_inv_coeff = 1/chi_coeff 
chi_inv_coeff_dag = conj(chi_inv_coeff)

# alpha is a parameter that we will play with to get a positive smallest 
# eigenvalue of our system 
alpha = 1e-6
innerLoopDim = 2
restartDim = 1
tol_MGS = 1.0e-12
tol_conv = 1.0e-3
tol_eigval = 1.0e-6
tol_bicgstab = 1e-6
tol_bicgstab_vals = [1e-2,1e-3] # ,1e-4,1e-5,1e-6
# cellsA = [2,1,1]
cellsA_vals = [[2,2,1],[2,1,1],[2,2,1],[2,2,2]] # 
# print("length(cellsA_vals) ",length(cellsA_vals),"\n")
solver_method = "bicgstab" # "direct"

# cellsA = cellsA_vals[1]
# # Green function creation 
# G_call = G_create(cellsA,cellsB,scaleA,scaleB,coordA,coordB)
# gMemSlfN = G_call[1]
# gMemSlfA = G_call[2]
# gMemExtN = G_call[3]
# # P matrix creation 
# diagonal = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3)
# rand!(diagonal)
# P = Diagonal(diagonal)
# print("P ", P, "\n")

# dims = cellsA[1]*cellsA[2]*cellsA[3]*3
# bCoeffs1 = Vector{ComplexF64}(undef, dims)
# bCoeffs2 = Vector{ComplexF64}(undef, dims)
# trgBasis = Array{ComplexF64}(undef, dims, dims)
# srcBasis = Array{ComplexF64}(undef, dims, dims)
# kMat = zeros(ComplexF64, dims, dims)

# print("Start of basic program \n")
# eigval_basic,nb_it_basic,nb_it_solve_basic  = 
# 	jacDavRitzHarm_basic(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,alpha,
# 		innerLoopDim,dims,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,
# 			solver_method)
# print("Start of restart program \n")
# eigval_restart,nb_it_restart,nb_it_solve_restart  = 
# 	jacDavRitzHarm_restart(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
# 		P,alpha,innerLoopDim,restartDim,dims,tol_MGS,tol_conv,tol_eigval,
# 		tol_bicgstab,solver_method)

# print("Basic program results \n")
# print("eigval_basic ", eigval_basic, "\n")
# print("nb_it_basic ", nb_it_basic, "\n")
# print("nb_it_solve_basic ", nb_it_solve_basic, "\n")

# print("Restart program results \n")
# print("eigval_restart ", eigval_restart, "\n")
# print("nb_it_restart ", nb_it_restart, "\n")
# print("nb_it_solve_restart ", nb_it_solve_restart, "\n")

# Start 
# """
# 1. 
# """
tol_bicgstab= tol_bicgstab_vals[1]
first_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(cellsA_vals))
first_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(cellsA_vals))
first_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(cellsA_vals))
first_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(cellsA_vals))
first_bicgstab_tol_nb_it_bicgstab_solve_basic = Vector{Float64}(undef, length(cellsA_vals))
first_bicgstab_tol_nb_it_bicgstab_solve_restart = Vector{Float64}(undef, length(cellsA_vals))
loop_nb = 1
# @threads for index = 1:length(cellsA_vals)
for index = 1:length(cellsA_vals)
	print("index ", index, "\n")
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	sum_nb_it_solve_basic = 0.0
	sum_nb_it_solve_restart = 0.0
	for i = 1:loop_nb
		# Define test volume, all lengths are defined relative to the wavelength. 
		# Number of cells in the volume. 
		cellsA = cellsA_vals[index]
		# Green function creation 
		G_call = G_create(cellsA,cellsB,scaleA,scaleB,coordA,coordB)
		gMemSlfN = G_call[1]
		gMemSlfA = G_call[2]
		gMemExtN = G_call[3]
		# P matrix creation 
		diagonal = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3)
		rand!(diagonal)
		P = Diagonal(diagonal)
		print("P ", P, "\n")
		
		dims = cellsA[1]*cellsA[2]*cellsA[3]*3
		bCoeffs1 = Vector{ComplexF64}(undef, dims)
		bCoeffs2 = Vector{ComplexF64}(undef, dims)
		trgBasis = Array{ComplexF64}(undef, dims, dims)
		srcBasis = Array{ComplexF64}(undef, dims, dims)
		kMat = zeros(ComplexF64, dims, dims)

		print("Start of basic program \n")
		eigval_basic,nb_it_basic,nb_it_solve_basic  = 
			jacDavRitzHarm_basic(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,alpha,
				innerLoopDim,dims,tol_MGS,tol_conv,tol_eigval,tol_bicgstab_vals[index],
					solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart,nb_it_solve_restart  = 
			jacDavRitzHarm_restart(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
				P,alpha,innerLoopDim,restartDim,dims,tol_MGS,tol_conv,tol_eigval,
				tol_bicgstab_vals[index],solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
		sum_nb_it_solve_basic += nb_it_solve_basic
		sum_nb_it_solve_restart += nb_it_restart
	end
	first_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	first_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	first_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	first_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
	first_bicgstab_tol_nb_it_bicgstab_solve_basic[index] = sum_nb_it_solve_basic/loop_nb
	first_bicgstab_tol_nb_it_bicgstab_solve_restart[index] = sum_nb_it_solve_restart/loop_nb
end
print("first_bicgstab_tol_eigvals_basic ", first_bicgstab_tol_eigvals_basic, "\n")
print("first_bicgstab_tol_nb_it_basic ", first_bicgstab_tol_nb_it_basic, "\n")
print("first_bicgstab_tol_eigvals_restart ", first_bicgstab_tol_eigvals_restart, "\n")
print("first_bicgstab_tol_nb_it_restart  ", first_bicgstab_tol_nb_it_restart , "\n")


# # Define test volume, all lengths are defined relative to the wavelength. 
# # Number of cells in the volume. 
# cellsA = [2,2,2]
# # Green function creation 
# G_call = G_create(cellsA,cellsB,scaleA,scaleB,coordA,coordB)
# gMemSlfN = G_call[1]
# gMemSlfA = G_call[2]
# gMemExtN = G_call[3]
# # P matrix creation 
# diagonal = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3)
# rand!(diagonal)
# P = Diagonal(diagonal)
# print("P ", P, "\n")
# dims = Int(cellsA[1]*cellsA[2]*cellsA[3]*3)
# innerLoopDim = Int(dims/2)
# restartDim = Int(dims/4)


# """
# Basic function call

# Here's the function call:
# jacDavRitzHarm_basic(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,alpha,
# 	innerLoopDim::Integer,dims::Integer,tol_MGS::Float64,tol_conv::Float64,
# 		tol_eigval::Float64, tol_bicgstab::Float64,solver_method)
# """
# fct_call_basic = jacDavRitzHarm_basic(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,alpha,
# 	innerLoopDim,dims,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,solver_method)

# """
# Restart function call

# Here's the function call:
# jacDavRitzHarm_restart(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
# P,alpha,innerLoopDim::Integer,restartDim::Integer,dims::Integer,
# tol_MGS::Float64,tol_conv::Float64,tol_eigval::Float64, tol_bicgstab::Float64,solver_method)
# """

# fct_call_restart = jacDavRitzHarm_restart(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
# 	P,alpha,innerLoopDim,restartDim,dims,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,solver_method)

# print("The smallest eigenvalue for the basic program is ", fct_call_basic, "\n")
# print("The smallest eigenvalue for the restart program is ", fct_call_restart, "\n")


