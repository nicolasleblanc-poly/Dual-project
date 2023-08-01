using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, bfgs_power_iteration_asym_only, dual_asym_only, gmres,
phys_setup, opt_setup

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
cellsA = [2,2,2]
cellsB = [1, 1, 1]
# Edge lengths of a cell relative to the wavelength. 
scaleA = (0.1, 0.1, 0.1)
scaleB = (0.2, 0.2, 0.2)
# Center position of the volume. 
coordA = (0.0, 0.0, 0.0)
coordB = (0.0, 0.0, 1.0)

# Green function creation 
G_call = G_create(cellsA,cellsB,scaleA,scaleB,coordA,coordB)
gMemSlfN = G_call[1]
gMemSlfA = G_call[2]
gMemExtN = G_call[3]

# Initial field creation 
ei = ei_create(gMemExtN, cellsA, cellsB)
print("ei ", ei, "\n")

# Let's define some values used throughout the program.
# chi coefficient
chi_coeff = 3.0 + 0.01im
# inverse chi coefficient
chi_inv_coeff = 1/chi_coeff 
chi_inv_coeff_dag = conj(chi_inv_coeff)

# Let's get the Ps 
# These values are hard-coded for now 
nb_complex_P = 0 + 1  # +1 for the number of complex constraint because the 0th constraint 
# 4
nb_real_P =  0 # 4
P = Ps(cellsA, nb_complex_P, nb_real_P)

# Let's generate some random starting Lagrange multiplier values 
multipliers = L_mults(nb_complex_P,nb_real_P)
# +1 for the number of complex constraint because the 0th constraint 
# is the asym only constraint that is the complex identity matrix 
l = multipliers[1]
l2 = multipliers[2]
print("l ", l, "\n")
print("l2 ", l2, "\n")

# Call the Lagrange multiplier optimizer: 
# This is the code for the main function call using bfgs with the power iteration
# method to solve for the Lagrange multiplier and gmres to solve for |T>.
# # Start 
bfgs = BFGS_fakeS_with_restart_pi(gMemSlfN,gMemSlfA,l,l2,dual,P,chi_inv_coeff,ei,
cellsA,validityfunc,power_iteration_second_evaluation)
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


# ITEM call code 
# item = ITEM(gMemSlfN, gMemSlfA,l,dual,P,chi_inv_coeff,ei,cellsA,validityfunc) 
# print("dof ", item[1], "\n")
# print("grad ", item[2], "\n")
# print("dualval ", item[3], "\n")
# print("objval ", item[4], "\n")