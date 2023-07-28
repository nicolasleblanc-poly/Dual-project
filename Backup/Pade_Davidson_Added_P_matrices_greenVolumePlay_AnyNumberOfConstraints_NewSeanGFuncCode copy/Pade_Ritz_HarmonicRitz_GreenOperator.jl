module Pade_Ritz_HarmonicRitz_matrix
# using LinearAlgebra,Random, Base.Threads,Plots,
#     Restart_Ritz_jacobiDavidson_operator,
#         Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson_operator,
#             PadeForRitz_GreenOperator_Code

using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
    Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
        MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, phys_setup,
            product, gmres, Plots,Restart_Ritz_jacobiDavidson_operator,
                Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson_operator,
                    PadeForRitz_GreenOperator_Code,gmres

""""
Initial parameter values
"""
xi = 0.5 # Temporary value
alpha_0 = 1
alpha_1 = 1
sz = 5
innerLoopDim = 3
restartDim = 1
tol_MGS = 1.0e-12
tol_conv = 1.0e-12
tol_eigval = 1.0e-9
tol_bicgstab = 1e-6
tol_bissection = 1e-4

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

"""
cellsA is defined below and will be varied so that the operator size is varied.
"""
cellsA = [2,2,1]
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

# P0 RND positive definite matrix
# P0 = rand([0.0,1.0],sz,sz)
# P0[:,:] = (P0 .+ transpose(P0)) ./ 2

# D = rand([0.0,0.5],sz,sz) # Just to initialize the variable
# # print("inv(D) 1 ", inv(D), "\n")
# P0 = adjoint(D)*D
# print("isposdef(P0) ", isposdef(P0), "\n")
# print("cond(P0) ", cond(P0), "\n")
# print("cond(D) ", cond(D), "\n")
# print("det(D) ", det(D), "\n")
# done = false


# function D_matrix(sz)
#     for it = 1:100000
#         D = rand([0.0,0.5],sz,sz)
#         # D[:,:] = (D .+ transpose(D)) ./ 2
#         # # print("inv(D) 2 ", inv(D), "\n")
#         P0 = adjoint(D)*D
#         # print("P0 ", P0, "\n")
#         if isposdef(P0) == false || cond(P0) > 1e6
#             D = rand([0.0,0.5],sz,sz)
#             D[:,:] = (D .+ transpose(D)) ./ 2
#             P0 = adjoint(D)*D
#             # print("P0 ", P0, "\n")
#             print("isposdef(P0) ", isposdef(P0), "\n")
#             print("cond(P0) ", cond(P0), "\n")
#             print("cond(D) ", cond(D), "\n")
#             print("det(D) ", det(D), "\n")
#         elseif  isposdef(P0) == true && 0 < cond(P0) < 1e6 && det(D) > 0 
#             print("isposdef(P0) done ",isposdef(P0),"\n")
#             return D
#         end
#     end
# end

# D = D_matrix(sz)
# print("D ", D,"\n")

# P0 = I

# # P1 RND semidefinite matrix
# P1 = rand([-2.0,2.0],sz,sz)
# P1[:,:] = (P1 .+ adjoint(P1)) ./ 2
# print("P1 ", P1, "\n")

# # print("cond(P0) ", cond(P0), "\n")
# print("cond(P1) ", cond(P1), "\n")
# # print("det(D) ", det(D), "\n")
# # print("inv(D) ", inv(D), "\n")

# # Check that P0 is positive definite and that P1 is semidefinite
# # trueEig_P0 = eigen(P0) 
# trueEig_P1 = eigen(P1) 
# # print("trueEig_P0 ", trueEig_P0.values , "\n")
# # print("trueEig_P1 ", trueEig_P1.values, "\n")
# # print("isposdef(P0) ", isposdef(P0), "\n")
# print("isposdef(P1) ", isposdef(P1), "\n")

function xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
        tol_conv,tol_eigval,tol_bicgstab,xi,P0)

    """
    Solve for extremum and closest to 0 eigvals for initial xi value (often 0)
    """
    # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
    # A = (alpha_1+xi)*I + alpha_0*P1
    # print("cond(A) ",cond(A), "\n")
    # print("A ", A, "\n")
    # trueEig_A = eigen(A)
    # A_smallest_eigval = trueEig_A.values[1]
    # print("A_smallest_eigval ", A_smallest_eigval, "\n")
    # print("All of A eigvals ", trueEig_A.values, "\n")

    # surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi)
    # print("surr_func_xi_first_update: ", surr_func_xi, "\n")

    (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(alpha_0,
        alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
            restartDim,tol_MGS,tol_conv, tol_eigval,
                tol_bicgstab)
    # jacDavRitz_restart(A,
    #     innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
    print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
    (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart) = 
        jacDavRitzHarm_restart(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,
            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
        # jacDavRitzHarm_restart(A,innerLoopDim,restartDim,tol_MGS,
        #     tol_conv,tol_eigval,tol_bicgstab)
    print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")

    shifted = false

    # for i = 1:1000
    if ritz_eigval_restart < 0
    """
    1. We want to increase xi until ritz_eigval_restart > 0
    """
        for it = 1:1000000
            if ritz_eigval_restart > 0
                """
                1.2 Check if harmonic_ritz_eigval_restart > 0 like we want 
                """
                if harmonic_ritz_eigval_restart < 0
                    """
                    1.2.1 Increase xi until harmonic_ritz_eigval_restart > 0
                    """
                    for it = 1:1000000
                        if harmonic_ritz_eigval_restart > 0 
                            """
                            Double check that harmonic_ritz_eigval_restart > 0 before breaking 
                            """
                            # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # A = (alpha_1+xi)*I + alpha_0*P1

                            # trueEig_A = eigen(A)
                            # A_smallest_eigval = trueEig_A.values[1]
                            # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                            # print("All of A eigvals ", trueEig_A.values, "\n")
        
                            (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(alpha_0,
                            alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
                                restartDim,tol_MGS,tol_conv, tol_eigval,
                                    tol_bicgstab)
                            # jacDavRitz_restart(A,
                            #     innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                            #         tol_bicgstab)
                            # print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
                            (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart)= 
                                jacDavRitzHarm_restart(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,
                                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                                        tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                            # print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n") 
                                
                                
                            # Test to see what happens when we shift xi (and therefore 
                            # the eigenvalues by )
                            xi_test = xi-ritz_eigval_restart
                            # print("xi_test 9 ", xi_test, "\n")
                            # print("xi ", xi, "\n")

                            # surr_func_xi_test = surrogate_function(alpha_0,alpha_1,P0,P1,xi_test)
                            # print("surr_func_xi_first_update_test: ", surr_func_xi_test, "\n")
                            
                            # A_test = (alpha_1+xi_test)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # A_test = (alpha_1+xi_test)*I + alpha_0*P1
                            # trueEig_A_test = eigen(A_test)
                            # A_smallest_eigval_test = trueEig_A_test.values[1]
                            # print("A_smallest_eigval_test ", A_smallest_eigval_test, "\n")
                            # print("All of A eigvals_test ", trueEig_A_test.values, "\n")
                
                            (ritz_eigval_restart_test, ritz_eigvec_restart_test) = jacDavRitz_restart(alpha_0,
                                alpha_1,xi_test,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
                                    restartDim,tol_MGS,tol_conv, tol_eigval,
                                        tol_bicgstab)
                            # jacDavRitz_restart(A_test,
                            #     innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                            # print("ritz_eigval_restart_test ", ritz_eigval_restart_test, "\n")
                            (harmonic_ritz_eigval_restart_test, harmonic_ritz_eigvec_restart_test) =
                                jacDavRitzHarm_restart(alpha_0,alpha_1,xi_test,P0,gMemSlfN,gMemSlfA,
                                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                                        tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                            # print("harmonic_ritz_eigval_restart_test ", harmonic_ritz_eigval_restart_test, "\n")
        
                            eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
                            # print("eigval_shift ", eigval_shift, "\n")
                            # print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
                            # print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
        
                            if abs(eigval_shift) > abs(ritz_eigval_restart)
                                """
                                This means that there was a negative eigenvalue that we 
                                didn't know about (aka it wasn't the extreme eigval or the
                                eigval closest to 0) since we get a value after the 
                                substraction that is largest than the value we substracted
                                by. Its like if we had a negative eigval we didn't know about
                                that is equal to -5, harmonic_ritz_eigval_restart = 1 and 
                                ritz_eigval_restart = 10. The substraction gives -15, which 
                                is bigger in absolute value than 10. If we didn't have any 
                                negative eigvals and 1 was the smallest eigal, we would get 
                                -9, which is smaller than 10 in absolute value. 
                                """
                                # xi_test -= ritz_eigval_restart
                                # print("xi ", xi, "\n")
                                # surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi_test)
                                # print("surr_func_xi_first_update: ", surr_func_xi, "\n")
        
                                shifted = true
                                # print("TBD situation where ritz_eigval_restart > 0 
                                #     and harmonic_ritz_eigval_restart > 0. 
                                #         Maybe need to shift xi by - ritz_eigval_restart 10 \n")
        
                                # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                                #     tol_conv,tol_eigval,tol_bicgstab,xi_test,P0,D)
                                
                                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                                # print("unk_neg_eigval ", unk_neg_eigval, "\n")
                                xi += unk_neg_eigval
                                # print("xi+unk_neg_eigval ", xi, "\n")
                                xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
                                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                        tol_conv,tol_eigval,tol_bicgstab,xi,P0)
                                
                            
                            else 
                            """
                            Essentially if 
                                abs(eigval_shift) < abs(harmonic_ritz_eigval_restart).
                            This means that there wasn't a negative eigval we didn't 
                            know about since we pretty much just calculated 
                            harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                            for more reasonning. 
                            """
                                # print("Converged 11 \n")
                                return xi 
                            end
                        else 
                        """
                        Essentially if harmonic_ritz_eigval_restart < 0.
                        Increase xi until harmonic_ritz_eigval_restart > 0.
                        """
                            xi += 10
                            # print("xi updated 2 ", xi, "\n")

                            # surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi)
                            # print("surr_func_xi_first_update: ", surr_func_xi, "\n")

                            # Matrix
                            # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # A = (alpha_1+xi)*I + alpha_0*P1

                            # Using Julia eigenvalue solver for now because I'm not sure 
                            # how Ritz or Hamonic Ritz can be used here.
                            # trueEig_A = eigen(A)
                            # A_smallest_eigval = trueEig_A.values[1]
                            # print("xi ", xi, "\n")
                            # print("trueEig_A ", trueEig_A.values , "\n")
                            # println("The smallest eigenvalue is 2 ", A_smallest_eigval,".")
        
                            # trueEig_A = eigen(A)
                            # A_smallest_eigval = trueEig_A.values[1]
                            # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                            # print("All of A eigvals ", trueEig_A.values, "\n")
                
                            (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(alpha_0,
                            alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
                                restartDim,tol_MGS,tol_conv, tol_eigval,
                                    tol_bicgstab)
                            # jacDavRitz_restart(A,
                            #     innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                            # print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
                            (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart) =
                                jacDavRitzHarm_restart(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,
                                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                                        tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                            # print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
                
                        end
                    end
                else 
                """
                Essentially if harmonic_ritz_eigval_restart > 0
                """

                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues by )
                    xi_test = xi-ritz_eigval_restart
                    # print("xi_test 3 ", xi_test, "\n")
                    # print("xi ", xi, "\n")

                    # surr_func_xi_test = surrogate_function(alpha_0,alpha_1,P0,P1,xi_test)
                    # print("surr_func_xi_first_update_test: ", surr_func_xi_test, "\n")

                    # A_test = (alpha_1+xi_test)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                    # A_test = (alpha_1+xi_test)*I + alpha_0*P1
                    # trueEig_A_test = eigen(A_test)
                    # A_smallest_eigval_test = trueEig_A_test.values[1]
                    # print("A_smallest_eigval ", A_smallest_eigval_test, "\n")
                    # print("All of A eigvals _test", trueEig_A_test.values, "\n")
        
                    (ritz_eigval_restart_test, ritz_eigvec_restart_test) = jacDavRitz_restart(alpha_0,
                        alpha_1,xi_test,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
                            restartDim,tol_MGS,tol_conv, tol_eigval,
                                tol_bicgstab)
                    # jacDavRitz_restart(A_test,
                    #     innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                    # print("ritz_eigval_restart_test", ritz_eigval_restart_test, "\n")
                    (harmonic_ritz_eigval_restart_test, harmonic_ritz_eigvec_restart_test) =
                        jacDavRitzHarm_restart(alpha_0,alpha_1,xi_test,P0,gMemSlfN,gMemSlfA,
                            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                                tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                    # print("harmonic_ritz_eigval_restart_test ", harmonic_ritz_eigval_restart_test, "\n")

                    eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
                    # print("eigval_shift ", eigval_shift, "\n")
                    # print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
                    # print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
                    
                    if abs(eigval_shift) > abs(ritz_eigval_restart)
                    """
                    This means that there was a negative eigenvalue that we 
                    didn't know about (aka it wasn't the extreme eigval or the
                    eigval closest to 0) since we get a value after the 
                    substraction that is largest than the value we substracted
                    by. Its like if we had a negative eigval we didn't know about
                    that is equal to -5, harmonic_ritz_eigval_restart = 1 and 
                    ritz_eigval_restart = 10. The substraction gives -15, which 
                    is bigger in absolute value than 10. If we didn't have any 
                    negative eigvals and 1 was the smallest eigal, we would get 
                    -9, which is smaller than 10 in absolute value. 
                    """
                        # xi_test -= ritz_eigval_restart
                        # print("xi ", xi, "\n")
                        # surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi_test)
                        # print("surr_func_xi_first_update: ", surr_func_xi, "\n")

                        # shifted = true
                        # print("TBD situation where ritz_eigval_restart > 0 
                        #     and harmonic_ritz_eigval_restart > 0. 
                        #         Maybe need to shift xi by - ritz_eigval_restart 4 \n")
                        # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                        #     tol_conv,tol_eigval,tol_bicgstab,xi_test,P0,D)
                        shifted = true
                        # print("TBD situation where ritz_eigval_restart > 0 
                        #     and harmonic_ritz_eigval_restart > 0. 
                        #         Maybe need to shift xi by - ritz_eigval_restart 10 \n")

                        # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                        #     tol_conv,tol_eigval,tol_bicgstab,xi_test,P0,D)
                        
                        unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                        # print("unk_neg_eigval ", unk_neg_eigval, "\n")
                        xi += abs(unk_neg_eigval)
                        # print("xi+unk_neg_eigval ", xi, "\n")

                        xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
                            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                tol_conv,tol_eigval,tol_bicgstab,xi,P0)
            
                    else 
                    """
                    Essentially if 
                        abs(eigval_shift) < abs(harmonic_ritz_eigval_restart).
                    This means that there wasn't a negative eigval we didn't 
                    know about since we pretty much just calculated 
                    harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                    for more reasonning. 
                    """
                        # print("Converged 5 \n")
                        # print("xi ", xi, "\n")
                        return xi 
                    end 

                    # xi_test -= ritz_eigval_restart
                    # print("xi ", xi, "\n")
                    # surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi_test)
                    # print("surr_func_xi_first_update: ", surr_func_xi, "\n")

                    # shifted = true
                    # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                    #     tol_conv,tol_eigval,tol_bicgstab,xi,P0,D)
        
                    # print("TBD situation where ritz_eigval_restart > 0 
                    #             and harmonic_ritz_eigval_restart > 0. 
                    #                 Maybe need to shift xi by - ritz_eigval_restart \n")
                end
                # break # Essentially doing return xi
            else # Essentially if ritz_eigval_restart < 0
                for j = 1:100000
                    if ritz_eigval_restart > 0
                        break
                    else # ritz_eigval_restart < 0
                        xi += 10
                        # print("xi increased 6 ", xi, "\n")

                        # surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi)
                        # print("surr_func_xi_first_update: ", surr_func_xi, "\n")
        
                        # Matrix
                        # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        # A = (alpha_1+xi)*I + alpha_0*P1
        
                        # Using Julia eigenvalue solver for now because I'm not sure 
                        # how Ritz or Hamonic Ritz can be used here.
                        # trueEig_A = eigen(A)
                        # A_smallest_eigval = trueEig_A.values[1]
                        # print("xi ", xi, "\n")
                        # print("trueEig_A ", trueEig_A.values , "\n")
                        # println("The smallest eigenvalue is 2 ", A_smallest_eigval,".")
            
                        # trueEig_A = eigen(A)
                        # A_smallest_eigval = trueEig_A.values[1]
                        # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                        # print("All of A eigvals ", trueEig_A.values, "\n")
        
                        (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(alpha_0,
                        alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
                            restartDim,tol_MGS,tol_conv, tol_eigval,
                                tol_bicgstab)
                        # jacDavRitz_restart(A,
                        #     innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                        # print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
                        (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart) =
                            jacDavRitzHarm_restart(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,
                                cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                                    tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                        # print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
                    end 
                end
            end
        end
    
    else 
    """
    This is the case where ritz_eigval_restart > 0
    """
        if harmonic_ritz_eigval_restart < 0
            for it = 1:1000000
                if harmonic_ritz_eigval_restart > 0
                """
                Double check that harmonic_ritz_eigval_restart > 0 before breaking 
                """
                    # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                    # A = (alpha_1+xi)*I + alpha_0*P1

                    # trueEig_A = eigen(A)
                    # A_smallest_eigval = trueEig_A.values[1]
                    # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                    # print("All of A eigvals ", trueEig_A.values, "\n")

                    (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(alpha_0,
                        alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
                            restartDim,tol_MGS,tol_conv, tol_eigval,tol_bicgstab)
                    # jacDavRitz_restart(A,
                    #     innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                    #         tol_bicgstab)
                    # print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
                    (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart)= 
                        jacDavRitzHarm_restart(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,
                            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                                tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                    # print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n") 
                        
                        
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues by )
                    xi_test = xi-ritz_eigval_restart
                    # print("xi_test 9 ", xi_test, "\n")
                    # print("xi ", xi, "\n")

                    # surr_func_xi_test = surrogate_function(alpha_0,alpha_1,P0,P1,xi_test)
                    # print("surr_func_xi_first_update_test: ", surr_func_xi_test, "\n")

                    # A_test = (alpha_1+xi_test)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                    # A_test = (alpha_1+xi_test)*I + alpha_0*P1
                    # trueEig_A_test = eigen(A_test)
                    # A_smallest_eigval_test = trueEig_A_test.values[1]
                    # print("A_smallest_eigval_test ", A_smallest_eigval_test, "\n")
                    # print("All of A eigvals_test ", trueEig_A_test.values, "\n")
        
                    (ritz_eigval_restart_test, ritz_eigvec_restart_test) = jacDavRitz_restart(alpha_0,
                        alpha_1,xi_test,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
                            restartDim,tol_MGS,tol_conv, tol_eigval,
                                tol_bicgstab)
                    # jacDavRitz_restart(A_test,
                    #     innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                    # print("ritz_eigval_restart_test ", ritz_eigval_restart_test, "\n")
                    (harmonic_ritz_eigval_restart_test, harmonic_ritz_eigvec_restart_test) =
                        jacDavRitzHarm_restart(alpha_0,alpha_1,xi_test,P0,gMemSlfN,gMemSlfA,
                            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                                tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                    # print("harmonic_ritz_eigval_restart_test ", harmonic_ritz_eigval_restart_test, "\n")

                    eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
                    # print("eigval_shift ", eigval_shift, "\n")
                    # print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
                    # print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")

                    if abs(eigval_shift) > abs(ritz_eigval_restart)
                    """
                    This means that there was a negative eigenvalue that we 
                    didn't know about (aka it wasn't the extreme eigval or the
                    eigval closest to 0) since we get a value after the 
                    substraction that is largest than the value we substracted
                    by. Its like if we had a negative eigval we didn't know about
                    that is equal to -5, harmonic_ritz_eigval_restart = 1 and 
                    ritz_eigval_restart = 10. The substraction gives -15, which 
                    is bigger in absolute value than 10. If we didn't have any 
                    negative eigvals and 1 was the smallest eigal, we would get 
                    -9, which is smaller than 10 in absolute value. 
                    """
                        # xi_test -= ritz_eigval_restart
                        # print("xi ", xi, "\n")
                        # surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi_test)
                        # print("surr_func_xi_first_update: ", surr_func_xi, "\n")

                        # shifted = true
                        # print("TBD situation where ritz_eigval_restart > 0 
                        #     and harmonic_ritz_eigval_restart > 0. 
                        #         Maybe need to shift xi by - ritz_eigval_restart 10 \n")

                        # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                        #     tol_conv,tol_eigval,tol_bicgstab,xi_test,P0,D)
                        shifted = true
                        print("TBD situation where ritz_eigval_restart > 0 
                            and harmonic_ritz_eigval_restart > 0. 
                                Maybe need to shift xi by - ritz_eigval_restart 10 \n")

                        # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                        #     tol_conv,tol_eigval,tol_bicgstab,xi_test,P0,D)
                        
                        unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                        # print("unk_neg_eigval ", unk_neg_eigval, "\n")
                        xi += abs(unk_neg_eigval)
                        # print("xi+unk_neg_eigval ", xi, "\n")

                        xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
                            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                tol_conv,tol_eigval,tol_bicgstab,xi,P0)
            
                        
                    else 
                    """
                    Essentially if 
                        abs(eigval_shift) < abs(harmonic_ritz_eigval_restart).
                    This means that there wasn't a negative eigval we didn't 
                    know about since we pretty much just calculated 
                    harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                    for more reasonning. 
                    """
                        # print("Converged 11 \n")
                        return xi 
                    end
                else # Essentially if ritz_eigval_restart < 0
                    for j = 1:100000
                        if ritz_eigval_restart > 0
                            break
                        else # ritz_eigval_restart < 0
                            xi += 10
                            # print("xi increased 12 ", xi, "\n")

                            # surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi)
                            # print("surr_func_xi_first_update: ", surr_func_xi, "\n")
            
                            # Matrix
                            # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # A = (alpha_1+xi)*I + alpha_0*P1
            
                            # Using Julia eigenvalue solver for now because I'm not sure 
                            # how Ritz or Hamonic Ritz can be used here.
                            # trueEig_A = eigen(A)
                            # A_smallest_eigval = trueEig_A.values[1]
                            # print("xi ", xi, "\n")
                            # print("trueEig_A ", trueEig_A.values , "\n")
                            # println("The smallest eigenvalue is 2 ", A_smallest_eigval,".")
                
                            # trueEig_A = eigen(A)
                            # A_smallest_eigval = trueEig_A.values[1]
                            # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                            # print("All of A eigvals ", trueEig_A.values, "\n")
            
                            (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(alpha_0,
                                alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
                                    restartDim,tol_MGS,tol_conv, tol_eigval,
                                        tol_bicgstab)
                            
                            # jacDavRitz_restart(A,
                            #     innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                            # print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
                            (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart) =
                                jacDavRitzHarm_restart(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,
                                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                                        tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                            # print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
                        end 
                    end
                end
            end
        else 
        """
        Essentially if harmonic_ritz_eigval_restart > 0
        """
            # xi -= ritz_eigval_restart
            # surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi)
            # print("surr_func_xi_first_update: ", surr_func_xi, "\n")
            # shifted = true
            # print("xi ", xi, "\n")
            # print("Recursive function call \n")
            # xi_first_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
            #     tol_conv,tol_eigval,tol_bicgstab,xi,P0,D)

            # print("TBD situation where ritz_eigval_restart > 0 
            #             and harmonic_ritz_eigval_restart > 0. 
            #                 Maybe need to shift xi by - ritz_eigval_restart \n")
            # Test to see what happens when we shift xi (and therefore 
            # the eigenvalues by )
            xi_test = xi - ritz_eigval_restart
            # print("xi_test 13 ", xi_test, "\n")
            # print("xi 13 ", xi, "\n")

            # surr_func_xi_test = surrogate_function(alpha_0,alpha_1,P0,P1,xi_test)
            # print("surr_func_xi_first_update: ", surr_func_xi_test, "\n")

            # A_test = (alpha_1+xi_test)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
            # A_test = (alpha_1+xi_test)*I + alpha_0*P1 
            # trueEig_A_test = eigen(A_test)
            # A_smallest_eigval_test = trueEig_A_test.values[1]
            # print("A_smallest_eigval_test ", A_smallest_eigval_test, "\n")
            # print("All of A eigvals_test ", trueEig_A_test.values, "\n")

            (ritz_eigval_restart_test, ritz_eigvec_restart_test) = jacDavRitz_restart(alpha_0,
                alpha_1,xi_test,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
                    restartDim,tol_MGS,tol_conv, tol_eigval,
                        tol_bicgstab)
            # jacDavRitz_restart(A_test,
            #     innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
            # print("ritz_eigval_restart_test ", ritz_eigval_restart_test, "\n")
            (harmonic_ritz_eigval_restart_test, harmonic_ritz_eigvec_restart_test) =
                jacDavRitzHarm_restart(alpha_0,alpha_1,xi_test,P0,gMemSlfN,gMemSlfA,
                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                        tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
            # print("harmonic_ritz_eigval_restart_test ", harmonic_ritz_eigval_restart_test, "\n")

            eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
            # print("eigval_shift ", eigval_shift, "\n")
            # print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
            # print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")

            if abs(eigval_shift) > abs(ritz_eigval_restart)
            """
            This means that there was a negative eigenvalue that we 
            didn't know about (aka it wasn't the extreme eigval or the
            eigval closest to 0) since we get a value after the 
            substraction that is largest than the value we substracted
            by. Its like if we had a negative eigval we didn't know about
            that is equal to -5, harmonic_ritz_eigval_restart = 1 and 
            ritz_eigval_restart = 10. The substraction gives -15, which 
            is bigger in absolute value than 10. If we didn't have any 
            negative eigvals and 1 was the smallest eigal, we would get 
            -9, which is smaller than 10 in absolute value. 
            """
                # xi_test -= ritz_eigval_restart
                # print("xi ", xi, "\n")
                # surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi_test)
                # print("surr_func_xi_first_update: ", surr_func_xi, "\n")

                # shifted = true
                # print("TBD situation where ritz_eigval_restart > 0 
                #     and harmonic_ritz_eigval_restart > 0. 
                #         Maybe need to shift xi by - ritz_eigval_restart 14 \n")
                # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                #     tol_conv,tol_eigval,tol_bicgstab,xi_test,P0,D)
                shifted = true
                # print("TBD situation where ritz_eigval_restart > 0 
                #     and harmonic_ritz_eigval_restart > 0. 
                #         Maybe need to shift xi by - ritz_eigval_restart 10 \n")

                # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                #     tol_conv,tol_eigval,tol_bicgstab,xi_test,P0,D)
                
                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                # print("unk_neg_eigval ", unk_neg_eigval, "\n")
                xi += abs(unk_neg_eigval)
                # print("xi+unk_neg_eigval ", xi, "\n")

                xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                        tol_conv,tol_eigval,tol_bicgstab,xi,P0)
                

            else 
            """
            Essentially if 
                abs(eigval_shift) < abs(harmonic_ritz_eigval_restart).
            This means that there wasn't a negative eigval we didn't 
            know about since we pretty much just calculated 
            harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
            for more reasonning. 
            """
                print("Converged 15 \n")

                print("xi ", xi, "\n")
                return xi 
            end
        end
    end
    # end
    print("Didn't converge. Went through all possible iterations. \n ")
    return xi
end

# xi = xi_first_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
#     tol_conv,tol_eigval,tol_bicgstab,xi,P0,D)
# print("xi such that smallest eigval of A is positive ", xi, "\n")
# surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,xi)
# print("surr_func_xi_first_update: ", surr_func_xi, "\n")

"""
1. Solve for extremal eigenvalue of A with xi = 0 
"""
P0 = I
xi = xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
        tol_conv,tol_eigval,tol_bicgstab,xi,P0)
print("xi such that smallest eigval of A is positive ", xi, "\n")
surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,xi,cellsA,gMemSlfN,
    gMemSlfA,chi_inv_coeff,P)[1]
print("surr_func_xi_first_update: ", surr_func_xi, "\n")

plot_surrogate_func(alpha_0,alpha_1,P0,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P,Int(round(xi))+100,1)


# max_val = 100
ans,surrogate_value = root_solve(alpha_0,alpha_1,P0,xi,cellsA,gMemSlfN,gMemSlfA,
    chi_inv_coeff,P,tol_bissection)
print("The guessed root is: ", ans)
plot_surrogate_func(alpha_0,alpha_1,P0,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P,Int(round(ans))+50,2)

# (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(alpha_0,
#     alpha_1,ans,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
#         restartDim,tol_MGS,tol_conv, tol_eigval,
#             tol_bicgstab)
# (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart) =
#     jacDavRitzHarm_restart(alpha_0,alpha_1,ans,P0,gMemSlfN,gMemSlfA,
#         cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
#             tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
# print("ritz_eigval_restart ",ritz_eigval_restart ,"\n")
# print("harmonic_ritz_eigval_restart ",harmonic_ritz_eigval_restart ,"\n")

# A = (alpha_1+ans)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
# print("A ", A, "\n")
# trueEig_A = eigen(A)
# A_smallest_eigval = trueEig_A.values[1]
# print("A_smallest_eigval ", A_smallest_eigval, "\n")
# print("All of A eigvals ", trueEig_A.values, "\n")

# surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,P1,ans)
# print("surr_func_xi : ", surr_func_xi, "\n")

end



# # Matrix
# A = (alpha_1+xi)*P0+alpha_0*P1
# # print("det(A) ", det(A),"\n")
# # print("A ", A,"\n")
# trueEig_A = eigen(A)
# A_smallest_eigval = trueEig_A.values[1]
# # print("trueEig_A ", trueEig_A.values , "\n")
# print("The smallest eigenvalue for xi first update is ", A_smallest_eigval,"\n")

# xi = 600
# # Matrix
# A = (alpha_1+xi)*P0+alpha_0*P1
# # print("det(A) ", det(A),"\n")
# # print("A ", A,"\n")
# trueEig_A = eigen(A)
# A_smallest_eigval = trueEig_A.values[1]
# # print("trueEig_A ", trueEig_A.values , "\n")
# print("The smallest eigenvalue positive surrogate func test is ", A_smallest_eigval,"\n")