module main_Pade_Ritz_HarmonicRitz_matrix

using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
    Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
        MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, phys_setup,
            product, gmres, Plots,Restart_Ritz_jacobiDavidson_operator,
                Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson_operator,
                    PadeForRitz_GreenOperator_Code,gmres

function eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
    chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
        tol_bicgstab)
    # Solve for extremal eigenvalue 
    (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(alpha_0,
        alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
            restartDim,tol_MGS,tol_conv, tol_eigval,
                tol_bicgstab)
    # Solve for eigenvalue closest to 0
    (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart) = 
        jacDavRitzHarm_restart(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,
            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                tol_MGS,tol_conv,tol_eigval,tol_bicgstab)

    return ritz_eigval_restart,ritz_eigvec_restart,harmonic_ritz_eigval_restart,harmonic_ritz_eigvec_restart
end 

function eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
    gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
        tol_eigval,tol_bicgstab,max_nb_it)
    # Test to see what happens when we shift xi (and therefore 
    # the eigenvalues by )
    xi_test = xi-ritz_eigval_restart
    # Extreme and closest to 0 eigenvalue solve
    (ritz_eigval_restart_test,ritz_eigvec_restart_test,harmonic_ritz_eigval_restart_test,
        ritz_eigvec_restart_test)=eigval_solve(alpha_0,alpha_1,xi_test,P0,gMemSlfN,gMemSlfA,cellsA,
            chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                tol_bicgstab)
    # Calculate the shift in the eigenvalues. This is calcualte to check
    # if the smallest extremal eigenvalue is largest than the ritz_eigval_restart,
    # which would indicate the is a unknown negative eigenvalue. 
    eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
    return ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test,eigval_shift
end 
                    

function xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
        tol_conv,tol_eigval,tol_bicgstab,xi,P0)

    # Extreme and closest to 0 eigenvalue solve
    eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
    chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
        tol_bicgstab)

    if ritz_eigval_restart < 0
    """
    1. We want to increase xi until ritz_eigval_restart > 0
    """
        for it = 1:max_nb_it
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
                            # Extreme and closest to 0 eigenvalue solve
                            eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                                chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                    tol_bicgstab)
                                
                            # Test to see what happens when we shift xi (and therefore 
                            # the eigenvalues by.)
                            (ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test,eigval_shift)=eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
                                    gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                        tol_eigval,tol_bicgstab,max_nb_it)
                    
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
                                # Calculate the unknown negative eigenvalue 
                                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                                # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                                xi += unk_neg_eigval
                                # Restart the function with the new xi value 
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
                                return xi 
                            end
                        else 
                        """
                        Essentially if harmonic_ritz_eigval_restart < 0.
                        Increase xi until harmonic_ritz_eigval_restart > 0.
                        """
                            # Increase the xi value by 10 (arbitrary value)
                            xi += 10
                            # Extreme and closest to 0 eigenvalue solve
                            eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                            chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                tol_bicgstab)
                        end
                    end
                else 
                """
                Essentially if harmonic_ritz_eigval_restart > 0
                """
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues by.)
                    (ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test,eigval_shift)=eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
                            gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                tol_eigval,tol_bicgstab,max_nb_it)

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
                        # Calculate the unknown negative eigenvalue 
                        unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                        # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                        xi += abs(unk_neg_eigval)
                        # Restart the function with the new xi value 
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
                        return xi 
                    end 
                end
            else # Essentially if ritz_eigval_restart < 0
                for j = 1:max_nb_it
                    if ritz_eigval_restart > 0
                        break
                    else # ritz_eigval_restart < 0
                        # Increase the xi value by 10 (arbitrary value)
                        xi += 10
                        # Extreme and closest to 0 eigenvalue solve
                        eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                            chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                tol_bicgstab)
                    end 
                end
            end
        end
    
    else 
    """
    This is the case where ritz_eigval_restart > 0
    """
        if harmonic_ritz_eigval_restart < 0
            for it = 1:max_nb_it
                if harmonic_ritz_eigval_restart > 0
                """
                Double check that harmonic_ritz_eigval_restart > 0 before breaking 
                """
                    # Extreme and closest to 0 eigenvalue solve
                    eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                        chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                            tol_bicgstab)
                        
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues by.)
                    (ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test,eigval_shift)=eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
                            gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                tol_eigval,tol_bicgstab,max_nb_it)
                
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
                        # Calculate the unknown negative eigenvalue 
                        unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                        # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                        xi += abs(unk_neg_eigval)
                        # Restart the function with the new xi value 
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
                        return xi 
                    end
                else # Essentially if ritz_eigval_restart < 0
                    for j = 1:100000
                        if ritz_eigval_restart > 0
                            break
                        else # ritz_eigval_restart < 0
                            # Increase the xi value by 10 (arbitrary value)
                            xi += 10
                            # Extreme and closest to 0 eigenvalue solve
                            eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                                chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                    tol_bicgstab)
                        end 
                    end
                end
            end
        else 
        """
        Essentially if harmonic_ritz_eigval_restart > 0
        """
            # Test to see what happens when we shift xi (and therefore 
            # the eigenvalues by.)
            (ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test,eigval_shift)=eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
                    gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                        tol_eigval,tol_bicgstab,max_nb_it)
        
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
                # Calculate the unknown negative eigenvalue 
                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                xi += abs(unk_neg_eigval)
                # Restart the function with the new xi value 
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
                return xi 
            end
        end
    end
    print("Didn't converge. Went through all possible iterations. \n ")
    return xi
end

"""
Initial parameter values
"""
xi = 0.5 # Temporary value
alpha_0 = 1
alpha_1 = 1
# sz = 5
innerLoopDim = 3
restartDim = 1
tol_MGS = 1.0e-12
tol_conv = 1.0e-12
tol_eigval = 1.0e-9
tol_bicgstab = 1e-6
tol_bissection = 1e-4
max_nb_it = 1000

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

P0 = I
xi = xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
        tol_conv,tol_eigval,tol_bicgstab,xi,P0)
print("xi such that smallest eigval of A is positive ", xi, "\n")
surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,xi,cellsA,gMemSlfN,
    gMemSlfA,chi_inv_coeff,P)[1]
print("surr_func_xi_first_update: ", surr_func_xi, "\n")

plot_surrogate_func(alpha_0,alpha_1,P0,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P,Int(round(xi))+100,1)


ans,surrogate_value = root_solve(alpha_0,alpha_1,P0,xi,cellsA,gMemSlfN,gMemSlfA,
    chi_inv_coeff,P,tol_bissection,max_nb_it)
print("The guessed root is: ", ans)
plot_surrogate_func(alpha_0,alpha_1,P0,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P,Int(round(ans))+50,2)


end
