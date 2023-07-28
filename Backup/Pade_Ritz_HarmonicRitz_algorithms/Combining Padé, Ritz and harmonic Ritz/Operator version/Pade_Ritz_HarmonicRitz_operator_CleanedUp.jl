module Pade_Ritz_HarmonicRitz_operator_CleanedUp

using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
    Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
        MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, phys_setup,
            product, gmres, Plots,Restart_Ritz_jacobiDavidson_operator,
                Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson_operator,
                    PadeForRitz_GreenOperator_Code,gmres, JLD2, FileIO, Serialization

function eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
    chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
        tol_bicgstab)
    # Solve for extremal eigenvalue 
    ritz_eigval_restart = jacDavRitz_restart(alpha_0,
        alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,
            restartDim,tol_MGS,tol_conv, tol_eigval,
                tol_bicgstab)
    print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
    # Solve for eigenvalue closest to 0
    harmonic_ritz_eigval_restart = 
        jacDavRitzHarm_restart(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,
            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
                tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
    print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
    return ritz_eigval_restart,harmonic_ritz_eigval_restart
end 

function eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
    gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
        tol_eigval,tol_bicgstab,max_nb_it)
    # Test to see what happens when we shift xi (and therefore 
    # the eigenvalues by )
    xi_test = xi-ritz_eigval_restart
    # Extreme and closest to 0 eigenvalue solve
    print("Test Ritz and harmonic Ritz eigenvalues \n")
    ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test=eigval_solve(alpha_0,alpha_1,xi_test,P0,gMemSlfN,gMemSlfA,cellsA,
            chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                tol_bicgstab)
    # Calculate the shift in the eigenvalues. This is calcualte to check
    # if the smallest extremal eigenvalue is largest than the ritz_eigval_restart,
    # which would indicate the is a unknown negative eigenvalue. 
    # eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
    return ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test
end 


function xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
        tol_conv,tol_eigval,tol_bicgstab,xi,P0)

    """
    Solve for extremum and closest to 0 eigvals for initial xi value (often 0)
    """
    ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
        chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
            tol_bicgstab)
    print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
    print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")

    # for iterations = 1:10
    if ritz_eigval_restart < 0
    """
    1. We want to increase xi until ritz_eigval_restart > 0
    """
        for it = 1:1000
            if ritz_eigval_restart > 0
                """
                1.2 Check if harmonic_ritz_eigval_restart > 0 like we want 
                """
                if harmonic_ritz_eigval_restart < 0
                    """
                    1.2.1 Increase xi until harmonic_ritz_eigval_restart > 0
                    """
                    for it = 1:1000
                        if harmonic_ritz_eigval_restart > 0 
                            """
                            Double check that harmonic_ritz_eigval_restart > 0 before breaking 
                            """
                            print("Double check that there isn't any negative eigenvalues we don't know about 1 \n")
                            ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                                chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                    tol_bicgstab)

                            # Test to see what happens when we shift xi (and therefore 
                            # the eigenvalues by )                            
                            ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
                                gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                    tol_eigval,tol_bicgstab,max_nb_it)

                            if abs(ritz_eigval_restart_test) > abs(ritz_eigval_restart)
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
                                # Matrix
                                # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                                # trueEig_A = eigen(A)
                                # A_smallest_eigval = trueEig_A.values[1]
                                # print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                                # print("All of A eigvals with old xi ", trueEig_A.values, "\n")
                                print("old xi ", xi, "\n")
                                print("There was a negative eigenvalue we didn't know about 1 \n")
                                print("old ritz_eigval_restart ", ritz_eigval_restart, "\n")
                                print("old harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
                                # Calculate the unknown negative eigenvalue 
                                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                                print("unk_neg_eigval ", unk_neg_eigval, "\n")
                                # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                                xi += abs(unk_neg_eigval)
                                print("new xi ", xi, "\n")
                                # Matrix
                                # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                                # trueEig_A = eigen(A)
                                # A_smallest_eigval = trueEig_A.values[1]
                                # print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                                # print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                                # Restart the function with the new xi value 
                                # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                                #     tol_conv,tol_eigval,tol_bicgstab,xi_test,D,P1) 
                                xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
                                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                        tol_conv,tol_eigval,tol_bicgstab,xi,P0)
                            else 
                            """
                            Essentially if 
                                abs(harmonic_ritz_eigval_restart_test) < abs(harmonic_ritz_eigval_restart).
                            This means that there wasn't a negative eigval we didn't 
                            know about since we pretty much just calculated 
                            harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                            for more reasonning. 
                            """
                                print("Converged off eigshift 1 \n")
                                print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                                print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                                print("xi ", xi, "\n")
                                # Matrix
                                # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                                # trueEig_A = eigen(A)
                                # A_smallest_eigval = trueEig_A.values[1]
                                # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                                return xi 
                            end
                        else 
                        """
                        Essentially if harmonic_ritz_eigval_restart < 0.
                        Increase xi until harmonic_ritz_eigval_restart > 0.
                        """
                            # Increase the xi value by 10 (arbitrary value)
                            # xi += 10
                            xi += abs(ritz_eigval_restart)

                            print("Increased xi 1 \n")

                            ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                                chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                    tol_bicgstab)           
                        end
                    end
                    print("Went through all first loop iterations \n")
                else 
                """
                Essentially if harmonic_ritz_eigval_restart > 0
                """
                    ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                        chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                            tol_bicgstab)
                            
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues by )
                    ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                        chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                            tol_bicgstab)

                    if abs(ritz_eigval_restart_test) > abs(ritz_eigval_restart)
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
                        # Matrix
                        # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        # trueEig_A = eigen(A)
                        # A_smallest_eigval = trueEig_A.values[1]
                        # print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                        # print("All of A eigvals with old xi ", trueEig_A.values, "\n")
                        # print("old xi ", xi, "\n")
                        # print("There was a negative eigenvalue we didn't know about 2 \n")
                        # Calculate the unknown negative eigenvalue 
                        print("old xi ", xi, "\n")
                        print("old ritz_eigval_restart ", ritz_eigval_restart, "\n")
                        print("old harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
                        unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                        print("unk_neg_eigval ", unk_neg_eigval, "\n")
                        # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                        xi += abs(unk_neg_eigval)
                        print("new xi ", xi, "\n")
                        # Matrix
                        # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        # trueEig_A = eigen(A)
                        # A_smallest_eigval = trueEig_A.values[1]
                        # print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                        # print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                        # Restart the function with the new xi value 
                        # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                        #     tol_conv,tol_eigval,tol_bicgstab,xi_test,D,P1) 
                        xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
                            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                tol_conv,tol_eigval,tol_bicgstab,xi,P0)
                    else 
                    """
                    Essentially if 
                        abs(ritz_eigval_restart_test) < abs(ritz_eigval_restart).
                    This means that there wasn't a negative eigval we didn't 
                    know about since we pretty much just calculated 
                    harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                    for more reasonning. 
                    """
                        print("Converged off eigshift 2 \n")
                        print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                        print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                        print("xi ", xi, "\n")
                        # Matrix
                        # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        # trueEig_A = eigen(A)
                        # A_smallest_eigval = trueEig_A.values[1]
                        # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                        # print("All of A eigvals ", trueEig_A.values, "\n")
                        return xi 
                    end 
                end
            else # Essentially if ritz_eigval_restart < 0
                for j = 1:1000
                    if ritz_eigval_restart > 0
                        # break
                        print("Double check that there isn't any negative eigenvalues we don't know about 2 \n")
                        
                        ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                            chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                tol_bicgstab)
                        
                        # Test to see what happens when we shift xi (and therefore 
                        # the eigenvalues by )
                        ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
                            gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                tol_eigval,tol_bicgstab,max_nb_it)

                        if abs(ritz_eigval_restart_test) > abs(ritz_eigval_restart)
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
                            # Matrix
                            # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # trueEig_A = eigen(A)
                            # A_smallest_eigval = trueEig_A.values[1]
                            # print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                            # print("All of A eigvals with old xi ", trueEig_A.values, "\n")
                            print("old xi ", xi, "\n")
                            print("There was a negative eigenvalue we didn't know about 3 \n")
                            print("old ritz_eigval_restart ", ritz_eigval_restart, "\n")
                            print("old harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
                            # Calculate the unknown negative eigenvalue 
                            unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                            print("unk_neg_eigval ", unk_neg_eigval, "\n")
                            # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                            xi += abs(unk_neg_eigval)
                            print("new xi ", xi, "\n")
                            # Matrix
                            # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # trueEig_A = eigen(A)
                            # A_smallest_eigval = trueEig_A.values[1]
                            # print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                            # print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                            # Restart the function with the new xi value 
                            # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                            #     tol_conv,tol_eigval,tol_bicgstab,xi_test,D,P1) 
                            xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
                                cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                    tol_conv,tol_eigval,tol_bicgstab,xi,P0)
                        else 
                        """
                        Essentially if 
                            abs(ritz_eigval_restart_test) < abs(ritz_eigval_restart).
                        This means that there wasn't a negative eigval we didn't 
                        know about since we pretty much just calculated 
                        harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                        for more reasonning. 
                        """
                            print("Converged off eigshift 3 \n")
                            print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                            print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                            # Matrix
                            print("xi ", xi, "\n")
                            # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # trueEig_A = eigen(A)
                            # A_smallest_eigval = trueEig_A.values[1]
                            # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                            return xi 
                        end
                    else # ritz_eigval_restart < 0
                        # Increase the xi value by 10 (arbitrary value)
                        # xi += 10
                        xi += abs(ritz_eigval_restart)
                        print("Increased xi 2 \n")

                        ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                            chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                tol_bicgstab)
                    end 
                end
                print("Went through all second loop iterations \n")
            end
        end
    
    else 
    """
    This is the case where ritz_eigval_restart > 0
    """
        if harmonic_ritz_eigval_restart < 0
            for it = 1:1000
                if harmonic_ritz_eigval_restart > 0
                """
                Double check that harmonic_ritz_eigval_restart > 0 before breaking 
                """
                    print("Double check that there isn't any negative eigenvalues we don't know about 3 \n")
                    ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                        chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                            tol_bicgstab)
                        
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues by )
                    ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
                        gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                            tol_eigval,tol_bicgstab,max_nb_it)

                    if abs(ritz_eigval_restart_test) > abs(ritz_eigval_restart)
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
                        # Matrix
                        # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        # trueEig_A = eigen(A)
                        # A_smallest_eigval = trueEig_A.values[1]
                        # print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                        # print("All of A eigvals with old xi ", trueEig_A.values, "\n")

                        print("There was a negative eigenvalue we didn't know about 4 \n")
                        print("old xi ", xi, "\n")
                        print("old ritz_eigval_restart ", ritz_eigval_restart, "\n")
                        print("old harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
                        # Calculate the unknown negative eigenvalue 
                        unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                        print("unk_neg_eigval ", unk_neg_eigval, "\n")
                        # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                        xi += abs(unk_neg_eigval)
                        print("new xi ", xi, "\n")
                        # Matrix
                        # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        # trueEig_A = eigen(A)
                        # A_smallest_eigval = trueEig_A.values[1]
                        # print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                        # print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                        # Restart the function with the new xi value 
                        # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                        #     tol_conv,tol_eigval,tol_bicgstab,xi_test,D,P1) 
                        xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
                            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                tol_conv,tol_eigval,tol_bicgstab,xi,P0)
                    else 
                    """
                    Essentially if 
                        abs(ritz_eigval_restart_test) < abs(ritz_eigval_restart).
                    This means that there wasn't a negative eigval we didn't 
                    know about since we pretty much just calculated 
                    harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                    for more reasonning. 
                    """
                        print("Converged off eigshift 4 \n")
                        print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                        print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                        # Matrix
                        print("xi ", xi, "\n")
                        # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        # trueEig_A = eigen(A)
                        # A_smallest_eigval = trueEig_A.values[1]
                        # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                        return xi 
                    end
                else # Essentially if harmonic_ritz_eigval_restart < 0
                    for j = 1:1000
                        if ritz_eigval_restart > 0
                            # break
                            print("Double check that there isn't any negative eigenvalues we don't know about 4 \n")
                            
                            ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                                chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                    tol_bicgstab)
                            
                            # Test to see what happens when we shift xi (and therefore 
                            # the eigenvalues by )
                            ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
                                gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                    tol_eigval,tol_bicgstab,max_nb_it)

                            if abs(ritz_eigval_restart_test) > abs(ritz_eigval_restart)
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
                                # Matrix
                                # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                                # trueEig_A = eigen(A)
                                # A_smallest_eigval = trueEig_A.values[1]
                                # print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                                # print("All of A eigvals with old xi ", trueEig_A.values, "\n")
                                print("old xi ", xi, "\n")
                                print("There was a negative eigenvalue we didn't know about 5 \n")
                                print("old ritz_eigval_restart ", ritz_eigval_restart, "\n")
                                print("old harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
                                # Calculate the unknown negative eigenvalue 
                                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                                print("unk_neg_eigval ", unk_neg_eigval, "\n")
                                # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                                xi += abs(unk_neg_eigval)
                                print("new xi ", xi, "\n")
                                # Matrix
                                # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                                # trueEig_A = eigen(A)
                                # A_smallest_eigval = trueEig_A.values[1]
                                # print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                                # print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                                # Restart the function with the new xi value 
                                # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                                #     tol_conv,tol_eigval,tol_bicgstab,xi_test,D,P1)  
                                xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
                                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                        tol_conv,tol_eigval,tol_bicgstab,xi,P0)
                            else 
                            """
                            Essentially if 
                                abs(ritz_eigval_restart_test) < abs(ritz_eigval_restart).
                            This means that there wasn't a negative eigval we didn't 
                            know about since we pretty much just calculated 
                            harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                            for more reasonning. 
                            """
                                print("Converged off eigshift 5 \n")
                                print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                                print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                                # Matrix
                                print("xi ", xi, "\n")
                                # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                                # trueEig_A = eigen(A)
                                # A_smallest_eigval = trueEig_A.values[1]
                                # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                                return xi 
                            end
                        else # ritz_eigval_restart < 0
                            # Increase the xi value by 10 (arbitrary value)
                            # xi += 10
                            xi += abs(ritz_eigval_restart)
                            print("Increased xi 3 \n")

                            ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                                chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                    tol_bicgstab)
                        end 
                    end
                end
                print("Went through all third loop iterations \n")
                # Matrix
                # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                # trueEig_A = eigen(A)
                # A_smallest_eigval = trueEig_A.values[1]
                # print("A_smallest_eigval end of third loop ", A_smallest_eigval, "\n")
                # print("All of A eigvals end of third loop ", trueEig_A.values, "\n")
                print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
            end
            print("Went through all fourth loop iterations \n")
        else 
        """
        Essentially if harmonic_ritz_eigval_restart > 0
        """
            print("Double check that there isn't any negative eigenvalues we don't know about 4 \n")
            
            ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,P0,gMemSlfN,gMemSlfA,cellsA,
                chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                    tol_bicgstab)
            
            # Test to see what happens when we shift xi (and therefore 
            # the eigenvalues by )
            ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_test_calcs(alpha_0,alpha_1,ritz_eigval_restart,P0,gMemSlfN,
                gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                    tol_eigval,tol_bicgstab,max_nb_it)

            if abs(ritz_eigval_restart_test) > abs(ritz_eigval_restart)
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
                # Matrix
                # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                # trueEig_A = eigen(A)
                # A_smallest_eigval = trueEig_A.values[1]
                # print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                # print("All of A eigvals with old xi ", trueEig_A.values, "\n")
                print("old xi ", xi, "\n")
                print("There was a negative eigenvalue we didn't know about 6 \n")
                print("old ritz_eigval_restart ", ritz_eigval_restart, "\n")
                print("old harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
                # Calculate the unknown negative eigenvalue 
                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                print("unk_neg_eigval ", unk_neg_eigval, "\n")
                # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                xi += abs(unk_neg_eigval)
                print("new xi ", xi, "\n")
                # Matrix
                # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                # trueEig_A = eigen(A)
                # A_smallest_eigval = trueEig_A.values[1]
                # print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                # print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                # Restart the function with the new xi value 
                # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                #     tol_conv,tol_eigval,tol_bicgstab,xi_test,D,P1)  
                xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                        tol_conv,tol_eigval,tol_bicgstab,xi,P0)
            else 
            """
            Essentially if 
                abs(ritz_eigval_restart_test) < abs(ritz_eigval_restart).
            This means that there wasn't a negative eigval we didn't 
            know about since we pretty much just calculated 
            harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
            for more reasonning. 
            """
                print("Converged off eigshift 6 \n")
                print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                print("xi ", xi, "\n")
                # Matrix
                # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                # trueEig_A = eigen(A)
                # A_smallest_eigval = trueEig_A.values[1]
                # print("A_smallest_eigval ", A_smallest_eigval, "\n")
                return xi 
            end
        end
    end
    # end
    # print("Didn't converge. Went through all possible iterations of xi_update. \n ")
    # return xi
end

""""
Initial parameter values
"""
# xi = -2.0 # Temporary value
xi = 1e-4
alpha_0 = 1
alpha_1 = 1
# sz = 15
innerLoopDim = 3
restartDim = 1
tol_MGS = 1.0e-12
tol_conv = 1.0e-3
tol_eigval = 1.0e-3
tol_bicgstab = 1e-3
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


"""
When the different parts of the Green function operator for a chosen size 
    haven't been created. 
"""
# Green function creation 
G_call = G_create(cellsA,cellsB,scaleA,scaleB,coordA,coordB)
gMemSlfN = G_call[1]
gMemSlfA = G_call[2]
gMemExtN = G_call[3]

"""
Need to run this everytime the terminal is restarted
"""
# # IObuffer function to convert object to byte streams
# io_gMemSlfN = IOBuffer();
# io_gMemSlfA = IOBuffer();
# io_gMemExtN = IOBuffer();
# # Serialize function takes stream and value as parameters
# serialize(io_gMemSlfN, gMemSlfN)
# serialize(io_gMemSlfA, gMemSlfA)
# serialize(io_gMemExtN, gMemExtN)
# take! Function fetches IOBUffer contents as Byte array and resets  
# Deserialize function takes stream as parameter
# gMemSlfN = deserialize(IOBuffer(take!(io_gMemSlfN)))
# gMemSlfA = deserialize(IOBuffer(take!(io_gMemSlfA)))
# gMemExtN = deserialize(IOBuffer(take!(io_gMemExtN)))


"""
When the different parts of the Green function operator for a chosen size 
    have been created. 
Instead of returning something, this function will write the outputs to a 
    serialized file. 
"""
# # Create a file for gMemSlfN
# gMemSlfN_file = File(format"JLD2", "/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/Operator/gMemSlfN.jld2")
# # Save data into the file
# save(gMemSlfN_file, "gMemSlfN",gMemSlfN)

# # Create a file for gMemSlfN
# gMemSlfA_file = File(format"JLD2", "/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/Operator/gMemSlfA.jld2")
# # Save data into the file
# save(gMemSlfA_file, "gMemSlfA", gMemSlfA)

# # Create a file for gMemSlfN
# gMemExtN_file = File(format"JLD2", "/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/Operator/gMemExtN.jld2")
# # Save data into the file
# save(gMemExtN_file, "gMemExtN" ,gMemExtN)

# # Load the files
# data_gMemSlfN = load(File(format"JLD2", "/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/Operator/gMemSlfN.jld2"))
# data_gMemSlfA = load(File(format"JLD2", "/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/Operator/gMemSlfA.jld2"))
# data_gMemExtN = load(File(format"JLD2", "/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/Operator/gMemExtN.jld2"))

# gMemSlfN = data_gMemSlfN["gMemSlfN"]
# gMemSlfA = data_gMemSlfA["gMemSlfA"]
# gMemExtN = data_gMemExtN["gMemExtN"]

# print("gMemSlfN ", data_gMemSlfN["gMemSlfN"], "\n")

# P matrix creation 
diagonal = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3)
rand!(diagonal)
P = Diagonal(diagonal)
print("P ", P, "\n")

dim = cellsA[1]*cellsA[2]*cellsA[3]*3
# RND vector
s = Vector{ComplexF64}(undef,dim)
rand!(s)


P0 = I

# xi_first_eval = 1e-4
xi_first_eval = 0.0
plot_surrogate_func(alpha_0,alpha_1,P0,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P,Int(round(xi_first_eval))+30,s,1)


# xi = xi_update(alpha_0,alpha_1,data_gMemSlfN["gMemSlfN"],data_gMemSlfA["gMemSlfA"],
#     cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
#         tol_conv,tol_eigval,tol_bicgstab,xi,P0)
xi = xi_update(alpha_0,alpha_1,gMemSlfN,gMemSlfA,
    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
        tol_conv,tol_eigval,tol_bicgstab,xi,P0)
print("xi such that smallest eigval of A is positive ", xi, "\n")
surr_func_xi = surrogate_function(alpha_0,alpha_1,P0,xi,cellsA,gMemSlfN,
    gMemSlfA,chi_inv_coeff,P,s)[1]
print("surr_func_xi_first_update: ", surr_func_xi, "\n")

plot_surrogate_func(alpha_0,alpha_1,P0,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P,Int(round(xi))+30,s,1)


ans,surrogate_value = root_solve(alpha_0,alpha_1,P0,xi,cellsA,gMemSlfN,gMemSlfA,
    chi_inv_coeff,P,s,tol_bissection)

print("The guessed root is: ", ans)
plot_surrogate_func(alpha_0,alpha_1,P0,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P,Int(round(ans))+30,s,2)


surr_func_ans = surrogate_function(alpha_0,alpha_1,P0,ans,cellsA,gMemSlfN,
    gMemSlfA,chi_inv_coeff,P,s)[1]
print("surr_func_xi_first_update: ", surr_func_ans, "\n")

end
