"""
This module is to find a first approximate of the xi value such that all of the 
eigenvalues are positive.

Author: Nicolas Leblanc
"""

module approx_xi_solve_Ritz_HarmonicRitz_GreenOperator_sym_and_asym

using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
    Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
        MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random,
            product_sym_and_asym, gmres, Plots,Restart_Ritz_jacobiDavidson_operator_sym_and_asym,
                Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson_operator_sym_and_asym,
                    PadeForRitz_GreenOperator_Code_sym_and_asym,gmres

export xi_update

function eigval_solve(xi,l,gMemSlfN,gMemSlfA,cellsA,
    chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
        tol_bicgstab)

    # Solve for extremal eigenvalue 
    ritz_eigval_restart = jacDavRitz_restart(xi,l,gMemSlfN,gMemSlfA,cellsA,
        chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv, tol_eigval,
            tol_bicgstab)
    print("ritz_eigval_restart ", ritz_eigval_restart, "\n")
    # Solve for eigenvalue closest to 0
    harmonic_ritz_eigval_restart = jacDavRitzHarm_restart(xi,l,gMemSlfN,gMemSlfA,
        cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,
            tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
    print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart, "\n")
    
    return ritz_eigval_restart,harmonic_ritz_eigval_restart
end 

function eigval_test_calcs(xi,l,ritz_eigval_restart,gMemSlfN,
    gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
        tol_eigval,tol_bicgstab)
    """
    Test to see what happens when we shift xi, and therefore 
    the eigenvalues, by ritz_eigval_restart
    """
    xi_test = xi-ritz_eigval_restart
    # Extreme and closest to 0 eigenvalue solve
    ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_solve(xi_test,l,gMemSlfN,gMemSlfA,cellsA,
        chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
            tol_bicgstab)
    """
    Calculate the shift in the eigenvalues. This is calcualte to check
    if the smallest extremal eigenvalue is largest than the ritz_eigval_restart,
    which would indicate the is a unknown negative eigenvalue. 
    eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
    """
    return ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test
end 

function xi_update(xi,l,gMemSlfN,gMemSlfA,
    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
        tol_conv,tol_eigval,tol_bicgstab)
    # Test extreme and closest to 0 eigenvalue solve
    ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(xi,l,gMemSlfN,gMemSlfA,cellsA,
        chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
            tol_bicgstab)
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
                            ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(xi,l,gMemSlfN,gMemSlfA,cellsA,
                                chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                    tol_bicgstab)
                                
                            # Test to see what happens when we shift xi (and therefore 
                            # the eigenvalues by ritz_eigval_restart)
                            ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test =
                                eigval_test_calcs(xi,l,ritz_eigval_restart,gMemSlfN,
                                    gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                        tol_eigval,tol_bicgstab)
                        
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
                                print("old xi ", xi, "\n")
                                print("There was a negative eigenvalue we didn't know about \n")
                                # Calculate the unknown negative eigenvalue 
                                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                                print("unk_neg_eigval ", unk_neg_eigval, "\n")
                                # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                                xi += abs(unk_neg_eigval)
                                print("new xi ", xi, "\n")
                                # Restart the function with the new xi value 
                                xi_update(xi,l,gMemSlfN,gMemSlfA,
                                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                        tol_conv,tol_eigval,tol_bicgstab)
                            
                            else 
                            """
                            Essentially if 
                                abs(eigval_shift) < abs(harmonic_ritz_eigval_restart).
                            This means that there wasn't a negative eigval we didn't 
                            know about since we pretty much just calculated 
                            harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                            for more reasonning. 
                            """
                                print("Converged off eigshift 1 \n")
                                print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                                print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                                print("xi ", xi, "\n")
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
                            # Solve for extremal eigenvalue 
                            ritz_eigval_restart,harmonic_ritz_eigval_restart =
                                eigval_solve(xi,l,gMemSlfN,gMemSlfA,cellsA,
                                    chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                        tol_bicgstab)
                        end
                    end
                else 
                """
                Essentially if harmonic_ritz_eigval_restart > 0
                """
                    print("Double check that there isn't any negative eigenvalues we don't know about 2 \n")
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues by )
                    ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test =
                        eigval_test_calcs(xi,l,ritz_eigval_restart,gMemSlfN,
                            gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                tol_eigval,tol_bicgstab)
                    
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
                        print("old xi ", xi, "\n")
                        print("There was a negative eigenvalue we didn't know about \n")
                        # Calculate the unknown negative eigenvalue 
                        unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                        print("unk_neg_eigval ", unk_neg_eigval, "\n")
                        # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                        xi += abs(unk_neg_eigval)
                        print("new xi ", xi, "\n")
                        # Restart the function with the new xi value 
                        xi_update(xi,l,gMemSlfN,gMemSlfA,
                            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                tol_conv,tol_eigval,tol_bicgstab)
                    else 
                    """
                    Essentially if 
                        abs(eigval_shift) < abs(harmonic_ritz_eigval_restart).
                    This means that there wasn't a negative eigval we didn't 
                    know about since we pretty much just calculated 
                    harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                    for more reasonning. 
                    """
                        print("Converged off eigshift 2 \n")
                        print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                        print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                        print("xi ", xi, "\n")
                        return xi 
                    end 
                end
            else # Essentially if ritz_eigval_restart < 0
                for j = 1:1000
                    if ritz_eigval_restart > 0
                        # break
                        # Test to see what happens when we shift xi (and therefore 
                        # the eigenvalues by )
                        print("Double check that there isn't any negative eigenvalues we don't know about 3 \n")
                        ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test =
                            eigval_test_calcs(xi,l,ritz_eigval_restart,gMemSlfN,
                                gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                    tol_eigval,tol_bicgstab)
                    
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
                            print("old xi ", xi, "\n")
                            print("There was a negative eigenvalue we didn't know about \n")
                            # Calculate the unknown negative eigenvalue 
                            unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                            print("unk_neg_eigval ", unk_neg_eigval, "\n")
                            # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                            xi += abs(unk_neg_eigval)
                            print("new xi ", xi, "\n")
                            # Restart the function with the new xi value 
                            xi_update(xi,l,gMemSlfN,gMemSlfA,
                                cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                    tol_conv,tol_eigval,tol_bicgstab)                        
                        else 
                        """
                        Essentially if 
                            abs(eigval_shift) < abs(harmonic_ritz_eigval_restart).
                        This means that there wasn't a negative eigval we didn't 
                        know about since we pretty much just calculated 
                        harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                        for more reasonning. 
                        """
                            print("Converged off eigshift 3 \n")
                            print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                            print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                            print("xi ", xi, "\n")
                            return xi 
                        end
                    else # rEssentially when ritz_eigval_restart < 0
                        xi += abs(ritz_eigval_restart)
                        print("Increased xi 2 \n")

                        ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(xi,l,gMemSlfN,gMemSlfA,cellsA,
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
                    ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(xi,l,gMemSlfN,gMemSlfA,cellsA,
                        chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                            tol_bicgstab)
                        
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues by )
                    print("Double check that there isn't any negative eigenvalues we don't know about 4 \n")
                    ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test =
                        eigval_test_calcs(xi,l,ritz_eigval_restart,gMemSlfN,
                            gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                tol_eigval,tol_bicgstab)

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
                        print("There was a negative eigenvalue we didn't know about 4 \n")
                        # Calculate the unknown negative eigenvalue 
                        unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                        print("unk_neg_eigval ", unk_neg_eigval, "\n")
                        # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                        xi += abs(unk_neg_eigval)
                        # Restart the function with the new xi value 
                        xi_update(xi,l,gMemSlfN,gMemSlfA,
                            cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                tol_conv,tol_eigval,tol_bicgstab)
                        nb_neg_it += 1
                    else 
                    """
                    Essentially if 
                        abs(eigval_shift) < abs(harmonic_ritz_eigval_restart).
                    This means that there wasn't a negative eigval we didn't 
                    know about since we pretty much just calculated 
                    harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                    for more reasonning. 
                    """
                        print("Converged off eigshift 4 \n")
                        print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                        print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                        return xi 
                    end
                else # Essentially if ritz_eigval_restart < 0
                    for j = 1:1000
                        if ritz_eigval_restart > 0
                            # Test to see what happens when we shift xi (and therefore 
                            # the eigenvalues by ritz_eigval_restart)
                            print("Double check that there isn't any negative eigenvalues we don't know about 5 \n")
                            ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test =
                            eigval_test_calcs(xi,l,ritz_eigval_restart,gMemSlfN,
                                gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                                    tol_eigval,tol_bicgstab)
                
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
                                print("old xi ", xi, "\n")
                                print("There was a negative eigenvalue we didn't know about \n")
                                # Calculate the unknown negative eigenvalue 
                                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                                print("unk_neg_eigval ", unk_neg_eigval, "\n")
                                # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                                xi += abs(unk_neg_eigval)
                                print("new xi ", xi, "\n")
                                # Restart the function with the new xi value 
                                xi_update(xi,l,gMemSlfN,gMemSlfA,
                                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                                        tol_conv,tol_eigval,tol_bicgstab)                            
                            else 
                            """
                            Essentially if 
                                abs(eigval_shift) < abs(harmonic_ritz_eigval_restart).
                            This means that there wasn't a negative eigval we didn't 
                            know about since we pretty much just calculated 
                            harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
                            for more reasonning. 
                            """
                                print("Converged off eigshift 4 \n")
                                print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                                print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                                print("xi ", xi, "\n")
                                return xi 
                            end
                        else # Essentially when ritz_eigval_restart < 0
                            xi += abs(ritz_eigval_restart)
                            print("Increased xi 3 \n")

                            ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(xi,l,gMemSlfN,gMemSlfA,cellsA,
                                chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                    tol_bicgstab)
                        end 
                    end
                end
                print("Went through all third loop iterations \n")
                print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
            end
            print("Went through all fourth loop iterations \n")
        else 
        """
        Essentially if harmonic_ritz_eigval_restart > 0
        """
            
            # Test to see what happens when we shift xi (and therefore 
            # the eigenvalues by )
            print("Double check that there isn't any negative eigenvalues we don't know about 6 \n")
            ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test =
                eigval_test_calcs(xi,l,ritz_eigval_restart,gMemSlfN,
                    gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,tol_conv,
                        tol_eigval,tol_bicgstab)

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
                print("old xi ", xi, "\n")
                print("There was a negative eigenvalue we didn't know about 6 \n")
                # Calculate the unknown negative eigenvalue 
                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                print("unk_neg_eigval ", unk_neg_eigval, "\n")
                # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                xi += abs(unk_neg_eigval)
                print("new xi ", xi, "\n")
                # Restart the function with the new xi value 
                xi_update(xi,l,gMemSlfN,gMemSlfA,
                    cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
                        tol_conv,tol_eigval,tol_bicgstab)
            else 
            """
            Essentially if 
                abs(eigval_shift) < abs(harmonic_ritz_eigval_restart).
            This means that there wasn't a negative eigval we didn't 
            know about since we pretty much just calculated 
            harmonic_ritz_eigval_restart-ritz_eigval_restart. See above
            for more reasonning. 
            """
                print("Converged off eigshift 6 \n")
                print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
                print("xi ", xi, "\n")
                return xi 
            end
        end
    end
    print("Didn't converge. Went through all possible iterations. \n ")
    return xi 
end

end
