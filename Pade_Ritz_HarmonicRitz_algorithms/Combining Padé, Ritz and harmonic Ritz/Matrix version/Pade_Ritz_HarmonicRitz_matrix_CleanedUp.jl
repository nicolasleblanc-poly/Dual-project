"""
This module is the main file that is run to test the Padé, 
Ritz and harmonic Ritz duality checker algorithm.

Author: Nicolas Leblanc
"""

module Pade_Ritz_HarmonicRitz_matrix
using LinearAlgebra,Random, Base.Threads,Plots,
    .Restart_Ritz_jacobiDavidson,
        Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson,
            PadeForRitz_Code, Peaks

function D_matrix(sz)
    """
    We want to generate a matrix D such that the product of P0 = adjoint(D)*D 
        gives a positive definite matrix. The other important matrix, P1, is a 
        semidefinite matrix.
    One important thing to note is that I used a for loop instead of a while 
        loop because while loops suck in julia since you need to mess with global 
        variables.
    I also have a check that the determinant of D is positive to avoid 
        singularities and that the condition number of P0 isn't too big for
        a similar reason.
    """
    for it = 1:100000 # An arbitrary number of iterations to generate good matrices
        D = rand([0.0,0.5],sz,sz)
        P0 = adjoint(D)*D
        if isposdef(P0) == false || cond(P0) > 1e6
            D = rand([0.0,0.5],sz,sz)
            D[:,:] = (D .+ transpose(D)) ./ 2
            P0 = adjoint(D)*D
        elseif  isposdef(P0) == true && det(D) > 0 
            return D
        end
    end
end

function eigval_solve(alpha_0,alpha_1,xi,D,P1,innerLoopDim,restartDim,tol_MGS,
    tol_conv,tol_eigval,tol_bicgstab, max_nb_it, basis_solver)
    # Matrix 
    A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
    trueEig_A = eigen(A)
    A_smallest_eigval = trueEig_A.values[1]
    print("A_smallest_eigval ", A_smallest_eigval, "\n")
    print("All of A eigvals ", trueEig_A.values, "\n")

    # Solve for extremal eigenvalue 
    # The ritz_nb_it_restart and ritz_nb_it_total_bicgstab_solve variables are not used here.
    # They are used for the Ritz convergence tests.
    (ritz_eigval_restart,ritz_nb_it_restart,ritz_nb_it_total_bicgstab_solve) = jacDavRitz_restart(A,
        innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)
    print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
            
    # Solve for eigenvalue closest to 0
    # The harmonic_ritz_nb_it_restart and harmonic_ritz_nb_it_total_bicgstab_solve variables are not used here.
    # They are used for the harmonic Ritz convergence tests.
    (harmonic_ritz_eigval_restart,harmonic_ritz_nb_it_restart,harmonic_ritz_nb_it_total_bicgstab_solve) =
        jacDavRitzHarm_restart(A,innerLoopDim,restartDim,tol_MGS,
            tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)
    print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")

    return ritz_eigval_restart,harmonic_ritz_eigval_restart
end 

function eigval_test_calcs(alpha_0,alpha_1,xi,ritz_eigval_restart,D,P1,
    innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)
    # Test to see what happens when we shift xi (and therefore 
    # the eigenvalues) by ritz_eigval_restart
    xi_test = xi-ritz_eigval_restart
    # Extreme and closest to 0 eigenvalue solve
    (ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test)=eigval_solve(alpha_0,alpha_1,xi_test,D,P1,innerLoopDim,
            restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)
    """
    Calculate the shift in the eigenvalues. This is calcualte to check
        if the smallest extremal eigenvalue is largest than the ritz_eigval_restart,
        which would indicate the is a unknown negative eigenvalue. 
        eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
    """
    return ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test
end 

function xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
    tol_conv,tol_eigval,tol_bicgstab,xi,D,P1,max_nb_it,basis_solver)
    
    """
    Solve for extremum and closest to 0 eigvals for initial xi value (often 0)
    """
    ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,D,P1,innerLoopDim,restartDim,tol_MGS,
        tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)

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
                    for it = 1:max_nb_it
                        if harmonic_ritz_eigval_restart > 0 
                            """
                            Double check that harmonic_ritz_eigval_restart > 0 before breaking 
                            """
                            ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,D,P1,innerLoopDim,restartDim,tol_MGS,
                                tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)

                            # Test to see what happens when we shift xi (and therefore 
                            # the eigenvalues) by ritz_eigval_restart            
                            ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_test_calcs(alpha_0,alpha_1,xi,ritz_eigval_restart,D,P1,
                                innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)

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
                                A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                                trueEig_A = eigen(A)
                                A_smallest_eigval = trueEig_A.values[1]
                                print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                                print("All of A eigvals with old xi ", trueEig_A.values, "\n")
                                print("old xi ", xi, "\n")
                                print("There was a negative eigenvalue we didn't know about 1 \n")
                                # Calculate the unknown negative eigenvalue 
                                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                                print("unk_neg_eigval ", unk_neg_eigval, "\n")
                                # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                                xi += abs(unk_neg_eigval)
                                print("new xi ", xi, "\n")
                                # Matrix
                                A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                                trueEig_A = eigen(A)
                                A_smallest_eigval = trueEig_A.values[1]
                                print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                                print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                                # Restart the function with the new xi value 
                                # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                                #     tol_conv,tol_eigval,tol_bicgstab,xi,D,P1) 
                                xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                                  tol_conv,tol_eigval,tol_bicgstab,xi,D,P1,max_nb_it,basis_solver)                              
                            
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
                                A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                                trueEig_A = eigen(A)
                                A_smallest_eigval = trueEig_A.values[1]
                                print("A_smallest_eigval ", A_smallest_eigval, "\n")
                                return xi 
                            end
                        else 
                        """
                        Essentially if harmonic_ritz_eigval_restart < 0.
                        Increase xi until harmonic_ritz_eigval_restart > 0.
                        """
                            # Increase the xi value by harmonic_ritz_eigval_restart
                            # so that all the eigenvalues should be positive 
                            # seeing that ritz_eigval_restart > 0
                            xi += abs(harmonic_ritz_eigval_restart)

                            print("Increased xi 1 \n")

                            ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,D,P1,innerLoopDim,restartDim,tol_MGS,
                                tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)            
                        end
                    end
                    print("Went through all first loop iterations \n")
                else 
                """
                Essentially if harmonic_ritz_eigval_restart > 0
                """
                    # Matrix
                    A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                    trueEig_A = eigen(A)
                    A_smallest_eigval = trueEig_A.values[1]
                    print("A_smallest_eigval ", A_smallest_eigval, "\n")
                    print("All of A eigvals ", trueEig_A.values, "\n")
                
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues) by ritz_eigval_restart
                    ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_test_calcs(alpha_0,alpha_1,xi,ritz_eigval_restart,D,P1,
                                innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)

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
                        A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        trueEig_A = eigen(A)
                        A_smallest_eigval = trueEig_A.values[1]
                        print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                        print("All of A eigvals with old xi ", trueEig_A.values, "\n")
                        print("old xi ", xi, "\n")
                        print("There was a negative eigenvalue we didn't know about 2 \n")
                        # Calculate the unknown negative eigenvalue 
                        unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                        print("unk_neg_eigval ", unk_neg_eigval, "\n")
                        # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                        xi += abs(unk_neg_eigval)
                        print("new xi ", xi, "\n")
                        # Matrix
                        A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        trueEig_A = eigen(A)
                        A_smallest_eigval = trueEig_A.values[1]
                        print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                        print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                        # Restart the function with the new xi value 
                        # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                        #     tol_conv,tol_eigval,tol_bicgstab,xi,D,P1) 
                        xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                            tol_conv,tol_eigval,tol_bicgstab,xi,D,P1,max_nb_it,basis_solver)  
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
                        A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        trueEig_A = eigen(A)
                        A_smallest_eigval = trueEig_A.values[1]
                        print("A_smallest_eigval ", A_smallest_eigval, "\n")
                        print("All of A eigvals ", trueEig_A.values, "\n")
                        return xi 
                    end 
                end
            else # Essentially if ritz_eigval_restart < 0
                for j = 1:max_nb_it
                    if ritz_eigval_restart > 0
                        # Test to see what happens when we shift xi (and therefore 
                        # the eigenvalues) by ritz_eigval_restart
                        ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_test_calcs(alpha_0,alpha_1,xi,ritz_eigval_restart,D,P1,
                            innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)

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
                            A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            trueEig_A = eigen(A)
                            A_smallest_eigval = trueEig_A.values[1]
                            print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                            print("All of A eigvals with old xi ", trueEig_A.values, "\n")
                            print("old xi ", xi, "\n")
                            print("There was a negative eigenvalue we didn't know about 3 \n")
                            # Calculate the unknown negative eigenvalue 
                            unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                            print("unk_neg_eigval ", unk_neg_eigval, "\n")
                            # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                            xi += abs(unk_neg_eigval)
                            print("new xi ", xi, "\n")
                            # Matrix
                            A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            trueEig_A = eigen(A)
                            A_smallest_eigval = trueEig_A.values[1]
                            print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                            print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                            # Restart the function with the new xi value 
                            # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                            #     tol_conv,tol_eigval,tol_bicgstab,xi,D,P1) 
                            xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                                tol_conv,tol_eigval,tol_bicgstab,xi,D,P1,max_nb_it,basis_solver)  
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
                            print("xi ", xi, "\n")
                            # Matrix
                            A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            trueEig_A = eigen(A)
                            A_smallest_eigval = trueEig_A.values[1]
                            print("A_smallest_eigval ", A_smallest_eigval, "\n")
                            return xi 
                        end
                    else # ritz_eigval_restart < 0
                        # Increase the xi value by ritz_eigval_restart
                        xi += abs(ritz_eigval_restart)
                        print("Increased xi 2 \n")

                        ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,D,P1,innerLoopDim,restartDim,tol_MGS,
                            tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)
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
            for it = 1:max_nb_it
                if harmonic_ritz_eigval_restart > 0
                """
                Double check that harmonic_ritz_eigval_restart > 0 before breaking 
                """
                    ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,D,P1,innerLoopDim,restartDim,tol_MGS,
                        tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)
                        
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues) by ritz_eigval_restart
                    ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_test_calcs(alpha_0,alpha_1,xi,ritz_eigval_restart,D,P1,
                        innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)

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
                        A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        trueEig_A = eigen(A)
                        A_smallest_eigval = trueEig_A.values[1]
                        print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                        print("All of A eigvals with old xi ", trueEig_A.values, "\n")

                        print("There was a negative eigenvalue we didn't know about 4 \n")
                        # Calculate the unknown negative eigenvalue 
                        unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                        print("unk_neg_eigval ", unk_neg_eigval, "\n")
                        # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                        xi += abs(unk_neg_eigval)
                        # Matrix
                        A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        trueEig_A = eigen(A)
                        A_smallest_eigval = trueEig_A.values[1]
                        print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                        print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                        # Restart the function with the new xi value 
                        # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                        #     tol_conv,tol_eigval,tol_bicgstab,xi,D,P1) 
                        xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                            tol_conv,tol_eigval,tol_bicgstab,xi,D,P1,max_nb_it,basis_solver)  
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
                        A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        trueEig_A = eigen(A)
                        A_smallest_eigval = trueEig_A.values[1]
                        print("A_smallest_eigval ", A_smallest_eigval, "\n")
                        return xi 
                    end
                # New 
                else 
                    """
                    Essentially if harmonic_ritz_eigval_restart < 0.
                    Increase xi until harmonic_ritz_eigval_restart > 0.
                    """
                    # Increase the xi value by harmonic_ritz_eigval_restart
                    # so that all the eigenvalues should be positive 
                    # seeing that ritz_eigval_restart > 0
                    xi += abs(harmonic_ritz_eigval_restart)

                    print("Increased xi 3 \n")

                    ritz_eigval_restart,harmonic_ritz_eigval_restart = eigval_solve(alpha_0,alpha_1,xi,D,P1,innerLoopDim,restartDim,tol_MGS,
                        tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)            
                end
                
                print("Went through all third loop iterations \n")
                # Matrix
                A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                trueEig_A = eigen(A)
                A_smallest_eigval = trueEig_A.values[1]
                print("A_smallest_eigval end of third loop ", A_smallest_eigval, "\n")
                print("All of A eigvals end of third loop ", trueEig_A.values, "\n")
                print("ritz_eigval_restart ", ritz_eigval_restart,"\n")
                print("harmonic_ritz_eigval_restart ", harmonic_ritz_eigval_restart,"\n")
            end
            print("Went through all fourth loop iterations \n")
        else 
        """
        Essentially if harmonic_ritz_eigval_restart > 0
        """
            # Matrix
            A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
            trueEig_A = eigen(A)
            A_smallest_eigval = trueEig_A.values[1]
            print("A_smallest_eigval ", A_smallest_eigval, "\n")
            print("All of A eigvals ", trueEig_A.values, "\n")

            # Test to see what happens when we shift xi (and therefore 
            # the eigenvalues by )
            ritz_eigval_restart_test,harmonic_ritz_eigval_restart_test = eigval_test_calcs(alpha_0,alpha_1,xi,ritz_eigval_restart,D,P1,
                innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,max_nb_it,basis_solver)

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
                A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                trueEig_A = eigen(A)
                A_smallest_eigval = trueEig_A.values[1]
                print("A_smallest_eigval with old xi ", A_smallest_eigval, "\n")
                print("All of A eigvals with old xi ", trueEig_A.values, "\n")
                print("old xi ", xi, "\n")
                print("There was a negative eigenvalue we didn't know about 6 \n")
                # Calculate the unknown negative eigenvalue 
                unk_neg_eigval = ritz_eigval_restart_test+ritz_eigval_restart
                print("unk_neg_eigval ", unk_neg_eigval, "\n")
                # Shift up xi and the eigenvalues by the unknown negative eigenvalue
                xi += abs(unk_neg_eigval)
                print("new xi ", xi, "\n")
                # Matrix
                A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                trueEig_A = eigen(A)
                A_smallest_eigval = trueEig_A.values[1]
                print("A_smallest_eigval with new xi ", A_smallest_eigval, "\n")
                print("All of A eigvals with new xi ", trueEig_A.values, "\n")
                # Restart the function with the new xi value 
                # xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                #     tol_conv,tol_eigval,tol_bicgstab,xi,D,P1)  
                xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                    tol_conv,tol_eigval,tol_bicgstab,xi,D,P1,max_nb_it,basis_solver) 
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
                print("xi ", xi, "\n")
                # Matrix
                A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                trueEig_A = eigen(A)
                A_smallest_eigval = trueEig_A.values[1]
                print("A_smallest_eigval ", A_smallest_eigval, "\n")
                return xi 
            end
        end
    end
end

""""
Initial parameter values
"""
xi = 0.1 # Temporary value
alpha_0 = 1
alpha_1 = 1
sz = 15
innerLoopDim = 5
restartDim = 2
tol_MGS = 1.0e-12
tol_conv = 1.0e-12
tol_eigval = 1.0e-9
tol_bicgstab = 1e-6
tol_bissection = 1e-4
max_nb_it = 1000
basis_solver = "bicgstab"

"""
Generate random matrices and vectors
"""
D = D_matrix(sz) # +2*I
# P1 RND semidefinite matrix
P1 = rand([-2.0,2.0],sz,sz) # +4*I
P1[:,:] = (P1 .+ adjoint(P1)) ./ 2

# Check that P1 is semidefinite
trueEig_P1 = eigen(P1) 
print("isposdef(P1) ", isposdef(P1), "\n")

# RND vector
s = Vector{ComplexF64}(undef,sz)
rand!(s)
print("s ", s, "\n")

"""
Calculation of an apprimate xi value such that all of the eigenvalues of the
    initial matrix A are positive
"""
xi = xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
    tol_conv,tol_eigval,tol_bicgstab,xi,D,P1,max_nb_it,basis_solver)
print("xi such that smallest eigval of A is positive ", xi, "\n")
surr_func_xi = surrogate_function(alpha_0,alpha_1,D,P1,xi,s)
print("surr_func_xi_first_update: ", surr_func_xi, "\n")

"""
Check if all of the eigenvalues are positive after the first xi value 
    approximate calculation
"""
A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
trueEig_A = eigen(A)
A_smallest_eigval = trueEig_A.values[1]
print("A_smallest_eigval ", A_smallest_eigval, "\n")
print("All of A eigvals ", trueEig_A.values, "\n")

plot_surrogate_func(alpha_0,alpha_1,D,P1,Int(round(xi)),s,1)
# Calcule the last zero crossing and it's associated surrogate value
ans,surrogate_value = root_solve(alpha_0,alpha_1,D,P1,xi,s,tol_bissection)
print("The guessed root is: ", ans)
# Plot the surrogate function a second time to have a better look around the 
# found root 
plot_surrogate_func(alpha_0,alpha_1,D,P1,Int(round(ans))+50,s,2)
# 
print("surrogate_function(alpha_0,alpha_1,D,P1,ans) ", surrogate_function(alpha_0,alpha_1,D,P1,ans,s), "\n")

# Check if all of the eigenvalues are actually positive like they should be
A = (alpha_1+ans)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
trueEig_A = eigen(A)
A_smallest_eigval = trueEig_A.values[1]
print("A_smallest_eigval ", A_smallest_eigval, "\n")
print("All of A eigvals ", trueEig_A.values, "\n")

end
