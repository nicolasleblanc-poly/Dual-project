module main_Pade_Ritz_HarmonicRitz_matrix
using LinearAlgebra,Random, Base.Threads,Plots,
    Restart_Ritz_jacobiDavidson,
        Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson,
            PadeForRitz_Code, Peaks

function D_matrix(sz)
    for it = 1:100000
        D = rand([0.0,0.5],sz,sz)
        # D[:,:] = (D .+ transpose(D)) ./ 2
        # # print("inv(D) 2 ", inv(D), "\n")
        P0 = adjoint(D)*D
        # print("P0 ", P0, "\n")
        if isposdef(P0) == false || cond(P0) > 1e6
            D = rand([0.0,0.5],sz,sz)
            D[:,:] = (D .+ transpose(D)) ./ 2
            P0 = adjoint(D)*D
            # print("P0 ", P0, "\n")
            print("isposdef(P0) ", isposdef(P0), "\n")
            print("cond(P0) ", cond(P0), "\n")
            print("cond(D) ", cond(D), "\n")
            print("det(D) ", det(D), "\n")
        elseif  isposdef(P0) == true  && det(D) > 0 # && 0 < cond(P0)< 1e6
            print("isposdef(P0) done ",isposdef(P0),"\n")
            return D
        end
    end
end

function xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
    tol_conv,tol_eigval,tol_bicgstab,xi,D,P1)
    
    """
    Solve for extremum and closest to 0 eigvals for initial xi value (often 0)
    """
    # Matrix 
    A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
    # Solve for extremal eigenvalue 
    (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(A,
        innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
    # Solve for eigenvalue closest to 0
    (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart) =
        jacDavRitzHarm_restart(A,innerLoopDim,restartDim,tol_MGS,
            tol_conv,tol_eigval,tol_bicgstab)

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
                            # Matrix 
                            A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # Solve for extremal eigenvalue 
                            (ritz_eigval_restart, ritz_eigvec_restart) = 
                                jacDavRitz_restart(A,
                                    innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                        tol_bicgstab)
                            # Solve for eigenvalue closest to 0
                            (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart)= 
                                jacDavRitzHarm_restart(A,innerLoopDim,restartDim,
                                    tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                                 
                            # Test to see what happens when we shift xi (and therefore 
                            # the eigenvalues by )
                            xi_test = xi-ritz_eigval_restart
                            # Matrix
                            A_test = (alpha_1+xi_test)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # Solve for extremal eigenvalue 
                            (ritz_eigval_restart_test, ritz_eigvec_restart_test) = jacDavRitz_restart(A_test,
                                innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                            # Solve for eigenvalue closest to 0
                            (harmonic_ritz_eigval_restart_test, harmonic_ritz_eigvec_restart_test) =
                                jacDavRitzHarm_restart(A_test,innerLoopDim,restartDim,tol_MGS,
                                    tol_conv,tol_eigval,tol_bicgstab)
                            # Calculate the shift in the eigenvalues. This is calcualte to check
                            # if the smallest extremal eigenvalue is largest than the ritz_eigval_restart,
                            # which would indicate the is a unknown negative eigenvalue. 
                            eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
                
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
                                xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                                    tol_conv,tol_eigval,tol_bicgstab,xi,D,P1)                            
                            
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
                            # Matrix 
                            A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # Solve for extremal eigenvalue 
                            (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(A,
                                innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                            # Solve for eigenvalue closest to 0
                            (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart) =
                                jacDavRitzHarm_restart(A,innerLoopDim,restartDim,tol_MGS,
                                    tol_conv,tol_eigval,tol_bicgstab)                
                        end
                    end
                else 
                """
                Essentially if harmonic_ritz_eigval_restart > 0
                """
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues by )
                    xi_test = xi-ritz_eigval_restart
                    # Matrix
                    A_test = (alpha_1+xi_test)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                    # Solve for extremal eigenvalue 
                    (ritz_eigval_restart_test, ritz_eigvec_restart_test) = jacDavRitz_restart(A_test,
                        innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                    # Solve for eigenvalue closest to 0
                    (harmonic_ritz_eigval_restart_test, harmonic_ritz_eigvec_restart_test) =
                        jacDavRitzHarm_restart(A_test,innerLoopDim,restartDim,tol_MGS,
                            tol_conv,tol_eigval,tol_bicgstab)
                    # Calculate the shift in the eigenvalues. This is calcualte to check
                    # if the smallest extremal eigenvalue is largest than the ritz_eigval_restart,
                    # which would indicate the is a unknown negative eigenvalue. 
                    eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
            
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
                        xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                            tol_conv,tol_eigval,tol_bicgstab,xi,D,P1)  
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
                for j = 1:100000
                    if ritz_eigval_restart > 0
                        break
                    else # ritz_eigval_restart < 0
                        # Increase the xi value by 10 (arbitrary value)
                        xi += 10
                        # Matrix 
                        A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                        # Solve for extremal eigenvalue 
                        (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(A,
                            innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                        # Solve for eigenvalue closest to 0
                        (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart) =
                            jacDavRitzHarm_restart(A,innerLoopDim,restartDim,tol_MGS,
                                tol_conv,tol_eigval,tol_bicgstab)   
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
                    # Matrix 
                    A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                    # Solve for extremal eigenvalue 
                    (ritz_eigval_restart, ritz_eigvec_restart) = 
                        jacDavRitz_restart(A,
                            innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
                                tol_bicgstab)
                    # Solve for eigenvalue closest to 0
                    (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart)= 
                        jacDavRitzHarm_restart(A,innerLoopDim,restartDim,
                            tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                         
                    # Test to see what happens when we shift xi (and therefore 
                    # the eigenvalues by )
                    xi_test = xi-ritz_eigval_restart
                    # Matrix
                    A_test = (alpha_1+xi_test)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                    # Solve for extremal eigenvalue 
                    (ritz_eigval_restart_test, ritz_eigvec_restart_test) = jacDavRitz_restart(A_test,
                        innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                    # Solve for eigenvalue closest to 0
                    (harmonic_ritz_eigval_restart_test, harmonic_ritz_eigvec_restart_test) =
                        jacDavRitzHarm_restart(A_test,innerLoopDim,restartDim,tol_MGS,
                            tol_conv,tol_eigval,tol_bicgstab)
                    # Calculate the shift in the eigenvalues. This is calcualte to check
                    # if the smallest extremal eigenvalue is largest than the ritz_eigval_restart,
                    # which would indicate the is a unknown negative eigenvalue. 
                    eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
        
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
                        xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                            tol_conv,tol_eigval,tol_bicgstab,xi,D,P1)  
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
                            # Matrix 
                            A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
                            # Solve for extremal eigenvalue 
                            (ritz_eigval_restart, ritz_eigvec_restart) = jacDavRitz_restart(A,
                                innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                            # Solve for eigenvalue closest to 0
                            (harmonic_ritz_eigval_restart, harmonic_ritz_eigvec_restart) =
                                jacDavRitzHarm_restart(A,innerLoopDim,restartDim,tol_MGS,
                                    tol_conv,tol_eigval,tol_bicgstab) 
                        end 
                    end
                end
            end
        else 
        """
        Essentially if harmonic_ritz_eigval_restart > 0
        """
            # Test to see what happens when we shift xi (and therefore 
            # the eigenvalues by )
            xi_test = xi-ritz_eigval_restart
            # Matrix
            A_test = (alpha_1+xi_test)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
            # Solve for extremal eigenvalue 
            (ritz_eigval_restart_test, ritz_eigvec_restart_test) = jacDavRitz_restart(A_test,
                innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
            # Solve for eigenvalue closest to 0
            (harmonic_ritz_eigval_restart_test, harmonic_ritz_eigvec_restart_test) =
                jacDavRitzHarm_restart(A_test,innerLoopDim,restartDim,tol_MGS,
                    tol_conv,tol_eigval,tol_bicgstab)
            # Calculate the shift in the eigenvalues. This is calcualte to check
            # if the smallest extremal eigenvalue is largest than the ritz_eigval_restart,
            # which would indicate the is a unknown negative eigenvalue. 
            eigval_shift = harmonic_ritz_eigval_restart_test-ritz_eigval_restart_test
    
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
                xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
                    tol_conv,tol_eigval,tol_bicgstab,xi,D,P1)  
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

""""
Initial parameter values
"""
xi = 0.1 # Temporary value
alpha_0 = 1
alpha_1 = 1
sz = 256
innerLoopDim = 25
restartDim = 5
tol_MGS = 1.0e-12
tol_conv = 1.0e-12
tol_eigval = 1.0e-9
tol_bicgstab = 1e-6
tol_bissection = 1e-4

D = D_matrix(sz)+2*I
# P1 RND semidefinite matrix
P1 = rand([-2.0,2.0],sz,sz)+4*I
P1[:,:] = (P1 .+ adjoint(P1)) ./ 2

# Check that P0 is positive definite and that P1 is semidefinite
trueEig_P1 = eigen(P1) 
print("isposdef(P1) ", isposdef(P1), "\n")

# RND vector
s = Vector{ComplexF64}(undef,sz)
rand!(s)

xi = xi_update(alpha_0,alpha_1,innerLoopDim,restartDim,tol_MGS,
    tol_conv,tol_eigval,tol_bicgstab,xi,D,P1)
print("xi such that smallest eigval of A is positive ", xi, "\n")
surr_func_xi = surrogate_function(alpha_0,alpha_1,D,P1,xi,s)
print("surr_func_xi_first_update: ", surr_func_xi, "\n")
 
print("s ", s, "\n")

plot_surrogate_func(alpha_0,alpha_1,D,P1,50,s,1)

ans,surrogate_value = root_solve(alpha_0,alpha_1,D,P1,xi,s,tol_bissection)
print("The guessed root is: ", ans)
plot_surrogate_func(alpha_0,alpha_1,D,P1,Int(round(ans))+50,s,2)

print("surrogate_function(alpha_0,alpha_1,D,P1,ans) ", surrogate_function(alpha_0,alpha_1,D,P1,ans,s), "\n")


A = (alpha_1+ans)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
# print("A ", A, "\n")
trueEig_A = eigen(A)
A_smallest_eigval = trueEig_A.values[1]
print("A_smallest_eigval ", A_smallest_eigval, "\n")
print("All of A eigvals ", trueEig_A.values, "\n")

end
