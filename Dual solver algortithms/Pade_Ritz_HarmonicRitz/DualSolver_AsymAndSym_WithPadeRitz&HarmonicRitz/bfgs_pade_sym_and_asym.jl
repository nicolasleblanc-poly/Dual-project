"""
This module is a bfgs algorithm that is used to update the Lagrange
multipliers. It uses the root solve algorihm composed of the Padé, Ritz and 
harmonic Ritz algorithms.

Author: Nicolas Leblanc
"""

module bfgs_pade_sym_and_asym
export validityfunc, BFGS_fakeS_with_restart_pade
using LinearAlgebra, product_sym_and_asym, 
    approx_xi_solve_Ritz_HarmonicRitz_GreenOperator_sym_and_asym,PadeForRitz_GreenOperator_Code_sym_and_asym,
        Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson_operator_sym_and_asym,
            Restart_Ritz_jacobiDavidson_operator_sym_and_asym
# BFGS with restart code
# This file also contains code for the minimum eigenvalue
# calculation and a function that verifies if we are
# still in the domain of duality

function BFGS_fakeS_with_restart_pade(gMemSlfN, gMemSlfA,xi,l,innerLoopDim,
    restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,tol_bissection,dgfunc,P,
        chi_inv_coeff,ei,b,cellsA,root_solve,min_eig_func, gradConverge=false, 
            opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, 
                min_iter=6)
    
    # print("Entered BFGS")
    dofnum = length(l)
    dualval = 0.0
    grad = zeros(Float64, dofnum, 1)
    tmp_grad = zeros(ComplexF64, dofnum, 1)
    Hinv = I #approximate inverse Hessian
    prev_dualval = Inf 
    dof = l
    objval=0.0
    function justfunc(xi,d,fSlist)
        return dgfunc(xi,d,Array([]),P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,fSlist,false)
    end
    # justfunc = lambda d: dgfunc(d, np.array([]), fSlist, get_grad=False)
    olddualval = Inf #+0.0im
    reductCount = 0
    while true #gradual reduction of modified source amplitude, outer loop
        fSlist = [] #remove old fake sources and restart with each outer iteration
        alpha_last = 1.0 #reset step size
        Hinv = I #reset Hinv
        val = dgfunc(xi,dof,grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,fSlist,true)  
        dualval = val[1]
        grad = val[2]
        obj = val[3]
        # T = val[4]
        # A = val[5]
        # b = val[6]

        # This gets the value of the dual function and the dual's gradient
        print('\n', "Outer iteration # ", reductCount, " the starting dual value is ", dualval, " fakeSratio ", fakeSratio, "\n")
        iternum = 0
        while true
            print("In 1: \n")
            iternum += 1
            print("Outer iteration # ", reductCount, " Inner iteration # ", iternum, "\n")
            Ndir = - Hinv * grad
            pdir = Ndir / norm(Ndir)
            # Backtracking line search, impose feasibility and Armijo condition
            p_dot_grad = dot(pdir, grad)
            print("pdir dot grad is: ", p_dot_grad, "\n")
            c_reduct = 0.7
            c_A = 1e-4
            c_W = 0.9
            alpha = alpha_start = alpha_last
            print("starting alpha ", alpha_start, "\n") 

            # New domain of duality checker 
            # This is the root_solve function 
            xi,surr_func_xi_zeros = root_solve(xi,l,P,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,ei,b,
                innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,
                    tol_bissection)

            alpha_feas = alpha
            print("alpha before backtracking is ", alpha_feas, "\n")
            alphaopt = alpha
            Dopt = Inf #+0.0im
            while true
                # tmp_dual 1
                tmp_dual = justfunc(xi,dof.+alpha*pdir, fSlist)[1] # , Array([]), fSlist, tsolver, false
                print("tmp_dual",tmp_dual,"\n")
                if tmp_dual < Dopt #the dual is still decreasing as we backtrack, continue
                    Dopt = tmp_dual
                    alphaopt=alpha
                else
                    alphaopt=alpha ###ISSUE!!!!
                    break
                end
                if tmp_dual <= dualval + c_A*alpha*p_dot_grad #Armijo backtracking condition
                    alphaopt = alpha
                    break
                end                      
                alpha *= c_reduct
            end
            added_fakeS = false

            print("alpha_feas ", alpha_feas, "\n")
            print("alphaopt ", alphaopt, "\n")
            print("alpha_start ", alpha_start, "\n")
            print("alphaopt/alpha_start: ", alphaopt/alpha_start, "\n")
            print("alpha_feas/alpha_start ", alpha_feas/alpha_start, "\n")
            print("alphaopt/alpha_feas ", alphaopt/alpha_feas, "\n")
            print("(c_reduct+1)/2: ", (c_reduct+1)/2, "\n")
            if alphaopt/alpha_start>(c_reduct+1)/2 #in this case can start with bigger step
                alpha_last = alphaopt*2
                print("In 2: \n")
            else
                alpha_last = alphaopt

                if alpha_feas/alpha_start < (c_reduct+1)/2 && alphaopt/alpha_feas > (c_reduct+1)/2 #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                    print("In 3: \n")
                    added_fakeS = true
                    print("encountered feasibility wall, adding a fake source term \n")
                    singular_dof = dof .+ alpha_feas*pdir #dof that is roughly on duality boundary
                    print("singular_dof", singular_dof, "\n")
                    # for your info: min_eig_func = jacDavRitzHarm_restart here
                    eig_fct_eval = min_eig_func(xi,singular_dof,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
                        P,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
                    
                    # Find the subspace of ZTT closest to being singular to target with fake source
                    mineigv = eig_fct_eval[1] # eigenvector 
                    mineigw = eig_fct_eval[2] # eigenvalue
                    fakeS_eval = dgfunc(xi,dof,Array([]),P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,[mineigv],false)
                    
                    fakeSval = fakeS_eval[1]
                    # This only gets us the value of the dual function
                    epsS = sqrt(fakeSratio*abs(dualval/fakeSval))
                    print("Appending to fSlist \n")
                    append!(fSlist,epsS*[mineigv]) #add new fakeS to fSlist                                  
                    print("length of fSlist ", length(fSlist), "\n")
                end
            end
            print("Done \n")
            print("stepsize alphaopt is ", alphaopt, '\n')
            delta = alphaopt * pdir
            print("delta ", delta, "\n")
            ######### Decide how to update Hinv ############
            
            tmp_val = dgfunc(xi,dof+delta,tmp_grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,fSlist, true)
            tmp_dual = tmp_val[1] #tmp_dual 2
            tmp_grad = tmp_val[2]
            # T = tmp_val[4]
            # A = tmp_val[5]
            # b = tmp_val[6]
            p_dot_tmp_grad = dot(pdir, tmp_grad)
            if added_fakeS == true
                Hinv = I # The objective has been modified; restart Hinv from identity
            elseif p_dot_tmp_grad > c_W*p_dot_grad #satisfy Wolfe condition, update Hinv
                print("updating Hinv \n")
                gamma = tmp_grad - grad
                gamma_dot_delta = alpha * (p_dot_tmp_grad-p_dot_grad)
                Hinv_dot_gamma = Hinv * tmp_grad + Ndir
                Hinv -= ((Hinv_dot_gamma.*delta') + (delta.*Hinv_dot_gamma') - (1+dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*(delta.*delta') ) / gamma_dot_delta
                print("Hinv2: ", Hinv, "\n")
            end
            dualval = tmp_dual
            grad[1:end] = tmp_grad[1:end]
            dof += delta 
            print("delta", delta, "\n")
            print("dualval ", dualval, "\n")
            objval = dualval - dot(dof,grad)
            print("objval ", objval, "\n")
            eqcstval = dot(abs.(dof),abs.(grad))
            print("eqcstval ", eqcstval, "\n")
            print("at iteration # ", iternum, " the dual, objective, eqconstraint value is ", dualval,"  " ,objval,"  " , eqcstval, "\n")
            print("normgrad is ", norm(grad), "\n")
            if gradConverge==false && iternum>min_iter && abs(abs(dualval)-abs(objval))<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) && norm(grad)<opttol * abs(dualval) #objective and gradient norm convergence termination
                break
                print("First condition is attained \n")
            end
            if gradConverge==true && iternum>min_iter && abs(dualval-objval)<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
                print("Second condition is attained \n")
            end
            if mod(iternum,iter_period)==0
                print("prev_dualval is ", prev_dualval, "\n")
                if abs(prev_dualval.-dualval)<abs(dualval)*opttol #dual convergence / stuck optimization termination
                    break
                end
                prev_dualval = dualval
            end
        end
        if abs(olddualval-dualval)<opttol*abs(dualval)
            break
        end
        """
        #if len(fSlist)<=1 and np.abs(olddualval-dualval)<opttol*np.abs(dualval): #converged w.r.t. reducing fake sources
            #break
        # if len(fSlist)==0 #converged without needing fake sources, suggests strong duality
        #     break
        """
        olddualval = dualval
        reductCount += 1
        fakeSratio *= reductFactor #reduce the magnitude of the fake sources for the next iteration
    end
    return dof, grad, dualval, objval
end
end