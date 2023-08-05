"""
This module is for the BFGS with restart code. It also contains code for the minimum eigenvalue
calculation and a function that verifies if we are
still in the domain of duality by seeing if the minimum 
eigenvalue is positive or negative.

Author: Nicolas Leblanc
"""

module bfgs_power_iteration_asym_only
export mineigfunc, validityfunc, BFGS_fakeS_with_restart_pi, power_iteration_first_evaluation, power_iteration_second_evaluation
using LinearAlgebra, product

# BFGS with restart code
# This file 
function power_iteration_first_evaluation(l,l2,cellsA,gMemSlfN, gMemSlfA, chi_inv_coeff, P)
    # Ideally choose a random vector to decrease the chance that our vector
    # is orthogonal to the eigenvector
    b_k = rand(ComplexF64, 3*cellsA[1]*cellsA[2]*cellsA[3], 1)
    # let's implement an error check using the ratios of different iterations 
    good = false 
    while good == false
        # Calculate the n+1 term
        # calculate the operator-vector product (A linear operator and b_k vector)
        # G|v> type calculation
        A_bk = sym_and_asym_sum(l,l2,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, b_k) 
        
        # Calculate the norm
        A_bk_norm = norm(A_bk)
        # Renormalize the vector
        b_k1 = A_bk / A_bk_norm

        # Calculate the n+2 term
        # G|v> type calculation
        A_bk1 = sym_and_asym_sum(l,l2,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, b_k1) 
    
        # Calculate the norm
        A_bk1_norm = norm(A_bk1)
        # Renormalize the vector
        b_k = A_bk1 / A_bk1_norm # technically b_k2 but it's called b_k since it will be the
        # b_k for the next iteration of the loop

        # Calculate the A*b_k product with the new b_k (the n+2 one)
        # G|v> type calculation
        A_bk2 = sym_and_asym_sum(l,l2,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, b_k)        

        # Calculate the ratios 
        # norm(A^{n+1}*x)/norm(A^{n}*x)
        ratio_n1_n = norm(A_bk1)/norm(A_bk)
        # norm(A^{n+2}*x)/norm(A^{n+1}*x)
        ratio_n2_n1 = norm(A_bk2)/norm(A_bk1)
        # do this for 3 times in a row

        if abs(ratio_n2_n1-ratio_n1_n)/ratio_n1_n < 0.01
            good = true
        end
        global eigvector = b_k
        global A_eigvector = A_bk2
    end
    # print("Done \n")

    # it is better to implement an error check than a for loop using the number of iterations like below
    # input and output ratio 
    # norm(A^{n+1}*x)/norm(A^{n}*x) -> for a couple in a row -> ratio of the ratios
    # abs({n+2/n+1}-{n+1/n})/{n+1/n}
    # new ratio at each iteration 

    # The last b_k is the largest eigenvector of the linear operator A
    # calculate the eigenvalue corresponding to the largest eigenvector b_k 
    # using the formula: eigenvalue = (Ax * x)/(x*x), where x=b_k in our case
    # A*x=A*b_k is a G|v> type calculation
    # A_bk = output(l,b_k,cellsA)
    A_bk2_conj_tr = conj.(transpose(A_eigvector)) 
    bk_conj_tr = conj.(transpose(eigvector)) 
    eigenvalue = real((A_bk2_conj_tr*eigvector)/(bk_conj_tr*eigvector))[1]

    return b_k, eigenvalue
end

function power_iteration_second_evaluation(l,l2,cellsA,gMemSlfN,gMemSlfA, chi_inv_coeff, P)
    # Ideally choose a random vector to decrease the chance that our vector
    # is orthogonal to the eigenvector
    b_k = rand(ComplexF64, 3*cellsA[1]*cellsA[2]*cellsA[3], 1)
    # Let's do the first run of the power iteration method to get an intial largest eigenvalue
    # and corresponding largest eigenvector
    evaluation = power_iteration_first_evaluation(l,l2,cellsA,gMemSlfN, gMemSlfA, chi_inv_coeff, P)
    eigvect_1 = evaluation[1] 
    eigval_1 = evaluation[2] 

    # I didn't include P in the calculation since P = I for this case
    # b_k = rand(ComplexF64, 3*cellsA[1]*cellsA[2]*cellsA[3], 1)
    good = false 
    while good == false
        # Calculate the n+1 term
        # Calculate the operator-vector product (A linear operator and b_k vector)
        # G|v> type calculation
        A_bk = sym_and_asym_sum(l,l2,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, b_k) 

        # Let's do the (eigval*I-A)v> = eigval*|v>-A*|v> calculation
        eigval_A_bk = eigval_1*b_k .- A_bk # this the A_v>=A*b_k calculation
        # Calculate the norm
        eigval_A_bk_norm = norm(eigval_A_bk)
        # Renormalize the vector
        b_k1 = eigval_A_bk / eigval_A_bk_norm # this is the n+1 term

        # Calculate the n+2 term
        # G|v> type calculation
        A_bk1 = sym_and_asym_sum(l,l2,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, b_k1) 
        
        # Let's do the (eigval*I-A)v> = eigval*|v>-A*|v> calculation
        eigval_A_bk1 = eigval_1*b_k1 .- A_bk1 # this the A_v>=A*b_k calculation
        # Calculate the norm
        eigval_A_bk1_norm = norm(eigval_A_bk1)
        # Renormalize the vector
        b_k = eigval_A_bk1 / eigval_A_bk1_norm # this is the n+2 term
        # Technically b_k2 but it's called b_k since it will be the
        # b_k for the next iteration of the loop

        # Calculate the A*b_k product with the new b_k (the n+2 one)
        # G|v> type calculation
        A_bk2 = sym_and_asym_sum(l,l2,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, b_k) 

        # Let's do the (eigval*I-A)v> = eigval*|v>-A*|v> calculation
        eigval_A_bk2 = eigval_1*b_k .- A_bk2 # this the A_v>=A*b_k calculation
        # Calculate the ratios 
        # norm(A^{n+1}*x)/norm(A^{n}*x)
        ratio_n1_n = norm(eigval_A_bk1)/norm(eigval_A_bk)
        # norm(A^{n+2}*x)/norm(A^{n+1}*x)
        ratio_n2_n1 = norm(eigval_A_bk2)/norm(eigval_A_bk1)
        if abs(ratio_n2_n1-ratio_n1_n)/ratio_n1_n < 0.01
            good = true
        end
        global eigvector = b_k
        global A_eigvector = eigval_A_bk2
    end

    # The last b_k is the largest eigenvector of the linear operator A for the second run 
    # of the power iteration method
    # Calculate the eigenvalue corresponding to the largest eigenvector b_k 
    # using the formula: eigenvalue = (Ax * x)/(x*x), where x=b_k in our case
    # A*x=A*b_k is a G|v> type calculation
    # A_bk = output(l,b_k,cellsA)
    A_bk2_conj_tr = conj.(transpose(A_eigvector)) 
    bk_conj_tr = conj.(transpose(eigvector)) 
    eigenvalue_2 = real((A_bk2_conj_tr*eigvector)/(bk_conj_tr*eigvector))[1]

    # The minimum eigenvalue will be found by substracting the largest eigenvalue of the second run 
    # and that of the first run => lambda_2-lambda_1
    # will the minimum eigenvector be found by substracting the two eigenvectors?
    # min_eigval = eigenvalue_2 - eigval_1
    min_eigval = eigval_1 - eigenvalue_2 
    min_eigvec = eigvector - eigvect_1
    print("min_eigval ", min_eigval, "\n")
    return min_eigvec, min_eigval
end

function validityfunc(l,l2,cellsA,gMemSlfN, gMemSlfA, chi_inv_coeff, P)
    eval_2 = power_iteration_second_evaluation(l,l2,cellsA,gMemSlfN, gMemSlfA, chi_inv_coeff, P)
    
    min_eigvec = eval_2[1] # Not needed here but needed later
    min_eigval = eval_2[2]
    if min_eigval>0
        return 1
    else
        return -1
    end
end


# mineigfunc is the power_iteration_second_evaluation function
# l = initdof
function BFGS_fakeS_with_restart_pi(gMemSlfN, gMemSlfA,l,l2,dgfunc,P,chi_inv_coeff,ei,cellsA,validityfunc, 
    mineigfunc, gradConverge=false, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6)
    og_l = l
    og_l2 = l2
    dofnum = length(l)+length(l2)
    dualval = 0.0
    grad = zeros(Float64, dofnum, 1)
    tmp_grad = zeros(ComplexF64, dofnum, 1)
    Hinv = I #approximate inverse Hessian
    prev_dualval = Inf 
    dof = vcat(l,l2) # Combine the sym and asym L mults into one list 
    # dof = l # For when you only have one multiplier, so when l2 isn't needed (when you only
    # consider the asym constraint)
    objval=0.0
    function justfunc(d,d2,fSlist)
        return dgfunc(d,d2,Array([]),P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,fSlist,false)
    end
    olddualval = Inf #+0.0im
    reductCount = 0
    while true # Gradual reduction of modified source amplitude, outer loop
        fSlist = [] #remove old fake sources and restart with each outer iteration
        alpha_last = 1.0 # Reset step size
        Hinv = I # Reset Hinv
        val = dgfunc(l,l2,grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,fSlist,true)  
        # The line below is for when you only have sym or asym constraints
        # val = dgfunc(dof,grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,fSlist,true)  
        dualval = val[1]
        grad = val[2]
        obj = val[3]
        # T = val[4]
        # A = val[5]
        # b = val[6]
#             print("grad",grad,"\n")
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

            if length(l) > 0 && length(l2) == 0 # Asym constraints only
                while validityfunc(og_l.+alpha*pdir,og_l2,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)<=0 #move back into feasibility region
                    alpha *= c_reduct
                    print("alpha ", alpha, "\n")
                end  
            elseif length(l2) > 0 && length(l) == 0 # Sym constraints only 
                while validityfunc(og_l,og_l2.+alpha*pdir,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)<=0 #move back into feasibility region
                    alpha *= c_reduct
                    print("alpha ", alpha, "\n")
                end
            else # Sym and asym constraints 
                print("Sym and asym constraints \n")
                while validityfunc(og_l+alpha*pdir[1:length(l)],og_l2.+alpha*pdir[length(l)+1:end],cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)<=0 #move back into feasibility region
                    alpha *= c_reduct
                    print("alpha ", alpha, "\n")
                end
            end 

            alpha_feas = alpha
            print("alpha before backtracking is ", alpha_feas, "\n")
            alphaopt = alpha
            Dopt = Inf #+0.0im
            while true
        
                print("alpha ", alpha, "\n")
                print("pdir ", pdir, "\n")

                if length(l) > 0 && length(l2) == 0 # Asym constraints only
                    print("Asym constraints only \n")
                    tmp_dual = justfunc(og_l.+alpha*pdir,og_l2,fSlist)[1]
   
                elseif length(l2) > 0 && length(l) == 0 # Sym constraints only 
                    print("Sym constraints only \n")
                    tmp_dual = justfunc(og_l,og_l2.+alpha*pdir,fSlist)[1]
                else # Sym and asym constraints 
                    print("Sym and asym constraints \n")
                    tmp_dual = justfunc(og_l.+alpha*pdir[1:length(l)],og_l2.+alpha*pdir[length(l)+1:end],fSlist)[1]
                end 
                print("tmp_dual",tmp_dual,"\n")
                if tmp_dual < Dopt # The dual is still decreasing as we backtrack, continue
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

                    if length(l) > 0 && length(l2) == 0 # Asym constraints only
                        singular_dof_asym = og_l.+alpha_feas*pdir 
                        singular_dof_sym = og_l2
                    elseif length(l2) > 0 && length(l) == 0 # Sym constraints only 
                        singular_dof_sym = og_l2.+alpha_feas*pdir 
                        singular_dof_asym = og_l
                    else # Sym and Asym constraints 
                        singular_dof_asym = og_l.+alpha_feas*pdir[1:length(l)] 
                        singular_dof_sym = og_l2.+alpha_feas*pdir[length(l)+1:end]
                    end 
                    print("singular_dof_asym ", singular_dof_asym , "\n")
                    print("singular_dof_sym ", singular_dof_sym , "\n")
                    eig_fct_eval = mineigfunc(singular_dof_asym,singular_dof_sym,cellsA,gMemSlfN,gMemSlfA, chi_inv_coeff, P)
                    mineigv=eig_fct_eval[1] # eigenvector 
                    mineigw=eig_fct_eval[2] # eigenvalue
                   
                    fakeS_eval = dgfunc(og_l,og_l2,Array([]),P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,[mineigv],false)
                    
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
            if length(l) > 0 && length(l2) == 0 # Asym constraints only 
                tmp_val = dgfunc(og_l  + delta,og_l2,tmp_grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,fSlist, true)
            elseif length(l2) > 0 && length(l) == 0 # Sym constraints only 
                tmp_val = dgfunc(og_l,og_l2 + delta,tmp_grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,fSlist, true)
            else # Sym and asym constraints 
                print("Sym and asym constraints \n")
                tmp_val = dgfunc(og_l + delta[1:length(l)],og_l2 + delta[length(l)+1:end],tmp_grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff,cellsA,fSlist, true)
            end 
            
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

            if length(l) > 0 && length(l2) == 0 # Asym constraints only 
                og_l .+= delta 
            elseif length(l2) > 0 && length(l) == 0 # Sym constraints only 
                og_l2 .+= delta 
            else # Sym and asym constraints 
                og_l .+= delta[1:length(l)]
                og_l2 += delta[length(l)+1:end]
            end 

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
