module cg_asym_only 
export cg
using product, LinearAlgebra, vector
# Based on the example code from https://en.wikipedia.org/wiki/Conjugate_gradient_method
# Code for the AA case 

# m is the maximum number of iterations
function cg(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)
    # tol = 1e-5 # The program terminates once 
    # there is an r for which its norm is smaller
    # than the chosen tolerance. 

    xk = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    pk = rk = b 
    # k = 0
    for k in 1:length(b)
        # alpha_k coefficient calculation 
        # Top term
        rkrk = conj.(transpose(rk))*rk
        # Bottom term 
        A_pk = (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, pk)
        pk_A_pk = conj.(transpose(pk))*A_pk
        # Division
        alpha_k = rkrk/pk_A_pk

        # x_{k+1} calculation 
        xk = xk + alpha_k.*pk

        # r_{k+1} calculation 
        rk = rk - alpha_k.*A_pk

        # print("norm(rk_plus1) ",norm(rk), "\n")
        # if norm(rk)<tol
        #     return xk
        # end

        # beta_k coefficient calculation 
        # Top term 
        rkplus1_rkplus1 = conj.(transpose(rk))*rk
        # The bottom term is the same one calculated earlier 
        # Division 
        beta_k = rkplus1_rkplus1/rkrk

        pk = rk + beta_k.*pk

        # global k+=1 

    end
    return xk
end 
end 
    