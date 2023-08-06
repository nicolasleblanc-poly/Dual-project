"""
This module is for the biconjugate gradient program without a preconditionner. The written function is for the 
AA case of the Green function. 

Author: Nicolas Leblanc
"""

module bicg_asym_only 
export bicg
using product, LinearAlgebra, vector, Base.Threads 
# Based on the example code from p. 686 (or p.696 of the pdf) of the Introduction to Numerical Analysis textbook
# Code for the AA case 
# This is a biconjugate gradient program without a preconditioner
# m is the maximum number of iterations
function bicg(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)
    xk = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    pk = rk = b 
    for k in 1:length(b)
        # Step 1
        # a_k coefficient calculation 
        # Top term
        rkrk = conj.(transpose(rk))*rk
        # Bottom term 
        A_pk = (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, pk)
        pk_A_pk = conj.(transpose(pk))*A_pk
        # Division
        a_k = rkrk/pk_A_pk

        # x_{k+1} calculation 
        xk = xk + a_k.*pk

        # r_{k+1} calculation 
        rk = rk - a_k.*A_pk

        # R_{k+1} calculation 
        # Rk = Rk - a_k.*conj.(transpose(A_pk)) # A^T... not sure how to do this here since we have operators and not matrices 

        # Step 2
        # b_k coefficient calculation 
        # Top term 
        rkplus1_rkplus1 = conj.(transpose(rk))*rk
        # The bottom term is the same one calculated earlier 
        # Division 
        b_k = rkplus1_rkplus1/rkrk

        pk = rk + b_k.*pk

    end
    return xk
end
end