module bicgstab_asym_only 
export bicgstab
using product, LinearAlgebra, vector
# Based on the example code from p. 686 (or p.696 of the pdf) of the Introduction to Numerical Analysis textbook
# Code for the AA case 
# This is a biconjugate gradient program without a preconditioner
# m is the maximum number of iterations
function bicgstab(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b 
    rho_m1 = alpha = omega_m1 = 1
    for k in 1:length(b)
        rho_k = conj.(transpose(r0))*r_m1
        
        # beta calculation
        # First term 
        first_term = rho_k/rho_m1
        # Second term 
        second_term = alpha/omega_m1
        # Calculation 
        beta = first_term*second_term
        
        pk = r_m1 + beta.*(p_m1-omega_m1.*v_m1)
        
        vk = sym_and_asym_sum(l,l2,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, pk)
        # (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, pk)

        # alpha calculation
        # Bottom term
        bottom_term = conj.(transpose(r0))*vk
        # Calculation 
        alpha = rho_k/bottom_term 
        
        h = xk_m1 + alpha.*pk
        # If h is accurate enough, then set xk=h and quantity
        # What does accurate enough mean?

        s = r_m1 - alpha.*vk

        t = sym_and_asym_sum(l,l2,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, s)
        # (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, s)

        # omega_k calculation 
        # Top term 
        ts = conj.(transpose(t))*s
        # Bottom term
        tt = conj.(transpose(t))*t
        # Calculation 
        omega_k = ts/tt

        xk_m1 = h + omega_k.*s

        r_m1 = s-omega_k.*t

    end
    return xk_m1
end
end