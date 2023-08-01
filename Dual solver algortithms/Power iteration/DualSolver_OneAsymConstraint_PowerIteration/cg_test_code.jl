using LinearAlgebra
# function conj_grad(A,b)
#     tol = 1e-5 # The program terminates once 
#     # there is an r for which its norm is smaller
#     # than the chosen tolerance. 

#     xk = zeros(ComplexF64,length(b),1)
#     # Ax=0 since the initial xk is 0
#     pk = rk = b 
#     # k = 0
#     for k in 1:length(b)
#         # alpha_k coefficient calculation 
#         # Top term
#         rkrk = conj.(transpose(rk))*rk
#         # Bottom term 
#         A_pk = A*pk
#         pk_A_pk = conj.(transpose(pk))*A_pk
#         # Division
#         alpha_k = rkrk/pk_A_pk

#         # x_{k+1} calculation 
#         xk = xk + alpha_k.*pk

#         # r_{k+1} calculation 
#         rk = rk - alpha_k.*A_pk

#         # print("norm(rk_plus1) ",norm(rk), "\n")
#         # if norm(rk)<tol
#         #     return xk
#         # end

#         # beta_k coefficient calculation 
#         # Top term 
#         rkplus1_rkplus1 = conj.(transpose(rk))*rk
#         # The bottom term is the same one calculated earlier 
#         # Division 
#         beta_k = rkplus1_rkplus1/rkrk

#         pk = rk + beta_k.*pk

#         # global k+=1 

#     end
#     return xk
# end 

function bicgstab(A,b)
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
        
        vk = A*pk

        # alpha calculation
        # Bottom term
        bottom_term = conj.(transpose(r0))*vk
        # Calculation 
        alpha = rho_k/bottom_term 
        
        h = xk_m1 + alpha.*pk
        # If h is accurate enough, then set xk=h and quantity
        # What does accurate enough mean?

        s = r_m1 - alpha.*vk

        t = A*s

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

A = zeros(Int8, 2,2)
A[1,1] = 4
A[1,2] = 1
A[2,1] = 1
A[2,2] = 3

b = zeros(Int8, 2,1)
b[1,1]=1
b[2,1]=2

# x_julia = cg(A, b)
print("x  ", bicgstab(A,b))