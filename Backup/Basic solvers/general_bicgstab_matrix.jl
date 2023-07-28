using LinearAlgebra, Random
# Based on the example code from p. 686 (or p.696 of the pdf) of the Introduction to Numerical Analysis textbook
# Code for the AA case 
# This is a biconjugate gradient program without a preconditioner
# m is the maximum number of iterations

# function bicgstab_test(A,b,tol_bicgstab,nb_it)
#     v_m1 = p_m1 = zeros(ComplexF64,length(b),1)
#     # Ax=0 since the initial xk is 0

#     # New code 
#     xk_m1 = Vector{ComplexF64}(undef, vecDim)
#     rand!(xk_m1)
#     r_m1 = b-A*xk_m1
#     r0_hat = Vector{ComplexF64}(undef, vecDim)
#     rand!(r0_hat)

#     # Old code 
    
#     # r0 = r_m1 = b 
#     rho_m1 = alpha = omega_m1 = 1
#     for k in 1:nb_it
#         # print("r_m1 ", r_m1, "\n")
#         rho_k = dot(r0_hat,r_m1) # real((adjoint(r0)*r_m1)[1])
#         # print("rho_k ", rho_k, "\n")
#         # beta calculation
#         # First term 
#         first_term = rho_k/rho_m1
#         # Second term 
#         second_term = alpha/omega_m1
#         # Calculation 
#         beta = first_term*second_term
#         pk = r_m1 + beta.*(p_m1-omega_m1.*v_m1)
#         vk = A*pk
#         # alpha calculation
#         # Bottom term
#         bottom_term = dot(r0_hat,vk) #real((adjoint(r0)*vk)[1])
#         # Calculation 
#         alpha = rho_k/bottom_term 
#         h = xk_m1 + alpha.*pk
#         s = r_m1 - alpha.*vk

#         if norm(s)/norm(b) < tol_bicgstab
#             return h,k # k is essentially the number of iterations 
# 			# to reach the chosen tolerance
#         end 

#         t = A*s
#         # omega_k calculation 
#         # Top term 
#         ts = dot(t,s) # real((adjoint((t))*s)[1])
#         # Bottom term
#         tt = dot(t,t) # real((adjoint(t)*t)[1])
#         # Calculation 
#         omega_k = ts/tt
#         xk_m1 = h + omega_k.*s
#         r_m1 = s-omega_k.*t
#         if norm(r_m1)/norm(b) < tol_bicgstab
# 			print("r_m1 ", r_m1, "\n")
# 			return xk_m1,nb_it # k is essentially the number of iterations 
# 			# to reach the chosen tolerance
#         end
#         rho_m1 = rho_k
#         omega_m1 = omega_k
#         v_m1 = vk
#         nb_it += 1
#     end
#     print("Reached max number of iterations \n")
#     print("r_m1 ", r_m1, "\n")
#     return xk_m1,nb_it
# end


# function bicgstab(A,b,tol_bicgstab,max_nb_it)
#     vecDim = length(b)
#     xk = Vector{Float64}(undef, vecDim)
#     rand!(xk)
#     rk = b-A*xk
#     r0_hat = Vector{Float64}(undef, vecDim)
#     rand!(r0_hat)
#     print("dot(r0_hat,rk_m1) ", dot(r0_hat,rk),"\n")
#     pk = rk
#     alpha = 1
#     nb_it = 0
#     for k in 1:max_nb_it
#         # alpha calculation 
#         A_pk = A*pk
#         alpha_top_term = dot(r0_hat,rk)
#         alpha_bottom_term = dot((A_pk),r0_hat)
#         alpha = alpha_top_term/alpha_bottom_term

#         s = rk-alpha*A_pk

#         # omega calculation 
#         A_s = A*s
#         omega_top_term = dot(A_s,s)
#         omega_bottom_term = dot(A_s,A_s)
#         omega = omega_top_term/omega_bottom_term

#         xk = xk + alpha*pk + omega*s

#         rk = s - omega*A_s

#         # if norm(rk)/norm(b) < tol_bicgstab
#         print("norm(rk) ", norm(rk), "\n")
#         if norm(rk) < tol_bicgstab
# 			# print("rk ", rk, "\n")
#             print("norm(rk) ", norm(rk), "\n")
# 			return xk
#         end
#         # bêta calculation 
#         beta = (alpha/omega)*(dot(rk,r0_hat)/dot(rk,r0_hat))
#         pk = rk + beta*(pk-omega*A_pk)
#         nb_it += 1
#     end
#     print("xk ", xk, "\n")
#     print("nb_it ", nb_it, "\n")
#     print("Reached max number of iterations \n")
#     print("Restart a new function call \n")
#     xk = bicgstab(A,b,tol_bicgstab,max_nb_it)
#     return xk
# end

function bicgstab(A,b,tol_bicgstab,max_nb_it)
	dim = size(A)[1]
    vk = pk = xk = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    r0_hat = rk = b 
    rho_m1 = alpha = omega_k = 1
	k = 0
    for k in 1 : max_nb_it
        rho_k = dot(r0_hat,rk) # conj.(transpose(r0))*r_m1  
        # Bêta calculation
        # First term 
        first_term = rho_k/rho_m1
        # Second term 
        second_term = alpha/omega_k
        # Calculation 
        beta = first_term*second_term
        pk = rk .+ beta.*(pk-omega_k.*vk)
        # pkPrj = harmVec(dim, hTrg, hSrc, prjC, pk)
        vk = A*pk
        # alpha calculation
        # Bottom term
        bottom_term = dot(r0_hat,vk) # conj.(transpose(r0))*vk
        # Calculation 
        alpha = rho_k / bottom_term 
        h = xk + alpha.*pk
        s = rk - alpha.*vk
        # sPrj = harmVec(dim, hTrg, hSrc, prjC, s)
        t = A*s
        # omega_k calculation 
        # Top term 
        ts = dot(t,s) # conj.(transpose(t))*s
        # Bottom term
        tt = dot(t,t) # conj.(transpose(t))*t
        # Calculation 
        omega_k = ts ./ tt
        xk = h + omega_k.*s
        # r_old = r_m1
        rk = s-omega_k.*t
		rho_m1 = rho_k
		print("norm(rk) ", norm(rk), "\n")
        if norm(rk) < tol_bicgstab
			print("Converged off bicgstab residual tolerence \n")
			return xk # k is essentially the number of iterations 
			# to reach the chosen tolerance
        end
    end
	print("Didn't converge off bicgstab tolerence \n")
    return xk # k is essentially the number of iterations 

end

function cg_matrix(A,b,cg_tol,max_nb_it)
    # tol = 1e-5 # The program terminates once 
    # there is an r for which its norm is smaller
    # than the chosen tolerance. 

    xk = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    pk = rk = b 
    # k = 0
    # for k in 1:length(b)
    for k in 1:max_nb_it
        # alpha_k coefficient calculation 
        # Top term
       
        rkrk = dot(rk,rk) # conj.(transpose(rk))*rk
        print("rkrk ", rkrk, "\n")
        # Bottom term 
        A_pk = A*pk
        pk_A_pk = dot(pk,A_pk) # conj.(transpose(pk))*A_pk
        # Division
        alpha_k = rkrk/pk_A_pk
        print("alpha_k ", alpha_k, "\n")

        # x_{k+1} calculation 
        xk = xk + alpha_k.*pk

        # r_{k+1} calculation 
        rk = rk - alpha_k.*A_pk

        # print("norm(rk_plus1) ",norm(rk), "\n")
        print("norm(rk) ", norm(rk), "\n")
        if norm(rk)  < cg_tol
            return xk
        end

        # beta_k coefficient calculation 
        # Top term 
        rkplus1_rkplus1 = dot(rk,rk) # conj.(transpose(rk))*rk
        # The bottom term is the same one calculated earlier 
        # Division 
        # print("rkplus1_rkplus1 ", rkplus1_rkplus1, "\n")
        # print("rkrk ", rkrk, "\n")

        beta_k = rkplus1_rkplus1/rkrk

        pk = rk + beta_k.*pk

    end
    return xk
end 

A = Array{Float64}(undef, 2, 2)
A[1,1] = 4
A[1,2] = 1
A[2,1] = 1
A[2,2] = 3 

b = Array{Float64}(undef, 2, 1)
b[1,1] = 1
b[2,1] = 2

tol_cg = 1e-4
tol_bicgstab = 1e-6
max_nb_it = 1000
print("A ", A, "\n")
print("Test bicgstab ", bicgstab(A,b,tol_bicgstab,max_nb_it), "\n")
print("Test cg ", cg_matrix(A,b,tol_cg,max_nb_it), "\n")
print("Direct solve ", A\b, "\n")

# vecDim = 2
# tol_bicgstab = 1e-12
# A = Array{ComplexF64}(undef, vecDim,vecDim)
# A[1,1] = 4
# A[1,2] = 1
# A[2,1] = 1
# A[2,2] = 3
# b = Vector{ComplexF64}(undef, vecDim)
# b[1,1] = 1
# b[2,1] = 2
# direct_solve = A\b
# print("direct solve ", direct_solve,"\n")

# direct_solve_residual = b-A*direct_solve
# print("norm(direct_solve_residual), ", norm(direct_solve_residual), "\n")

# # Jacobi-Davidson direction
# basis = bicgstab(A,b,tol_bicgstab)
# print("basis ", basis, "\n")

# vecDim = 3
# tol_bicgstab = 1e-4
# A = Array{ComplexF64}(undef, vecDim,vecDim)
# b = Vector{ComplexF64}(undef, vecDim)
# rand!(A)
# rand!(b)
# A = (adjoint(A)+A)/2
# print("det(opt) ", det(A), "\n")

# print("Test ", cg_matrix(A,b,cg_tol), "\n")
# print("Direct solve ", A\b, "\n")

# # Jacobi-Davidson direction
# max_nb_it = 1000
# basis,nb_it = bicgstab_test(A,b,tol_bicgstab,max_nb_it)
# # basis,nb_it = bicgstab(A,b,tol_bicgstab,max_nb_it)
# print("basis ", basis, "\n")
# print("nb_it ", nb_it, "\n")

# direct_solve = A\b
# print("direct solve ", direct_solve,"\n")

# direct_solve_residual = b-A*direct_solve
# print("norm(direct_solve_residual), ", norm(direct_solve_residual), "\n")