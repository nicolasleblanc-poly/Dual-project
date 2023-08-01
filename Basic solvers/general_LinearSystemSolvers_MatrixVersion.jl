using LinearAlgebra, Random
# Based on the example code from p. 686 (or p.696 of the pdf) of the Introduction to Numerical Analysis textbook
# Code for the AA case 
# This is a biconjugate gradient program without a preconditioner
# m is the maximum number of iterations


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

function GMRES_with_restart(A, b, gmres_tol,max_nb_it) # add cellsB for the BA case 
    n = length(b)
    max_nb_it = 20
    Q = zeros(ComplexF64,n,max_nb_it+1)
    Q[:,1] = reshape(b, (n,1))/norm(b)
    H = zeros(ComplexF64,max_nb_it+1,max_nb_it)
    # Initial solution is zero.
    x = 0
    
    residual = [norm(b);zeros(max_nb_it)]
    for j in 1:max_nb_it
        v = A*Q[:,j]
        for i in 1:j
            H[i,j] = dot(Q[:,i],v)
            v -= H[i,j]*Q[:,i] 
        end
        H[j+1,j] = norm(v)
        Q[:,j+1] = v/H[j+1,j]
        r = [norm(b); zeros(ComplexF64,j)]
        z = H[1:j+1,1:j] \ r # I took out the +1 in the two indices of H.
        x = Q[:,1:j]*z# I removed a +1 in the 2nd index of Q
        # second G|v> type calculation
        value = A*x
        # output(l,x,cellsA)
        residual[j+1] = norm(value - b)
        if norm(residual) < gmres_tol
            return x 
        end
    end
    return x 
end

function bicg_matrix(A,b,bicg_tol,max_nb_it)
    xk = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    pk = rk = b 
    # Pk = Rk = conj.(transpose(rk))
    # for k in 1:length(b)
    for k in 1:max_nb_it
        # Step 1
        # a_k coefficient calculation 
        # Top term
        rkrk = conj.(transpose(rk))*rk
        # Bottom term 
        A_pk = A*pk
        pk_A_pk = conj.(transpose(pk))*A_pk
        # Division
        a_k = rkrk/pk_A_pk

        # x_{k+1} calculation 
        xk = xk + a_k.*pk

        # r_{k+1} calculation 
        rk = rk - a_k.*A_pk

        if norm(rk)  < bicg_tol
            return xk
        end

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
        # Pk = Rk + b_k.*Pk

        # global k+=1 



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

tol_cg = 1e-6
tol_bicgstab = 1e-6
tol_bicg = 1e-6
tol_gmres = 1e-6
max_nb_it = 1000
print("A ", A, "\n")
print("Test bicgstab ", bicgstab(A,b,tol_bicgstab,max_nb_it), "\n")
print("Test bicg ", bicg_matrix(A,b,tol_bicg,max_nb_it), "\n")
print("Test cg ", cg_matrix(A,b,tol_cg,max_nb_it), "\n")
# print("Test gmres ", GMRES_with_restart(A,b,tol_gmres,max_nb_it), "\n")
print("Direct solve ", A\b, "\n")

