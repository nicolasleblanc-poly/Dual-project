"""
The Restart_Ritz_jacobiDavidson module allows to calculate the extremal 
eigenvalue, which can be positive or negative, of a given matrix.

Author: Sean Molesky & Nicolas Leblanc
"""

module Restart_Ritz_jacobiDavidson
export ritz_bicgstab_matrix,jacDavRitz_basic,jacDavRitz_basic_for_restart,
	jacDavRitz_restart
using LinearAlgebra, Random 

@inline function projVec(dim::Integer, pVec::Vector{T}, sVec::Array{T})::Array{T} where T <: Number

	return sVec .- (BLAS.dotc(dim, pVec, 1, sVec, 1) .* pVec)
end

# This is a biconjugate gradient program without a preconditioner. 
# m is the maximum number of iterations
function ritz_bicgstab_matrix(A, theta, u, b, tol_bicgstab)
	dim = size(A)[1]
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b 
    rho_m1 = alpha = omega_m1 = 1
	k = 0
    for k in 1 : 100000
        rho_k = real((adjoint(r0)*r_m1)[1])
        # Bêta calculation
        # First term 
		# print("rho_k ", rho_k, "\n")
        first_term = rho_k/rho_m1
        # Second term 
        second_term = alpha/omega_m1
        # Calculation 
        beta = first_term*second_term
        pk = r_m1 .+ beta.*(p_m1-omega_m1.*v_m1)
        pkPrj = projVec(dim, u, pk)
        vk = projVec(dim, u, A * pkPrj .- (theta .* pkPrj))
        # alpha calculation
        # Bottom term
        bottom_term = real((adjoint(r0)*vk)[1])
        # Calculation 
        alpha = rho_k / bottom_term 
        h = xk_m1 + alpha.*pk
        s = r_m1 - alpha.*vk
        sPrj = projVec(dim, u, s) 
        t = projVec(dim, u, A * sPrj .- (theta .* sPrj))
        # omega_k calculation 
        # Top term 
        ts = conj.(transpose(t))*s
        # Bottom term
        tt = conj.(transpose(t))*t
        # Calculation 
        omega_k = ts ./ tt
        xk_m1 = h + omega_k.*s
        # r_old = r_m1
        r_m1 = s-omega_k.*t
		rho_m1 = rho_k
		omega_m1 = omega_k
		v_m1 = vk
        if norm(r_m1) < tol_bicgstab
			print("r_m1 ", r_m1, "\n")
			return xk_m1,k # k is essentially the number of iterations 
			# to reach the chosen tolerance
        end
    end
	print("Reached max number of iterations \n")
    return xk_m1,k # k is essentially the number of iterations 
end

function ritz_cg_matrix(A,theta,u,b,tol_cg)
    # tol = 1e-5 # The program terminates once 
    # there is an r for which its norm is smaller
    # than the chosen tolerance. 
	dim = size(A)[1]
    xk = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    pk = rk = b 
    # k = 0
    # for k in 1:length(b)
	nb_it = 0
    for k in 1:100000
        # alpha_k coefficient calculation 
        # Top term
        rkrk = dot(rk,rk) # conj.(transpose(rk))*rk
        
		# Bottom term 
        pkPrj = projVec(dim, u, pk)
        A_pkPrj = projVec(dim, u, A * pkPrj .- (theta .* pkPrj))
        # A_pk = A*pk
        pk_A_pkPrj = dot(pk,A_pkPrj) # conj.(transpose(pk))*A_pk
        # Division
        alpha_k = rkrk/pk_A_pkPrj

        # x_{k+1} calculation 
        xk = xk + alpha_k.*pk

        # r_{k+1} calculation 
        rk = rk - alpha_k.*A_pkPrj

        # print("norm(rk_plus1) ",norm(rk), "\n")
        if norm(rk)  < tol_cg
            return xk,nb_it
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
		nb_it += 1
    end
    return xk,nb_it
end 

# function ritz_bicgstab_matrix(A, theta, u, b, tol_bicgstab)
# 	max_nb_it = 10000
# 	dim = size(A)[1]
#     xk_m1 = Vector{Float64}(undef, dim)
#     rand!(xk_m1)
#     rk_m1 = b-A*xk_m1
#     r0_hat = Vector{Float64}(undef, dim)
#     rand!(r0_hat)
#     print("dot(r0_hat,rk_m1) ", dot(r0_hat,rk_m1),"\n")
#     pk_m1 = rk_m1
#     alpha = 1
#     nb_it = 0
#     for k in 1 : max_nb_it
# 		# alpha calculation 
# 		pkPrj = projVec(dim, u, pk_m1)
# 		A_pk_m1 = projVec(dim, u, A * pkPrj .- (theta .* pkPrj))
# 		# A_pk_m1 = A*pk_m1
# 		alpha_top_term = dot(r0_hat,rk_m1)
# 		alpha_bottom_term = dot((A_pk_m1),r0_hat)
# 		alpha = alpha_top_term/alpha_bottom_term

# 		s = rk_m1-alpha*A_pk_m1

# 		# omega calculation 
# 		sPrj = projVec(dim, u, s) 
# 		A_s = projVec(dim, u, A * sPrj .- (theta .* sPrj))
# 		# A_s = A*s
# 		omega_top_term = dot(A_s,s)
# 		omega_bottom_term = dot(A_s,A_s)
# 		omega = omega_top_term/omega_bottom_term

# 		xk_m1 = xk_m1 + alpha*pk_m1 + omega*s

# 		rk = s - omega*A_s

# 		# if norm(rk)/norm(b) < tol_bicgstab
# 		if norm(rk) < tol_bicgstab
# 			# print("rk ", rk, "\n")
# 			print("norm(rk) ", norm(rk), "\n")
# 			return xk_m1,nb_it 
# 		end
# 		# bêta calculation 
# 		beta = (alpha/omega)*(dot(rk,r0_hat)/dot(rk_m1,r0_hat))
# 		pk_m1 = rk + beta*(pk_m1-omega*A_pk_m1)
# 		rk_m1 = rk
# 		nb_it += 1
#     end
# 	print("xk_m1 ", xk_m1, "\n")
#     print("nb_it ", nb_it, "\n")
#     print("Reached max number of iterations \n")
#     print("Restart a new function call \n")
#     xk_m1,nb_it = ritz_bicgstab_matrix(A, theta, u, b, tol_bicgstab)
#     return xk_m1,nb_it
# end

# vecDim = 3
# theta = 2
# # max_nb_it = 500
# tol_bicgstab = 1e-4
# opt = Array{ComplexF64}(undef, vecDim,vecDim)
# resVec = Vector{ComplexF64}(undef, vecDim)
# ritzVec = Vector{ComplexF64}(undef, vecDim)
# rand!(opt)
# rand!(resVec)
# rand!(ritzVec)
# print("det(opt) ", det(opt), "\n")

# A_product = (I-ritzVec*adjoint(ritzVec))*(opt-theta*I)*(I-ritzVec*adjoint(ritzVec))
# direct_solve = A_product\resVec
# print("direct solve ", direct_solve,"\n")

# direct_solve_residual = resVec-A_product*direct_solve
# print("direct_solve_residual, ", direct_solve_residual, "\n")

# # Jacobi-Davidson direction
# basis,nb_it = ritz_bicgstab_matrix(opt, theta, ritzVec, resVec, tol_bicgstab)

# print("basis ", basis, "\n")


function gramSchmidt!(basis::Array{T}, n::Integer, tol::Float64) where T <: Number
	# Dimension of vector space
	dim = size(basis)[1]
	# Orthogonality check
	prjNrm = 1.0;
	# Check that basis does not exceed dimension
	if n > dim
		error("Requested basis size exceeds dimension of vector space.")
	end
	# Norm calculation
	nrm = BLAS.nrm2(dim, view(basis,:,n), 1)
	# Renormalize new vector
	basis[:,n] = basis[:,n] ./ nrm
	nrm = BLAS.nrm2(dim, view(basis,:,n), 1)
	# Guarded orthogonalization
	while prjNrm > (tol * 100) && abs(nrm) > tol
		# Remove projection into existing basis
 		BLAS.gemv!('N', -1.0 + im*0.0, view(basis, :, 1:(n-1)), 
 			BLAS.gemv('C', view(basis, :, 1:(n-1)), view(basis, :, n)), 
 			1.0 + im*0.0, view(basis, :, n)) 
 		# Recalculate the norm
 		nrm = BLAS.nrm2(dim, view(basis,:,n), 1) 
 		# Calculate projection norm
 		prjNrm = BLAS.nrm2(n-1, BLAS.gemv('C', 
 			view(basis, :, 1:(n-1)), view(basis, :, n)), 1) 
 	end
	# Check that remaining vector is sufficiently large
	if abs(nrm) < tol
		# Switch to random basis vector
		rand!(view(basis, :, n))
		gramSchmidt!(basis, n, tol)		
	else
		# Renormalize orthogonalized vector
		basis[:,n] = basis[:,n] ./ nrm
	end 
end

function jacDavRitz_basic(opt,
		innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64,
			tol_eigval::Float64,tol_bicgstab::Float64)::Tuple{Float64,Float64,
				Float64}
	# Memory initialization
	dims = size(opt) # opt is a square matrix, so dims[1]=dims[2]
	vecDim = dims[1]
	basis = Array{ComplexF64}(undef, vecDim, vecDim)
	hesse = zeros(ComplexF64, vecDim, vecDim)
	outVec = Vector{ComplexF64}(undef, vecDim)
	resVec = Vector{ComplexF64}(undef, vecDim)
	ritzVec = Vector{ComplexF64}(undef, vecDim)
	# Set starting vector
	rand!(view(basis, :, 1))
	# Normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(basis,:,1), 1)
	basis[:, 1] = basis[:, 1] ./ nrm
	# Algorithm initialization
	outVec = opt * basis[:, 1] 
	# Hessenberg matrix
	hesse[1,1] = BLAS.dotc(vecDim, view(basis, :, 1), 1, outVec, 1) 
	# Ritz value
	theta = hesse[1,1] 
	# Ritz vector
	ritzVec[:] = basis[:, 1]
	# Negative residual vector
	resVec = (theta .* ritzVec) .- outVec

	# Initialize some parameters
	previous_eigval = theta
	nb_it_vals_basic = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0

	for itr in 2 : innerLoopDim
		# Jacobi-Davidson direction
		basis[:, itr],nb_it = ritz_cg_matrix(opt, theta, ritzVec, resVec, 
			tol_bicgstab)
		# basis[:, itr],nb_it = ritz_bicgstab_matrix(opt, theta, ritzVec, resVec, 
		# 	tol_bicgstab)
		
		print("basis[:, itr] ", basis[:, itr], "\n")
		A_product = (I-ritzVec*adjoint(ritzVec))*(opt-theta*I)*(I-ritzVec*adjoint(ritzVec))
		print("direct solve ", A_product\resVec,"\n")

		nb_it_total_bicgstab_solve += nb_it

		# Orthogonalize
		gramSchmidt!(basis, itr, tol_MGS)
		# New image
		outVec = opt * basis[:, itr] 
		# Update Hessenberg
		hesse[1 : itr, itr] = BLAS.gemv('C', view(basis, :, 1 : itr), outVec)
		hesse[itr, 1 : (itr - 1)] = conj(hesse[1 : (itr - 1), itr])
		# Eigenvalue decomposition, largest real eigenvalue last. 
		# should replace by BLAS operation
		eigSys = eigen(view(hesse, 1 : itr, 1 : itr)) 
		# Update Ritz vector
		theta = eigSys.values[end]
		ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, end])
		outVec = opt * ritzVec
		# Update residual vector
		resVec = (theta * ritzVec) .- outVec

		# Direction vector tolerance check 
		if norm(resVec) < tol_conv
			print("Basic algo converged off resVec tolerance \n")
			return real(theta),nb_it_vals_basic,nb_it_total_bicgstab_solve #,ritzVec
		end
		# Eigenvalue tolerance check
		if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
			if nb_it_eigval_conv == 5
				print("Basic algo converged off eigval tolerance \n")
				return real(theta),nb_it_vals_basic,nb_it_total_bicgstab_solve
			end 
			nb_it_eigval_conv += 1
		end
		previous_eigval = theta
		nb_it_vals_basic += 1
	end
	print("Didn't converge off tolerance for basic program. 
		Atteined max set number of iterations \n")
	return (real(theta),nb_it_vals_basic,nb_it_total_bicgstab_solve) # ,ritzVec
end

function jacDavRitz_basic_for_restart(opt, 
			innerLoopDim::Integer,tol_MGS::Float64,
				tol_conv::Float64,tol_eigval::Float64,
					tol_bicgstab::Float64)		
	# Memory initialization
	dims = size(opt) # opt is a square matrix, so dims[1]=dims[2]
	vecDim = dims[1]
	basis = Array{ComplexF64}(undef, vecDim, vecDim)
	hesse = zeros(ComplexF64, vecDim, vecDim)
	outVec = Vector{ComplexF64}(undef, vecDim)
	resVec = Vector{ComplexF64}(undef, vecDim)
	ritzVec = Vector{ComplexF64}(undef, vecDim)
	# Set starting vector
	rand!(view(basis, :, 1))
	# Normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(basis,:,1), 1)
	basis[:, 1] = basis[:, 1] ./ nrm
	# Algorithm initialization
	outVec = opt * basis[:, 1] 
	# Hessenberg matrix
	hesse[1,1] = BLAS.dotc(vecDim, view(basis, :, 1), 1, outVec, 1) 
	# Ritz value
	theta = hesse[1,1] 
	# Ritz vector
	ritzVec[:] = basis[:, 1]
	# Negative residual vector
	resVec = (theta .* ritzVec) .- outVec

	# Initialize some parameters 
	previous_eigval = theta
	nb_it_vals_basic_for_restart = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)

	for itr in 2 : innerLoopDim
		# Test code
		# nb_it = 0 # Just for a test

		# A_product = (I-ritzVec*adjoint(ritzVec))*(opt-theta*I)*(I-ritzVec*adjoint(ritzVec))
		# # direct_solve =  A_product\resVec
		# basis[:, itr] = A_product\resVec # Just for a test
		# # print("direct_solve ", direct_solve,"\n")

		# Good code 
		# Jacobi-Davidson direction
		basis[:, itr],nb_it = ritz_cg_matrix(opt, theta, ritzVec, resVec, 
			tol_bicgstab)
		# basis[:, itr],nb_it = ritz_bicgstab_matrix(opt, theta, ritzVec, resVec, 
		# 	tol_bicgstab)

		print("basis[:, itr] ", basis[:, itr], "\n")

		A_product = (I-ritzVec*adjoint(ritzVec))*(opt-theta*I)*(I-ritzVec*adjoint(ritzVec))
		print("direct solve ", A_product\resVec,"\n")

		nb_it_total_bicgstab_solve += nb_it
		# Orthogonalize
		gramSchmidt!(basis, itr, tol_MGS)
		# New image
		outVec = opt * basis[:, itr] 
		# Update Hessenberg
		hesse[1 : itr, itr] = BLAS.gemv('C', view(basis, :, 1 : itr), outVec)
		hesse[itr, 1 : (itr - 1)] = conj(hesse[1 : (itr - 1), itr])
		# Eigenvalue decomposition, largest real eigenvalue last. 
		# should replace by BLAS operation
		eigSys = eigen(view(hesse, 1 : itr, 1 : itr)) 
		
		eigenvectors[1:itr,1:itr] = eigSys.vectors 
		eigenvalues[1:itr] = eigSys.values
		
		# Update Ritz vector
		theta = eigSys.values[end]
		ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, end])
		outVec = opt * ritzVec
		# Update residual vector
		resVec = (theta * ritzVec) .- outVec

		# Direction vector tolerance check 
		if norm(resVec) < tol_conv
			print("Basic algo converged off resVec tolerance \n")
			return (real(theta),basis,hesse,outVec,ritzVec,resVec,eigenvalues,
				eigenvectors,nb_it_vals_basic_for_restart,nb_it_total_bicgstab_solve)
		end
		# Eigenvalue tolerance check
		if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
			if nb_it_eigval_conv == 5
				print("Basic algo converged off eigval tolerance \n")
				return (real(theta),basis,hesse,outVec,ritzVec,resVec,eigenvalues,
					eigenvectors,nb_it_vals_basic_for_restart,nb_it_total_bicgstab_solve)
			end 
			nb_it_eigval_conv += 1
		end
		previous_eigval = theta
		nb_it_vals_basic_for_restart += 1
	end
	print("Didn't converge off tolerance for basic program. 
		Atteined max set number of iterations \n")
	return (real(theta),basis,hesse,outVec,ritzVec,resVec,eigenvalues,
		eigenvectors,nb_it_vals_basic_for_restart,nb_it_total_bicgstab_solve) # ,ritzVec
end

function jacDavRitz_restart(opt,innerLoopDim::Integer,
		restartDim::Integer,tol_MGS::Float64,tol_conv::Float64,
			tol_eigval::Float64,tol_bicgstab::Float64)::Tuple{Float64,Float64,
				Float64} 

	dims = size(opt) # opt is a square matrix, so dims[1]=dims[2]
	vecDim = dims[1]
	basis = Array{ComplexF64}(undef, vecDim, vecDim)
	hesse = zeros(ComplexF64, vecDim, vecDim)

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	# eigenvectors = Array{ComplexF64}(undef, innerLoopDim,restartDim)
	eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
	
	restart_outVec = Vector{ComplexF64}(undef, vecDim)
	restart_resVec = Vector{ComplexF64}(undef, vecDim)
	restart_ritzVec = Vector{ComplexF64}(undef, vecDim)
	restart_basis = Array{ComplexF64}(undef, vecDim, restartDim)
	restart_hesse = Array{ComplexF64}(undef, vecDim, restartDim)
	restart_theta = 0 
	# Change of basis matrix
	u_matrix = Array{ComplexF64}(undef, innerLoopDim, restartDim)

	# Memory initialization
	outVec = Vector{ComplexF64}(undef, vecDim)
	resVec = Vector{ComplexF64}(undef, vecDim)
	ritzVec = Vector{ComplexF64}(undef, vecDim)
	# Set starting vector
	rand!(view(basis, :, 1))
	# Normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(basis,:,1), 1)
	basis[:, 1] = basis[:, 1] ./ nrm
	# Algorithm initialization
	outVec = opt * basis[:, 1] 
	# Hessenberg matrix
	hesse[1,1] = BLAS.dotc(vecDim, view(basis, :, 1), 1, outVec, 1) 
	# Ritz value
	theta = hesse[1,1] 
	# Ritz vector
	ritzVec[:] = basis[:, 1]
	# Negative residual vector
	resVec = (theta .* ritzVec) .- outVec

	# Initialize some parameters
	previous_eigval = theta
	nb_it_restart = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0

	for it in 1 : 1000
		eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
		# Inner loop
		if it == 1
			theta,basis,hesse,outVec,ritzVec,resVec,eigenvalues,
				eigenvectors,nb_it_vals_basic_for_restart,
					nb_it_bicgstab_solve = jacDavRitz_basic_for_restart(opt, 
			innerLoopDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
			nb_it_restart += nb_it_vals_basic_for_restart
			nb_it_total_bicgstab_solve += nb_it_bicgstab_solve
		else # Essentially for it > 1
			outVec = Vector{ComplexF64}(undef, vecDim)
			outVec = restart_outVec
			resVec = Vector{ComplexF64}(undef, vecDim)
			resVec = restart_resVec
			ritzVec = Vector{ComplexF64}(undef, vecDim)
			ritzVec = restart_ritzVec

			eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
			eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)

			basis = Array{ComplexF64}(undef, vecDim, vecDim)
			basis[:,1:restartDim] = restart_basis

			hesse = Array{ComplexF64}(undef, vecDim, vecDim)
			hesse[1:restartDim,1:restartDim] = restart_hesse

			theta = restart_theta

			for itr in restartDim+1:innerLoopDim
				# # Test code 
				# nb_it = 0 # Just for a rest
				# A_product = (I-ritzVec*adjoint(ritzVec))*(opt-theta*I)*(I-ritzVec*adjoint(ritzVec))
				# # direct_solve =  A_product\resVec
				# basis[:, itr] = A_product\resVec # Just for a test
				# # print("direct_solve ", direct_solve,"\n")

				# Good code
				# Jacobi-Davidson direction
				basis[:, itr],nb_it = ritz_cg_matrix(opt, theta, ritzVec, resVec, 
					tol_bicgstab)
				# basis[:, itr],nb_it = ritz_bicgstab_matrix(opt, theta, ritzVec, resVec, 
				# 	tol_bicgstab)

				# A_product = (I-ritzVec*adjoint(ritzVec))*(opt-theta*I)*(I-ritzVec*adjoint(ritzVec))
				# print("direct solve ", A_product\resVec,"\n")

				nb_it_total_bicgstab_solve += nb_it

				# Orthogonalize
				gramSchmidt!(basis, itr, tol_MGS)
				# New image
				outVec = opt * basis[:, itr] 
				# Update Hessenberg
				hesse[1 : itr, itr] = BLAS.gemv('C', view(basis, :, 1 : itr), outVec)
				hesse[itr, 1 : (itr - 1)] = conj(hesse[1 : (itr - 1), itr])
				# Eigenvalue decomposition, largest real eigenvalue last. 
				# should replace by BLAS operation
				eigSys = eigen(view(hesse, 1 : itr, 1 : itr)) 

				eigenvectors[1:itr,1:itr] = eigSys.vectors
				eigenvalues[1:itr] = eigSys.values

				# Update Ritz vector
				if abs.(eigSys.values[end]) > abs.(eigSys.values[1])
					theta = eigSys.values[end]
					ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, end])
				else # For the case abs.(eigSys.values[end]) < abs.(eigSys.values[1])
					theta = eigSys.values[1]
					ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, 1])	
				end 
				
				outVec = opt * ritzVec
				# Update residual vector
				resVec = (theta * ritzVec) .- outVec

				# Direction vector tolerance check 
				if norm(resVec) < tol_conv
					print("Basic algo converged off resVec tolerance \n")
					return real(theta),nb_it_restart,nb_it_total_bicgstab_solve
				end
				# Eigenvalue tolerance check
				if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
					if nb_it_eigval_conv == 5
						print("Basic algo converged off eigval tolerance \n")
						return real(theta),nb_it_restart,nb_it_total_bicgstab_solve
						# ,ritzVec
					end 
					nb_it_eigval_conv += 1
				end
				previous_eigval = theta
				nb_it_restart += 1
			end
		end

		"""
		Time to save important stuff before restarting
		Once we have ran out of memory, we want to restart the inner loop 
		but not with random starting vectors and matrices but with the ones
		from the last inner loop iteration that was done before running out 
		of memory. 
		"""

		"""
		Create change of basis matrix: u_matrix
		# Create trgBasis and srcBasis for restart
		"""
		for i = 1:restartDim
			# Creation of new trgBasis and srcBasis matrices 
			if abs.(eigenvalues[end]) > abs.(eigenvalues[1]) 
				u_matrix[1:innerLoopDim,i] = eigenvectors[:, end]
				restart_basis[:,i] = basis[:, 1 : innerLoopDim] * (eigenvectors[:, end])
				pop!(eigenvalues) # This removes the last element
				eigenvectors = eigenvectors[:, 1:end-1] # This removes the last column
			else # abs.(eigSys.values[end]) < abs.(eigSys.values[1])
				u_matrix[1:innerLoopDim,i] = eigenvectors[:, 1]
				restart_basis[:,i] = basis[:, 1 : innerLoopDim] * (eigenvectors[:, 1])
				popfirst!(eigenvalues) # This removes the first element
				eigenvectors = eigenvectors[:, 2:end] # This removes the first column 
			end
		end
		# Orthogonalize
		gramSchmidt!(basis, innerLoopDim, tol_MGS)
		"""
		Create Hessenberg matrix for restart using change of basis matrix: u_matrix
		"""
		restart_hesse = adjoint(u_matrix)*hesse[1 : innerLoopDim, 1 : innerLoopDim]*u_matrix
		# Make the restart Hessenberg matrix hermitian
		restart_hesse = (restart_hesse.+adjoint(restart_hesse))./2
		"""
		Save other stuff for restart
		"""
		restart_outVec = outVec
		restart_ritzVec = ritzVec
		restart_resVec = resVec
	end
	print("Didn't converge off tolerance for basic program. 
		Atteined max set number of iterations \n")
	return (real(theta),nb_it_restart,nb_it_total_bicgstab_solve) # ritzVec,
end

# Test code 
# sz = 256
# innerLoopDim = 100
# restartDim = 10
# tol_MGS = 1e-9
# tol_conv = 1e-6
# tol_eigval = 1e-6
# tol_bicgstab = 1e-6
# opt = Array{ComplexF64}(undef,sz,sz)
# rand!(opt)
# opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# trueEigSys = eigen(opt)
# minEigPos = argmin(abs.(trueEigSys.values))
# minEig = trueEigSys.values[minEigPos]
# # println("The Julia smallest eigenvalue is ", minEig,".")
# dims = size(opt)
# # print("dims[1] ", dims[1], "\n")
# # print("dims[2] ", dims[2], "\n")
# bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# kMat = zeros(ComplexF64, dims[2], dims[2])
# # val = jacDavRitzHarm(trgBasis, srcBasis, kMat, opt, dims[1], dims[2], 1.0e-6)
# val,nb_it_restart,nb_it_total_bicgstab_solve = jacDavRitz_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
# println("The Julia eigenvalues are ", trueEigSys.values,".")
# print("Ritz eigenvalue closest to 0 is ", val, "\n")

end