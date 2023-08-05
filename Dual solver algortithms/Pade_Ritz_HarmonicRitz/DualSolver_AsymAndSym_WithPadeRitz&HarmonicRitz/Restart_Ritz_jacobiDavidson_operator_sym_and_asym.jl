"""
The Restart_Ritz_jacobiDavidson module allows to calculate the extremal 
eigenvalue, which can be positive or negative, of a given matrix.

Author: Sean Molesky & Nicolas Leblanc
"""

module Restart_Ritz_jacobiDavidson_operator_sym_and_asym
export ritz_bicgstab_operator,jacDavRitz_basic,jacDavRitz_basic_for_restart,
	jacDavRitz_restart
using LinearAlgebra, Random, product_sym_and_asym

@inline function projVec(dim::Integer, pVec::Vector{T}, sVec::Array{T})::Array{T} where T <: Number

	return sVec .- (BLAS.dotc(dim, pVec, 1, sVec, 1) .* pVec)
end

# This is a biconjugate gradient program without a preconditioner. 
# m is the maximum number of iterations
function ritz_bicgstab_operator(xi,l,gMemSlfN,gMemSlfA,cellsA,
    chi_inv_coeff,P, theta, u, b, tol_bicgstab)

	dim = cellsA[1]*cellsA[2]*cellsA[3]*3
    vk = pk = xk = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    r0_hat = rk = b 
    rho_m1 = alpha = omega_k = 1
	k = 0
    for k in 1 : 1000
        rho_k = dot(r0_hat,rk) 
        # Bêta calculation
        # First term 
        first_term = rho_k/rho_m1
        # Second term 
        second_term = alpha/omega_k
        # Calculation 
        beta = first_term*second_term
        pk = rk .+ beta.*(pk-omega_k.*vk)
        pkPrj = projVec(dim, u, pk)
		# Need to compare the output of these two ways of doing this product
        # Direct sum 
        A_pkPrj = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, pkPrj)+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, pkPrj)
        # Function to do the sum
        # A_pkPrj = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, pkPrj)
	
		vk = projVec(dim, u, A_pkPrj .- (theta .* pkPrj))        
		
		# alpha calculation
        # Bottom term
        bottom_term = dot(r0_hat,vk) # conj.(transpose(r0))*vk
        # Calculation 
        alpha = rho_k / bottom_term 
        h = xk + alpha.*pk
        s = rk - alpha.*vk
        sPrj = projVec(dim, u, s) 

		# Need to compare the output of these two ways of doing this product
        # Direct sum 
        A_sPrj = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, sPrj)+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, sPrj)
        # Function to do the sum
        # A_sPrj = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, skPrj)
		
		t = projVec(dim, u, A_sPrj .- (theta .* sPrj))
        
		# omega_k calculation 
        # Top term 
        ts = dot(t,s) # conj.(transpose(t))*s
        # Bottom term
        tt = dot(t,t) # conj.(transpose(t))*t
        # Calculation 
        omega_k = ts ./ tt
        xk = h + omega_k.*s
        rk = s-omega_k.*t
		rho_m1 = rho_k
        if norm(rk) < tol_bicgstab
			return xk,k # k is essentially the number of iterations 
			# to reach the chosen tolerance
        end
    end
	print("Didn't converge off tolerence \n")
    return xk,k # k is essentially the number of iterations 
end

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
		# renormalize orthogonalized vector
		basis[:,n] = basis[:,n] ./ nrm
	end 
end

function jacDavRitz_basic(xi,l,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64,
	tol_eigval::Float64,tol_bicgstab::Float64)::Float64 # Tuple{Float64,Float64,Float64}

	dim = cellsA[1]*cellsA[2]*cellsA[3]*3

	# basis = Array{ComplexF64}(undef, dim, dim)
	# hesse = zeros(ComplexF64, dim, dim)
	basis = zeros(ComplexF64, vecDim, vecDim)
	hesse = zeros(ComplexF64, vecDim, vecDim)

	# Memory initialization
	outVec = Vector{ComplexF64}(undef, dim)
	resVec = Vector{ComplexF64}(undef, dim)
	ritzVec = Vector{ComplexF64}(undef, dim)
	# Set starting vector
	rand!(view(basis, :, 1))
	# Normalize starting vector
	nrm = BLAS.nrm2(dim, view(basis,:,1), 1)
	basis[:, 1] = basis[:, 1] ./ nrm
	# Algorithm initialization
	# Need to compare the output of these two ways of doing this product
	# Direct sum 
	outVec = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,1])+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, basis[:,1])
	# Function to do the sum
	# outVec = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,1])

	# Hessenberg matrix
	hesse[1,1] = BLAS.dotc(dim, view(basis, :, 1), 1, outVec, 1) 
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
		basis[:, itr],nb_it = ritz_bicgstab_operator(xi,l,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P, theta, ritzVec, resVec, 
			tol_bicgstab)
		nb_it_total_bicgstab_solve += nb_it

		# Orthogonalize
		gramSchmidt!(basis, itr, tol_MGS)
		
		# New image
		# Need to compare the output of these two ways of doing this product
		# Direct sum 
		outVec = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,itr])+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, basis[:,itr])
		# Function to do the sum
		# outVec = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,itr])

		# Update Hessenberg
		hesse[1 : itr, itr] = BLAS.gemv('C', view(basis, :, 1 : itr), outVec)
		hesse[itr, 1 : (itr - 1)] = conj(hesse[1 : (itr - 1), itr])
		# Eigenvalue decomposition, largest real eigenvalue last. 
		# should replace by BLAS operation
		eigSys = eigen(view(hesse, 1 : itr, 1 : itr)) 
		# Update Ritz vector
		theta = eigSys.values[end]
		ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, end])
		# Need to compare the output of these two ways of doing this product
		# Direct sum 
		outVec = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, ritzVec)+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, ritzVec)
		# Function to do the sum
		# outVec = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, ritzVec)

		# Update residual vector
		resVec = (theta * ritzVec) .- outVec

		# Direction vector tolerance check 
		if norm(resVec) < tol_conv
			print("Basic algo converged off resVec tolerance \n")
			return real(theta) # ,nb_it_vals_basic,nb_it_total_bicgstab_solve #,ritzVec
		end
		# Eigenvalue tolerance check
		if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
			if nb_it_eigval_conv == 5
				print("Basic algo converged off eigval tolerance \n")
				return real(theta) # ,nb_it_vals_basic,nb_it_total_bicgstab_solve
			end 
			nb_it_eigval_conv += 1
		end
		previous_eigval = theta
		nb_it_vals_basic += 1
	end
	print("Didn't converge off tolerance for basic program. 
	Atteined max set number of iterations \n")
	return real(theta) # ,nb_it_vals_basic,nb_it_total_bicgstab_solve) # ,ritzVec
end

function jacDavRitz_basic_for_restart(xi,l,gMemSlfN,gMemSlfA,
	cellsA,chi_inv_coeff,P,innerLoopDim::Integer,tol_MGS::Float64,
		tol_conv::Float64,tol_eigval::Float64,tol_bicgstab::Float64)		

	dim = cellsA[1]*cellsA[2]*cellsA[3]*3

	# basis = Array{ComplexF64}(undef, dim, dim)
	basis = zeros(ComplexF64, dim, dim)
	hesse = zeros(ComplexF64, dim, dim)

	# Memory initialization
	outVec = Vector{ComplexF64}(undef, dim)
	resVec = Vector{ComplexF64}(undef, dim)
	ritzVec = Vector{ComplexF64}(undef, dim)
	# Set starting vector
	rand!(view(basis, :, 1))
	# Normalize starting vector
	nrm = BLAS.nrm2(dim, view(basis,:,1), 1)
	basis[:, 1] = basis[:, 1] ./ nrm
	# Algorithm initialization
	# Need to compare the output of these two ways of doing this product
	# Direct sum 
	outVec = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,1])+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, basis[:,1])
	# Function to do the sum
	# outVec = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,1])

	# Hessenberg matrix
	hesse[1,1] = BLAS.dotc(dim, view(basis, :, 1), 1, outVec, 1) 
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

		# Jacobi-Davidson direction 
		basis[:, itr],nb_it = ritz_bicgstab_operator(xi,l,
			gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P, theta, ritzVec, resVec, 
				tol_bicgstab)
		nb_it_total_bicgstab_solve += nb_it

		# Orthogonalize
		gramSchmidt!(basis, itr, tol_MGS)
		# New image
		# Need to compare the output of these two ways of doing this product
		# Direct sum 
		outVec = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,itr])+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, basis[:,itr])
		# Function to do the sum
		# outVec = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,itr])

		# Update Hessenberg
		hesse[1 : itr, itr] = adjoint(view(basis, :, 1 : itr))*outVec
		hesse[itr, 1 : (itr - 1)] = conj(hesse[1 : (itr - 1), itr])

		# Eigenvalue decomposition, largest real eigenvalue last. 
		# should replace by BLAS operation
		eigSys = eigen(view(hesse, 1 : itr, 1 : itr)) 

		eigenvectors[1:itr,1:itr] = eigSys.vectors 
		eigenvalues[1:itr] = eigSys.values

		# Update Ritz vector
		theta = eigSys.values[end]
		ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, end])
		# Need to compare the output of these two ways of doing this product
		# Direct sum 
		outVec = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, ritzVec)+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, ritzVec)
		# Function to do the sum
		# outVec = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, ritzVec)

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

function jacDavRitz_restart(xi,l,gMemSlfN,gMemSlfA,cellsA,
	chi_inv_coeff,P,innerLoopDim::Integer,restartDim::Integer,tol_MGS::Float64,
		tol_conv::Float64, tol_eigval::Float64,
			tol_bicgstab::Float64)::Float64 # Tuple{Float64,Float64,Float64} 

	dim = cellsA[1]*cellsA[2]*cellsA[3]*3

	basis = Array{ComplexF64}(undef, dim, dim)
	hesse = zeros(ComplexF64, dim, dim)

	eigenvectors = zeros(ComplexF64, innerLoopDim, innerLoopDim)
	eigenvalues = zeros(ComplexF64, innerLoopDim)

	restart_outVec = zeros(ComplexF64, dim)
	restart_resVec = zeros(ComplexF64, dim)
	restart_ritzVec = zeros(ComplexF64, dim)
	restart_basis = zeros(ComplexF64, dim, restartDim)
	restart_hesse = zeros(ComplexF64, dim, restartDim)
	restart_theta = 0 
	# Change of basis matrix
	u_matrix = zeros(ComplexF64, innerLoopDim, restartDim)

	# Memory initialization
	outVec = Vector{ComplexF64}(undef, dim)
	resVec = Vector{ComplexF64}(undef, dim)
	ritzVec = Vector{ComplexF64}(undef, dim)
	# Set starting vector
	rand!(view(basis, :, 1))
	# Normalize starting vector
	nrm = BLAS.nrm2(dim, view(basis,:,1), 1)
	basis[:, 1] = basis[:, 1] ./ nrm
	# Algorithm initialization
	# Need to compare the output of these two ways of doing this product
	# Direct sum 
	outVec = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,1])+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, basis[:,1])
	# Function to do the sum
	# outVec = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,1])

	# Hessenberg matrix
	hesse[1,1] = BLAS.dotc(dim, view(basis, :, 1), 1, outVec, 1) 
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
					nb_it_bicgstab_solve = jacDavRitz_basic_for_restart(xi,l,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P, 
							innerLoopDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
			nb_it_restart += nb_it_vals_basic_for_restart
			nb_it_total_bicgstab_solve += nb_it_bicgstab_solve
		else # Essentially for it > 1

			outVec = Vector{ComplexF64}(undef, dim)
			outVec = restart_outVec
			resVec = Vector{ComplexF64}(undef, dim)
			resVec = restart_resVec
			ritzVec = Vector{ComplexF64}(undef, dim)
			ritzVec = restart_ritzVec

			eigenvectors = zeros(ComplexF64, innerLoopDim, innerLoopDim)
			eigenvalues = zeros(ComplexF64, innerLoopDim)

			basis = zeros(ComplexF64, dim,dim)
			basis[:,1:restartDim] = restart_basis
			hesse = zeros(ComplexF64, dim, dim)
			hesse[1:restartDim,1:restartDim] = restart_hesse

			theta = restart_theta

			for itr in restartDim+1:innerLoopDim
				# Jacobi-Davidson direction
				basis[:, itr],nb_it = ritz_bicgstab_operator(xi,
					l,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P, theta,ritzVec, 
						resVec, tol_bicgstab)
				nb_it_total_bicgstab_solve += nb_it

				# Orthogonalize
				gramSchmidt!(basis, itr, tol_MGS)
				# New image
				# Need to compare the output of these two ways of doing this product
				# Direct sum 
				outVec = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,itr])+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, basis[:,itr])
				# Function to do the sum
				# outVec = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, basis[:,itr])

				# Update Hessenberg
				hesse[1 : itr, itr] = adjoint(view(basis, :, 1 : itr))*outVec
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

				# Need to compare the output of these two ways of doing this product
				# Direct sum 
				outVec = xi*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, ritzVec)+l[1]*sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff, P, ritzVec)
				# Function to do the sum
				# outVec = sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, ritzVec)

				# Update residual vector
				resVec = (theta * ritzVec) .- outVec

				# Direction vector tolerance check 
				if norm(resVec) < tol_conv
					print("Basic algo converged off resVec tolerance \n")
					return real(theta) # ,nb_it_restart,nb_it_total_bicgstab_solve
				end

				# Eigenvalue tolerance check
				if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
					if nb_it_eigval_conv == 5
						print("Basic algo converged off eigval tolerance \n")
						return real(theta) # ,nb_it_restart,nb_it_total_bicgstab_solve
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
	return (real(theta)) 
end

end

"""
Instead of bicgstab, conjugate gradient could have been used. The code below
shows the code that could have been used. 
"""
# function ritz_cg_matrix(A,theta,u,b,cg_tol)
#     # tol = 1e-5 # The program terminates once 
#     # there is an r for which its norm is smaller
#     # than the chosen tolerance. 
# 	dim = size(A)[1]
#     xk = zeros(ComplexF64,length(b),1)
#     # Ax=0 since the initial xk is 0
#     pk = rk = b 
#     # k = 0
#     # for k in 1:length(b)
# 	nb_it = 0
#     for k in 1:1000
#         # alpha_k coefficient calculation 
#         # Top term
#         rkrk = dot(rk,rk) # conj.(transpose(rk))*rk
#         print("rkrk ", rkrk, "\n")
# 		# Bottom term 
#         pkPrj = projVec(dim, u, pk)
#         A_pkPrj = projVec(dim, u, A * pkPrj .- (theta .* pkPrj))
#         # A_pk = A*pk
#         pk_A_pkPrj = dot(pk,A_pkPrj) # conj.(transpose(pk))*A_pk
#         # Division
#         alpha_k = rkrk/pk_A_pkPrj

#         # x_{k+1} calculation 
#         xk = xk + alpha_k.*pk

#         # r_{k+1} calculation 
#         rk = rk - alpha_k.*A_pkPrj

#         # print("norm(rk_plus1) ",norm(rk), "\n")
#         if norm(rk)  < cg_tol
#             return xk,nb_it
#         end

#         # beta_k coefficient calculation 
#         # Top term 
#         rkplus1_rkplus1 = dot(rk,rk) # conj.(transpose(rk))*rk
#         # The bottom term is the same one calculated earlier 
#         # Division 
#         # print("rkplus1_rkplus1 ", rkplus1_rkplus1, "\n")
#         # print("rkrk ", rkrk, "\n")

#         beta_k = rkplus1_rkplus1/rkrk

#         pk = rk + beta_k.*pk
# 		nb_it += 1
#     end
# 	print("Didn't converge off bicgstab tolerance \n")
#     return xk,nb_it
# end 