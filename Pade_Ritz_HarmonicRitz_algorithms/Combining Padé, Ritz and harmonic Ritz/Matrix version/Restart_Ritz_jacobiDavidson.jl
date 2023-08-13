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
function ritz_bicgstab_matrix(A, theta, u, b, tol_bicgstab, max_nb_it)
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
        pkPrj = projVec(dim, u, pk)
        vk = projVec(dim, u, A * pkPrj .- (theta .* pkPrj))
        # Alpha calculation
        # Bottom term
        bottom_term = dot(r0_hat,vk) # conj.(transpose(r0))*vk
        # Calculation 
        alpha = rho_k / bottom_term 
        h = xk + alpha.*pk
        s = rk - alpha.*vk
        sPrj = projVec(dim, u, s) 
        t = projVec(dim, u, A * sPrj .- (theta .* sPrj))
        # Omega_k calculation 
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
		# Renormalize orthogonalized vector
		basis[:,n] = basis[:,n] ./ nrm
	end 
end

function jacDavRitz_basic(opt,
		innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64,
			tol_eigval::Float64,tol_bicgstab::Float64,max_nb_it::Integer,
				basis_solver::String)::Tuple{Float64,Float64,
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
		# basis[:, itr],nb_it = ritz_cg_matrix(opt, theta, ritzVec, resVec, 
		# tol_bicgstab)	
		if basis_solver == "bicgstab"	
			basis[:, itr],nb_it = ritz_bicgstab_matrix(opt, theta, ritzVec, resVec, 
				tol_bicgstab, max_nb_it)
		end 

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
	return (real(theta),nb_it_vals_basic,nb_it_total_bicgstab_solve) 
end

function jacDavRitz_basic_for_restart(opt, 
			innerLoopDim::Integer,tol_MGS::Float64,
				tol_conv::Float64,tol_eigval::Float64,
					tol_bicgstab::Float64, max_nb_it::Integer,
						basis_solver::String)		
	# Memory initialization
	dims = size(opt) # opt is a square matrix, so dims[1]=dims[2]
	vecDim = dims[1]
	basis = zeros(ComplexF64, vecDim, vecDim)
	# basis = Array{ComplexF64}(undef, vecDim, vecDim)
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

	# eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	# eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
	eigenvectors = zeros(ComplexF64, innerLoopDim, innerLoopDim)
	eigenvalues = zeros(ComplexF64, innerLoopDim)


	for itr in 2 : innerLoopDim
		# Jacobi-Davidson direction
		# basis[:, itr],nb_it = ritz_cg_matrix(opt, theta, ritzVec, resVec, 
		# 	tol_bicgstab)
		if basis_solver == "bicgstab"
			basis[:, itr],nb_it = ritz_bicgstab_matrix(opt, theta, ritzVec, resVec, 
				tol_bicgstab, max_nb_it)
		end
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
			print("Converged in ", itr, "iterations \n")
			return (real(theta),basis,hesse,outVec,ritzVec,resVec,eigenvalues,
				eigenvectors,nb_it_vals_basic_for_restart,nb_it_total_bicgstab_solve)
		end
		# Eigenvalue tolerance check
		if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
			if nb_it_eigval_conv == 5
				print("Converged in ", itr, "iterations \n")
				print("Basic for restart algo converged off eigval tolerance \n")
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
			tol_eigval::Float64,tol_bicgstab::Float64,max_nb_it::Integer,
				basis_solver::String)::Float64 # ::Tuple{Float64,Float64,Float64} 

	dims = size(opt) # opt is a square matrix, so dims[1]=dims[2]
	vecDim = dims[1]
	basis = zeros(ComplexF64, vecDim, vecDim)
	hesse = zeros(ComplexF64, vecDim, vecDim)

	eigenvectors = zeros(ComplexF64, innerLoopDim, innerLoopDim)
	eigenvalues = zeros(ComplexF64, innerLoopDim)
	
	restart_outVec = zeros(ComplexF64, vecDim)
	restart_resVec = zeros(ComplexF64, vecDim)
	restart_ritzVec = zeros(ComplexF64, vecDim)
	restart_basis = zeros(ComplexF64, vecDim, restartDim)
	restart_hesse = zeros(ComplexF64, vecDim, restartDim)
	restart_theta = 0 
	# Change of basis matrix
	u_matrix = zeros(ComplexF64, innerLoopDim, restartDim)

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

	for it in 1 : max_nb_it
		eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
		# Inner loop
		if it == 1
			theta,basis,hesse,outVec,ritzVec,resVec,eigenvalues,
				eigenvectors,nb_it_vals_basic_for_restart,
					nb_it_bicgstab_solve = jacDavRitz_basic_for_restart(opt, 
						innerLoopDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab,max_nb_it,
							basis_solver)
			nb_it_restart += nb_it_vals_basic_for_restart
			nb_it_total_bicgstab_solve += nb_it_bicgstab_solve
		else # Essentially for it > 1
			outVec = Vector{ComplexF64}(undef, vecDim)
			outVec = restart_outVec
			resVec = Vector{ComplexF64}(undef, vecDim)
			resVec = restart_resVec
			ritzVec = Vector{ComplexF64}(undef, vecDim)
			ritzVec = restart_ritzVec

			eigenvectors = zeros(ComplexF64, innerLoopDim, innerLoopDim)
			eigenvalues = zeros(ComplexF64, innerLoopDim)

			basis = zeros(ComplexF64, vecDim, vecDim)
			basis[:,1:restartDim] = restart_basis

			hesse = zeros(ComplexF64, vecDim, vecDim)
			hesse[1:restartDim,1:restartDim] = restart_hesse

			theta = restart_theta

			for itr in restartDim+1:innerLoopDim
				# Jacobi-Davidson direction
				# basis[:, itr],nb_it = ritz_cg_matrix(opt, theta, ritzVec, resVec, 
				# 	tol_bicgstab)
				if basis_solver == "bicgstab"
					basis[:, itr],nb_it = ritz_bicgstab_matrix(opt, theta, ritzVec, resVec, 
						tol_bicgstab, max_nb_it)
				end

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
					print("Converged in ", itr, "iterations \n")
					return real(theta),nb_it_restart,nb_it_total_bicgstab_solve
				end
				# Eigenvalue tolerance check
				if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
					if nb_it_eigval_conv == 5
						print("Restart algo converged off eigval tolerance \n")
						print("Converged in ", itr, "iterations \n")
						return real(theta),nb_it_restart,nb_it_total_bicgstab_solve
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
	return real(theta),nb_it_restart,nb_it_total_bicgstab_solve
end

# Old test code
# sz = 100
# tol_MGS = 1e-9
# tol_conv = 1e-6
# tol_bicgstab = 1e-4
# tol_cg = 1e-4
# tol_eigval = 1e-6
# max_nb_it = 1000
# A = Array{ComplexF64}(undef,sz,sz)
# rand!(A)
# A[:,:] .= (A .+ adjoint(A)) ./ 2
# trueEigSys = eigen(A)
# minEigPos = argmin(abs.(trueEigSys.values))
# minEig = trueEigSys.values[minEigPos]
# dims = size(A)
# # print("dims[1] ", dims[1], "\n")
# # print("dims[2] ", dims[2], "\n")
# u = Vector{ComplexF64}(undef, dims[2])
# b = Vector{ComplexF64}(undef, dims[2])
# rand!(u)
# rand!(b)
# theta = 2

# # val,nb_it = ritz_bicgstab_matrix(A, theta, u, b, tol_bicgstab)
# innerLoopDim = 50
# restartDim = 5
# eigval,nb_it1,nb_it2 = jacDavRitz_restart(A,innerLoopDim,restartDim,tol_MGS,
# 	tol_conv,tol_eigval,tol_bicgstab)
# # print("bicgstab solution ", val, "\n")
# # print("direct solve ", ((I-u*adjoint(u))*(A-theta*I)*(I-u*adjoint(u)))\b, "\n")

# println("The different Julia eigenvalues are ", trueEigSys.values,".")
# print("Harmonic Ritz eigenvalue closest to 0 is ", eigval, "\n")
end