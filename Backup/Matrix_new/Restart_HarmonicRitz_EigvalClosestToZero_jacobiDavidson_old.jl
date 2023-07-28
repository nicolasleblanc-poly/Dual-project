"""
The Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson module allows to 
calculate the eigenvalue closest to 0, which can be positive or negative, of a 
given matrix.

Author: Sean Molesky & Nicolas Leblanc
"""

module Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson
export jacDavRitzHarm_basic,jacDavRitzHarm_basic_for_restart,jacDavRitzHarm_restart
using LinearAlgebra, Random, Base.Threads, Plots

function jacDavRitzHarm_basic(opt,innerLoopDim::Integer,
	tol_MGS::Float64,tol_conv::Float64,tol_eigval::Float64,
		tol_bicgstab::Float64)::Tuple{Float64,Float64,Float64}

	dims = size(opt)
	vecDim = dims[1]
	bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
	bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
	trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
	srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
	kMat = zeros(ComplexF64, dims[2], dims[2])

	resVec = Vector{ComplexF64}(undef, vecDim)
	hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	hRitzSrc = Vector{ComplexF64}(undef, vecDim)
	bCoeffs1 = Vector{ComplexF64}(undef, vecDim)
	bCoeffs2 = Vector{ComplexF64}(undef, vecDim)
	# Set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# Normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	### Algorithm initialization
	trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
	nrm = BLAS.nrm2(vecDim, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# Representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(vecDim, view(trgBasis, :, 1), 1,
		view(srcBasis, :, 1), 1) # Kk
	# Ritz value
	eigPos = 1
	theta = 1 / kMat[1,1] # eigenvalue 
	# Ritz vectors
	hRitzTrg[:] = trgBasis[:, 1] # hk = wk 
	hRitzSrc[:] = srcBasis[:, 1] # fk = vk

	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

	previous_eigval = theta
	nb_it_vals_basic = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0
	
	# Code for if we just want the inner loop, so with no restart  
	for itr in 2 : innerLoopDim  # Need to determine when this for loops stops 
		# Depending on how much memory the laptop can take before crashing.
		prjCoeff = BLAS.dotc(vecDim, hRitzTrg, 1, hRitzSrc, 1)
		# Calculate Jacobi-Davidson direction

		# Test code 
		# nb_it = 0 # Just for a rest
		# A_product = (I-ritzVec*adjoint(ritzVec))*(opt-theta*I)*(I-ritzVec*adjoint(ritzVec))
		# # direct_solve =  A_product\resVec
		# srcBasis[:, itr] = A_product\resVec # Just for a test
		# # print("direct_solve ", direct_solve,"\n")
		# # Jacobi-Davidson direction
		# # basis[:, itr],nb_it = ritz_bicgstab_matrix(opt, theta, ritzVec, resVec, 
		# # tol_bicgstab)
		# nb_it_total_bicgstab_solve += nb_it

		# Good code
		srcBasis[:, itr],nb_it = harm_ritz_cg_matrix(opt, theta, hRitzTrg,
			hRitzSrc, prjCoeff, resVec, tol_bicgstab)
		# srcBasis[:, itr],nb_it = harm_ritz_bicgstab_matrix(opt, theta, hRitzTrg,
		# 		hRitzSrc, prjCoeff, resVec, tol_bicgstab)
		nb_it_total_bicgstab_solve += nb_it

		trgBasis[:, itr] = opt * srcBasis[:, itr]

		# Orthogonalize
		gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, opt,
			itr, tol_MGS)

		# Update inverse representation of opt^{-1} in trgBasis
		kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
			view(srcBasis, :, itr))
		# Assuming opt^{-1} Hermitian matrix
		kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])

		# kMat is not Hermitian so let's make it Hermitian by adding its 
		# adjoint and dividing by 2
		kMat[1 : itr, 1 : itr] = (kMat[1 : itr, 1 : itr].+adjoint(kMat[1 : itr, 1 : itr]))./2

		# Eigenvalue decomposition, largest real eigenvalue last.
		# Should replace by BLAS operation
		eigSys = eigen(view(kMat, 1 : itr, 1 : itr))
	
		# Update Ritz vector
		if abs.(eigSys.values[end]) > abs.(eigSys.values[1])
			theta = 1/eigSys.values[end]
			hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, end])
			hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, end])
		else
			theta = 1/eigSys.values[1]
			hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
			hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
		end	

		# Update residual vector
		resVec = (theta * hRitzSrc) .- hRitzTrg
 
		# Direction vector tolerance check 
		if norm(resVec) < tol_conv
			print("Basic algo converged off resVec tolerance \n")
			# print("norm(resVec) ", norm(resVec), "\n")
			return real(theta),nb_it_vals_basic,nb_it_total_bicgstab_solve
			# println(real(theta))
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
		# print("norm(resVec) basic program ", norm(resVec),"\n")
	end
 
	print("Didn't converge off tolerance for basic program. 
		Atteined max set number of iterations \n")
	return real(theta),nb_it_vals_basic,nb_it_total_bicgstab_solve
end


function jacDavRitzHarm_basic_for_restart(opt, 
	innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64,
		tol_eigval::Float64,tol_bicgstab::Float64)
# function jacDavRitzHarm_basic(trgBasis::Array{ComplexF64}, 
# 	srcBasis::Array{ComplexF64}, kMat::Array{ComplexF64}, opt::Array{ComplexF64}, 
# 	vecDim::Integer, repDim::Integer, innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64)::Float64
	### Memory initialization

	dims = size(opt)
	vecDim = dims[1]
	bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
	bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
	trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
	srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
	kMat = zeros(ComplexF64, dims[2], dims[2])

	resVec = Vector{ComplexF64}(undef, vecDim)
	hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	hRitzSrc = Vector{ComplexF64}(undef, vecDim)
	bCoeffs1 = Vector{ComplexF64}(undef, vecDim)
	bCoeffs2 = Vector{ComplexF64}(undef, vecDim)
	# Set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# Normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	### Algorithm initialization
	trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
	nrm = BLAS.nrm2(vecDim, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# Representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(vecDim, view(trgBasis, :, 1), 1,
		view(srcBasis, :, 1), 1) # Kk
	# Ritz value
	eigPos = 1
	theta = 1 / kMat[1,1] # eigenvalue 
	# Ritz vectors
	hRitzTrg[:] = trgBasis[:, 1] # hk = wk 
	hRitzSrc[:] = srcBasis[:, 1] # fk = vk

	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

	previous_eigval = theta
	nb_it_vals_basic_for_restart = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
	
	# Code for if we just want the inner loop, so with no restart  
	for itr in 2 : innerLoopDim  # Need to determine when this for loops stops 
		# Depending on how much memory the laptop can take before crashing.
		prjCoeff = BLAS.dotc(vecDim, hRitzTrg, 1, hRitzSrc, 1)

		# # Test code 
		# nb_it = 0 # Just for a rest
		# A_product = (I-ritzVec*adjoint(ritzVec))*(opt-theta*I)*(I-ritzVec*adjoint(ritzVec))
		# # direct_solve =  A_product\resVec
		# srcBasis[:, itr] = A_product\resVec # Just for a test
		# # print("direct_solve ", direct_solve,"\n")
		# # Jacobi-Davidson direction
		# # basis[:, itr],nb_it = ritz_bicgstab_matrix(opt, theta, ritzVec, resVec, 
		# # tol_bicgstab)
		# nb_it_total_bicgstab_solve += nb_it

		# Good code
		# Calculate Jacobi-Davidson direction
		srcBasis[:, itr],nb_it = harm_ritz_cg_matrix(opt, theta, hRitzTrg,
			hRitzSrc, prjCoeff, resVec, tol_bicgstab)
		# srcBasis[:, itr],nb_it = harm_ritz_bicgstab_matrix(opt, theta, hRitzTrg,
		# 		hRitzSrc, prjCoeff, resVec, tol_bicgstab)
		nb_it_total_bicgstab_solve += nb_it
		
		trgBasis[:, itr] = opt * srcBasis[:, itr]

		# Orthogonalize
		gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, opt,
			itr, tol_MGS)

		# Update inverse representation of opt^{-1} in trgBasis
		kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
			view(srcBasis, :, itr))
		# Assuming opt^{-1} Hermitian matrix
		kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])

		# kMat is not Hermitian so let's make it Hermitian by adding its 
		# adjoint and dividing by 2
		kMat[1 : itr, 1 : itr] = (kMat[1 : itr, 1 : itr].+adjoint(kMat[1 : itr, 1 : itr]))./2

		# Eigenvalue decomposition, largest real eigenvalue last.
		# Should replace by BLAS operation
		eigSys = eigen(view(kMat, 1 : itr, 1 : itr))

		eigenvectors[1:itr,1:itr] = eigSys.vectors
		eigenvalues[1:itr] = eigSys.values
	
		# Update Ritz vector
		if abs.(eigSys.values[end]) > abs.(eigSys.values[1])
			theta = 1/eigSys.values[end]
			hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, end])
			hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, end])
		else
			theta = 1/eigSys.values[1]
			hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
			hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
		end	

		# Update residual vector
		resVec = (theta * hRitzSrc) .- hRitzTrg
 
		# Direction vector tolerance check 
		if norm(resVec) < tol_conv
			print("Basic for restart algo converged off resVec tolerance \n")
			# print("norm(resVec) ", norm(resVec), "\n")
			return real(theta),srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,
				eigenvalues,eigenvectors,nb_it_vals_basic_for_restart,
					nb_it_total_bicgstab_solve
			# println(real(theta))
		end
		# Eigenvalue tolerance check
		if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
			if nb_it_eigval_conv == 5
				print("Basic for restrart algo converged off eigval tolerance \n")
				return real(theta),srcBasis,trgBasis,kMat,resVec,hRitzSrc,
					hRitzTrg,eigenvalues,eigenvectors,nb_it_vals_basic_for_restart,
						nb_it_total_bicgstab_solve
			end 
			nb_it_eigval_conv += 1
		end
		previous_eigval = theta
		nb_it_vals_basic_for_restart += 1
	end
 
	print("Didn't converge off tolerance for basic for restart program. 
		Atteined max set number of iterations \n")
	return real(theta),srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,
		eigenvalues,eigenvectors,nb_it_vals_basic_for_restart,
			nb_it_total_bicgstab_solve
end

function jacDavRitzHarm_restart(opt,innerLoopDim::Integer,
	restartDim::Integer,tol_MGS::Float64,
		tol_conv::Float64,tol_eigval::Float64,tol_bicgstab::Float64)::Tuple{Float64, Float64,Float64}
	
	# Memory initialization
	dims = size(opt) # opt is a square matrix, so dims[1]=dims[2]
	vecDim = dims[1]
	bCoeffs1 = Vector{ComplexF64}(undef, vecDim)
	bCoeffs2 = Vector{ComplexF64}(undef, vecDim)
	trgBasis = Array{ComplexF64}(undef, vecDim, vecDim)
	srcBasis = Array{ComplexF64}(undef, vecDim, vecDim)
	kMat = zeros(ComplexF64, vecDim, vecDim)
	resVec = Vector{ComplexF64}(undef, vecDim)
	hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	hRitzSrc = Vector{ComplexF64}(undef, vecDim)

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)

	restart_resVec = Vector{ComplexF64}(undef, vecDim)
	restart_hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	restart_hRitzSrc = Vector{ComplexF64}(undef, vecDim)
	restart_trgBasis = Array{ComplexF64}(undef, vecDim, restartDim)
	restart_srcBasis = Array{ComplexF64}(undef, vecDim, restartDim)
	restart_kMat = zeros(ComplexF64, restartDim, restartDim)
	restart_theta = 0 

	# Change of basis matrix
	u_matrix = Array{ComplexF64}(undef, innerLoopDim, restartDim)

	# Set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# Normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# Algorithm initialization
	trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
	nrm = BLAS.nrm2(vecDim, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# Representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(vecDim, view(trgBasis, :, 1), 1,
		view(srcBasis, :, 1), 1) # Kk
	# Ritz value
	eigPos = 1
	theta = 1 / kMat[1,1] # eigenvalue 
	# Ritz vectors
	hRitzTrg[:] = trgBasis[:, 1] # hk = wk 
	hRitzSrc[:] = srcBasis[:, 1] # fk = vk
	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

	# Initialize some parameters
	previous_eigval = theta
	nb_it_restart = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0
	
	# Code with restart
	# Outer loop
	for it in 1: 1000 # restartDim # Need to think this over 
		eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
		# Inner loop

		if it == 1
			theta,srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,eigenvalues,eigenvectors,nb_it_basic_for_restart,nb_it_bicgstab_solve=
				jacDavRitzHarm_basic_for_restart(opt,innerLoopDim,tol_MGS,
					tol_conv,tol_eigval,tol_bicgstab)
			nb_it_restart += nb_it_basic_for_restart
			nb_it_total_bicgstab_solve += nb_it_bicgstab_solve
		
		else # Essentially for it > 1
			# Before we restart, we will create a new version of everything 
			resVec = Vector{ComplexF64}(undef, vecDim)
			resVec = restart_resVec
			hRitzTrg = Vector{ComplexF64}(undef, vecDim)
			hRitzTrg = restart_hRitzTrg
			hRitzSrc = Vector{ComplexF64}(undef, vecDim)
			hRitzSrc = restart_hRitzSrc

			eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
			eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)

			bCoeffs1 = Vector{ComplexF64}(undef, vecDim)
			bCoeffs2 = Vector{ComplexF64}(undef, vecDim)

			trgBasis = zeros(ComplexF64, vecDim, vecDim)
			trgBasis[:,1:restartDim] = restart_trgBasis

			srcBasis = zeros(ComplexF64, vecDim, vecDim) 
			srcBasis[:,1:restartDim] = restart_srcBasis

			theta = restart_theta
			
			kMat = zeros(ComplexF64, vecDim, vecDim)
			kMat[1:restartDim,1:restartDim] = restart_kMat

			for itr in restartDim+1: innerLoopDim # -restartDim # Need to determine when this for loops stops 
				# Depending on how much memory the laptop can take before crashing.
				prjCoeff = BLAS.dotc(vecDim, hRitzTrg, 1, hRitzSrc, 1)

				# Test code 
				# nb_it = 0 # Just for a rest
				# A_product = (I-ritzVec*adjoint(ritzVec))*(opt-theta*I)*(I-ritzVec*adjoint(ritzVec))
				# # direct_solve =  A_product\resVec
				# srcBasis[:, itr] = A_product\resVec # Just for a test
				# # print("direct_solve ", direct_solve,"\n")
				# # Jacobi-Davidson direction
				# # basis[:, itr],nb_it = ritz_bicgstab_matrix(opt, theta, ritzVec, resVec, 
				# # tol_bicgstab)
				# nb_it_total_bicgstab_solve += nb_it

				# Good code
				# Calculate Jacobi-Davidson direction
				srcBasis[:, itr],nb_it = harm_ritz_cg_matrix(opt, theta, hRitzTrg,
					hRitzSrc, prjCoeff, resVec, tol_bicgstab)
				# srcBasis[:, itr],nb_it = harm_ritz_bicgstab_matrix(opt, theta, hRitzTrg,
				# 		hRitzSrc, prjCoeff, resVec, tol_bicgstab)
				nb_it_total_bicgstab_solve += nb_it

				trgBasis[:, itr] = opt * srcBasis[:, itr]

				# Orthogonalize
				gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, opt,
					itr, tol_MGS)				

				# Update inverse representation of opt^{-1} in trgBasis
				kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
					view(srcBasis, :, itr))
				# Assuming opt^{-1} Hermitian matrix
				kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])

				"""
				kMat is not Hermitian so let's make it Hermitian by adding its 
				adjoint and dividing by 2
				"""
				kMat[1 : itr, 1 : itr] = (kMat[1 : itr, 1 : itr].+adjoint(kMat[1 : itr, 1 : itr]))./2

				# Eigenvalue decomposition, largest real eigenvalue last.
				# Should replace by BLAS operation
				eigSys = eigen(view(kMat, 1 : itr, 1 : itr))

				eigenvectors[1:itr,1:itr] = eigSys.vectors
				eigenvalues[1:itr] = eigSys.values

				# Update Ritz vector
				"""
				We want the largest eigenvalue in absolue value since 
				when we do 1/the largest eigenvalue it will give the smallest 
				eigenvalue that is closest to 0. 
				"""

				if abs.(eigSys.values[end]) > abs.(eigSys.values[1])
					theta = 1/eigSys.values[end]
					hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, end])
					hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, end])
				else
					theta = 1/eigSys.values[1]
					hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
					hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
				end	

				# Update residual vector
				resVec = (theta * hRitzSrc) .- hRitzTrg
		
				# Direction vector tolerance check 
				if norm(resVec) < tol_conv
					print("Restart algo converged off resVec tolerance \n")
					return real(theta),nb_it_restart,nb_it_total_bicgstab_solve
				end
				# Eigenvalue tolerance check
				if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
					if nb_it_eigval_conv == 5
						print("Restart algo converged off eigval tolerance \n")
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
		Create change of basis matrix (u_matrix), as well as trgBasis and 
		srcBasis for restart
		"""
		for i = 1:restartDim
			# Creation of new trgBasis and srcBasis matrices 
			if abs.(eigenvalues[end]) > abs.(eigenvalues[1]) 
				u_matrix[1:innerLoopDim,i] = eigenvectors[:, end]
				restart_trgBasis[:,i] = trgBasis[:, 1 : innerLoopDim] * (eigenvectors[:, end])
				restart_srcBasis[:,i] = srcBasis[:, 1 : innerLoopDim] * (eigenvectors[:, end])
				pop!(eigenvalues) # This removes the last element
				eigenvectors = eigenvectors[:, 1:end-1] # This removes the last column
			else # Essentially if abs.(eigSys.values[end]) < abs.(eigSys.values[1]
				u_matrix[1:innerLoopDim,i] = eigenvectors[:, 1]
				restart_trgBasis[:,i] = trgBasis[:, 1 : innerLoopDim] * (eigenvectors[:, 1])
				restart_srcBasis[:,i] = srcBasis[:, 1 : innerLoopDim] * (eigenvectors[:, 1])
				popfirst!(eigenvalues) # This removes the first element
				eigenvectors = eigenvectors[:, 2:end] # This removes the first column 
			end
		end
		# Orthogonalize using MGS
		gramSchmidtHarm!(restart_trgBasis, restart_srcBasis, bCoeffs1, bCoeffs2, opt,
		restartDim, tol_MGS)
		"""
		Create kMat for restart using change of basis matrix: u_matrix
		"""
		restart_kMat = adjoint(u_matrix)*kMat[1 : innerLoopDim, 1 : innerLoopDim]*u_matrix
		"""
		kMat is not Hermitian so let's make it Hermitian by adding its 
		adjoint and dividing by 2.
		"""
		restart_kMat = (restart_kMat.+adjoint(restart_kMat))./2	
		"""
		Save other stuff for restart 
		"""
		restart_resVec = resVec
		restart_hRitzTrg = hRitzTrg
		restart_hRitzSrc = hRitzSrc
		restart_theta = theta
	end 
	print("Didn't converge off tolerance for restart program. 
		Atteined max set number of iterations \n")
	return real(theta),nb_it_restart,nb_it_total_bicgstab_solve
end

# Perform Gram-Schmidt on target basis, adjusting source basis accordingly
function gramSchmidtHarm!(trgBasis, srcBasis,
	bCoeffs1::Vector{T}, bCoeffs2::Vector{T}, opt, n::Integer,
	tol::Float64) where T <: Number # ::Array{T}
	# Dimension of vector space
	dim = size(trgBasis)[1]
	# Initialize projection norm
	prjNrm = 1.0
	# Initialize projection coefficient memory
	bCoeffs1[1:(n-1)] .= 0.0 + im*0.0
	# Check that basis does not exceed dimension
	if n > dim
		error("Requested basis size exceeds dimension of vector space.")
	end
	# Norm of proposed vector
	nrm = BLAS.nrm2(dim, view(trgBasis,:,n), 1)
	# Renormalize new vector
	trgBasis[:,n] = trgBasis[:,n] ./ nrm
	srcBasis[:,n] = srcBasis[:,n] ./ nrm
	# Guarded orthogonalization
	while prjNrm > (tol * 100) && abs(nrm) > tol
		### Remove projection into existing basis
 		# Calculate projection coefficients
 		BLAS.gemv!('C', 1.0 + im*0.0, view(trgBasis, :, 1:(n-1)),
 			view(trgBasis, :, n), 0.0 + im*0.0,
 			view(bCoeffs2, 1:(n -1)))
 		# Remove projection coefficients
 		BLAS.gemv!('N', -1.0 + im*0.0, view(trgBasis, :, 1:(n-1)),
 			view(bCoeffs2, 1:(n -1)), 1.0 + im*0.0,
 			view(trgBasis, :, n))
 		# Update total projection coefficients
 		bCoeffs1 .= bCoeffs2 .+ bCoeffs1
 		# Calculate projection norm
 		prjNrm = BLAS.nrm2(n-1, bCoeffs2, 1)
 	end
 	# Remaining norm after removing projections
 	nrm = BLAS.nrm2(dim, view(trgBasis,:,n), 1)
	# Check that remaining vector is sufficiently large
	if abs(nrm) < tol
		# Switch to random search direction
		rand!(view(srcBasis, :, n))
		trgBasis[:, n] = opt * srcBasis[:, n]
		gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2,
			opt, n, tol)
	else
		# Renormalize
		trgBasis[:,n] = trgBasis[:,n] ./ nrm
		srcBasis[:,n] = srcBasis[:,n] ./ nrm
		bCoeffs1 .= bCoeffs1 ./ nrm
		# Remove projections from source vector
		BLAS.gemv!('N', -1.0 + im*0.0, view(srcBasis, :, 1:(n-1)),
 			view(bCoeffs1, 1:(n-1)), 1.0 + im*0.0, view(srcBasis, :, n))
	end
end

# Pseudo-projections for harmonic Ritz vector calculations
@inline function harmVec(dim::Integer, pTrg::Vector{T}, pSrc::Vector{T},
	prjCoeff::Number, sVec::Array{T})::Array{T} where T <: Number
	return sVec .- ((BLAS.dotc(dim, pTrg, 1, sVec, 1) / prjCoeff) .* pSrc)
end
# This is a biconjugate gradient program without a preconditioner.
# m is the maximum number of iterations
function harm_ritz_bicgstab_matrix(A, theta, hTrg, hSrc, prjC, b, tol_bicgstab)
	dim = size(A)[1]
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b
    rho_m1 = alpha = omega_m1 = 1
	k = 0 
    for k in 1 : 1000
        rho_k = real((adjoint(r0)*r_m1)[1])
        # Bêta calculation
        # First term
        first_term = rho_k/rho_m1
        # Second term
        second_term = alpha/omega_m1
        # Calculation
        beta = first_term*second_term
        pk = r_m1 .+ beta.*(p_m1-omega_m1.*v_m1)
        pkPrj = harmVec(dim, hTrg, hSrc, prjC, pk)
        vk = harmVec(dim, hTrg, hSrc, prjC, A * pkPrj .- (theta .* pkPrj))
        # Alpha calculation
        # Bottom term
        bottom_term = real((adjoint(r0)*vk)[1])
        # Calculation
        alpha = rho_k / bottom_term
        h = xk_m1 + alpha.*pk
        s = r_m1 - alpha.*vk
        sPrj = harmVec(dim, hTrg, hSrc, prjC, s)
        t = harmVec(dim, hTrg, hSrc, prjC, A * sPrj .- (theta .* sPrj))
        # Omega_k calculation
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
			return xk_m1,k # k is essentially the number of iterations 
			# to reach the chosen tolerance
        end
    end
	print("Didn't converge off bicgstab tolerance \n")
    return xk_m1,k # k is essentially the number of iterations 
end

function harm_ritz_cg_matrix(A,theta, hTrg, hSrc, prjC,b,cg_tol)
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
        pkPrj = harmVec(dim, hTrg, hSrc, prjC, pk)
        A_pkPrj = harmVec(dim, hTrg, hSrc, prjC, A * pkPrj .- (theta .* pkPrj))
        # A_pk = A*pk
        pk_A_pkPrj = dot(pk,A_pkPrj) # conj.(transpose(pk))*A_pk
        # Division
        alpha_k = rkrk/pk_A_pkPrj

        rk_old = rk

        # x_{k+1} calculation 
        xk = xk + alpha_k.*pk

        # r_{k+1} calculation 
        rk = rk - alpha_k.*A_pkPrj

        # print("norm(rk_plus1) ",norm(rk), "\n")
        if norm(rk)  < cg_tol
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
    return xk, nb_it
end 

# function harm_ritz_bicgstab_matrix(A, theta, hTrg, hSrc, prjC, b, tol_bicgstab)
#     dim = length(b)
# 	max_nb_it = 10000
#     xk_m1 = Vector{Float64}(undef, dim)
#     rand!(xk_m1)
#     rk_m1 = b-A*xk_m1
#     r0_hat = Vector{Float64}(undef, dim)
#     rand!(r0_hat)
#     print("dot(r0_hat,rk_m1) ", dot(r0_hat,rk_m1),"\n")
#     pk_m1 = rk_m1
#     alpha = 1
#     nb_it = 0
#     for k in 1:max_nb_it
#         # alpha calculation
#         pkPrj = harmVec(dim, hTrg, hSrc, prjC, pk_m1)
#         A_pk_m1 = harmVec(dim, hTrg, hSrc, prjC, A * pkPrj .- (theta .* pkPrj))
         
#         # A_pk_m1 = A*pk_m1
#         alpha_top_term = dot(r0_hat,rk_m1)
#         alpha_bottom_term = dot((A_pk_m1),r0_hat)
#         alpha = alpha_top_term/alpha_bottom_term

#         s = rk_m1-alpha*A_pk_m1

#         # omega calculation 
#         sPrj = harmVec(dim, hTrg, hSrc, prjC, s)
#         A_s = harmVec(dim, hTrg, hSrc, prjC, A * sPrj .- (theta .* sPrj))
#         # A_s = A*s
#         omega_top_term = dot(A_s,s)
#         omega_bottom_term = dot(A_s,A_s)
#         omega = omega_top_term/omega_bottom_term

#         xk_m1 = xk_m1 + alpha*pk_m1 + omega*s

#         rk = s - omega*A_s

#         # if norm(rk)/norm(b) < tol_bicgstab
#         if norm(rk) < tol_bicgstab
# 			# print("rk ", rk, "\n")
#             print("norm(rk) ", norm(rk), "\n")
# 			return xk_m1,nb_it 
#         end
#         # bêta calculation 
#         beta = (alpha/omega)*(dot(rk,r0_hat)/dot(rk_m1,r0_hat))
#         pk_m1 = rk + beta*(pk_m1-omega*A_pk_m1)
#         rk_m1 = rk
#         nb_it += 1
#     end
#     print("xk_m1 ", xk_m1, "\n")
#     print("nb_it ", nb_it, "\n")
#     print("Reached max number of iterations \n")
#     print("Restart a new function call \n")
#     xk_m1,nb_it = harm_ritz_bicgstab_matrix(A, theta, hTrg, hSrc, prjC, b, tol_bicgstab)
#     return xk_m1,nb_it
# end

# # Test code 
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
# dims = size(opt)
# # print("dims[1] ", dims[1], "\n")
# # print("dims[2] ", dims[2], "\n")
# bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# kMat = zeros(ComplexF64, dims[2], dims[2])
# # val = jacDavRitzHarm(trgBasis, srcBasis, kMat, opt, dims[1], dims[2], 1.0e-6)
# val,nb_it_restart,nb_it_total_bicgstab_solve = jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)


# println("The Julia smallest eigenvalue is ", minEig,".")
# print("Harmonic Ritz eigenvalue closest to 0 is ", val, "\n")
end
