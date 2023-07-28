module Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson_operator

export jacDavRitzHarm_basic,jacDavRitzHarm_basic_for_restart,jacDavRitzHarm_restart

using LinearAlgebra, Random, Base.Threads, Plots, product


function jacDavRitzHarm_basic(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,
	innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64,
		tol_eigval::Float64, tol_bicgstab::Float64)::Tuple{Float64,Float64,Float64}
	
	dim = cellsA[1]*cellsA[2]*cellsA[3]*3
	
	# Matrices memory initialization
	trgBasis = Array{ComplexF64}(undef, dim, dim)
	srcBasis = Array{ComplexF64}(undef, dim, dim)
	kMat = zeros(ComplexF64, dim, dim)

	# Set starting vector
	# Memory initialization
	resVec = Vector{ComplexF64}(undef, dim)
	hRitzTrg = Vector{ComplexF64}(undef, dim)
	hRitzSrc = Vector{ComplexF64}(undef, dim)
	bCoeffs1 = Vector{ComplexF64}(undef, dim)
	bCoeffs2 = Vector{ComplexF64}(undef, dim)
	# Set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# Normalize starting vector
	nrm = BLAS.nrm2(dim, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# Algorithm initialization
	# trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
	trgBasis[:, 1] = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
		gMemSlfA,cellsA,chi_inv_coeff,P,srcBasis[:, 1])
	nrm = BLAS.nrm2(dim, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# Representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(dim, view(trgBasis, :, 1), 1,
		view(srcBasis, :, 1), 1) # Kk
	# Ritz value
	theta = 1 / kMat[1,1] # eigenvalue 
	# Ritz vectors
	hRitzTrg[:] = trgBasis[:, 1] # hk = wk 
	hRitzSrc[:] = srcBasis[:, 1] # fk = vk

	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

	# Initialize some parameters
	previous_eigval = theta
	nb_it_vals_basic = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0
	
	# Code for if we just want the inner loop, so with no restart  
	for itr in 2 : innerLoopDim  # Need to determine when this for loops stops 
		# depending on how much memory the laptop can take before crashing.
		prjCoeff = BLAS.dotc(dim, hRitzTrg, 1, hRitzSrc, 1)
		# Calculate Jacobi-Davidson direction
		print("Start of bicgstab in basic \n ")
		
		srcBasis[:, itr],nb_it = harmonic_ritz_cg_operator(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
			P, theta, hRitzTrg, hRitzSrc, prjCoeff, resVec, tol_bicgstab)
		# srcBasis[:, itr],nb_it = harmonic_ritz_bicgstab_operator(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
		# 	P, theta, hRitzTrg, hRitzSrc, prjCoeff, resVec, tol_bicgstab)

		print("Done of bicgstab in basic \n ")
		nb_it_total_bicgstab_solve += nb_it
		
		trgBasis[:, itr] = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
			gMemSlfA,cellsA,chi_inv_coeff,P,srcBasis[:, itr])

		# Orthogonalize
		print("Start of MGS in basic \n ")
		gramSchmidtHarm!(alpha_0,alpha_1,xi,P_0,trgBasis, srcBasis, bCoeffs1, bCoeffs2, gMemSlfN,
			gMemSlfA,cellsA,chi_inv_coeff,P,itr, tol_MGS)
		print("End of MGS in basic \n ")
		

		# Update inverse representation of opt^{-1} in trgBasis
		kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
			view(srcBasis, :, itr))
		# Assuming opt^{-1} Hermitian matrix
		kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])
		
		# Eigenvalue decomposition, largest real eigenvalue last.
		# should replace by BLAS operation
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
			return real(theta), nb_it_vals_basic,nb_it_total_bicgstab_solve
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
	return real(theta),nb_it_vals_basic,nb_it_total_bicgstab_solve
end

function jacDavRitzHarm_basic_for_restart(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,
	innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64,
		tol_eigval::Float64, tol_bicgstab::Float64)

	dim = cellsA[1]*cellsA[2]*cellsA[3]*3	

	# Matrices memory initialization
	trgBasis = Array{ComplexF64}(undef, dim, dim)
	srcBasis = Array{ComplexF64}(undef, dim, dim)
	kMat = zeros(ComplexF64, dim, dim)
	# Memory initialization
	resVec = Vector{ComplexF64}(undef, dim)
	hRitzTrg = Vector{ComplexF64}(undef, dim)
	hRitzSrc = Vector{ComplexF64}(undef, dim)
	bCoeffs1 = Vector{ComplexF64}(undef, dim)
	bCoeffs2 = Vector{ComplexF64}(undef, dim)
	# Set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# Normalize starting vector
	nrm = BLAS.nrm2(dim, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# Algorithm initialization
	trgBasis[:, 1] = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
		gMemSlfA,cellsA,chi_inv_coeff,P,srcBasis[:, 1])

	nrm = BLAS.nrm2(dim, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# Representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(dim, view(trgBasis, :, 1), 1,
		view(srcBasis, :, 1), 1) # Kk
	# Ritz value
	theta = 1 / kMat[1,1] # eigenvalue 
	# Ritz vectors
	hRitzTrg[:] = trgBasis[:, 1] # hk = wk 
	hRitzSrc[:] = srcBasis[:, 1] # fk = vk

	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

	# Initialize some parameters
	previous_eigval = theta
	nb_it_vals_basic_for_restart = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
	
	# Code for if we just want the inner loop, so with no restart  
	for itr in 2 : innerLoopDim  # Need to determine when this for loops stops 
		# Depending on how much memory the laptop can take before crashing.
		prjCoeff = BLAS.dotc(dim, hRitzTrg, 1, hRitzSrc, 1)
		# Calculate Jacobi-Davidson direction
		print("Start of bicgstab in basic \n ")
		# srcBasis[:, itr],nb_it = harmonic_ritz_cg_operator(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
		# 	P, theta, hRitzTrg, hRitzSrc, prjCoeff, resVec, tol_bicgstab)
		srcBasis[:, itr],nb_it = harmonic_ritz_bicgstab_operator(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
			P, theta, hRitzTrg, hRitzSrc, prjCoeff, resVec, tol_bicgstab)
		print("Done of bicgstab in basic \n ")
		nb_it_total_bicgstab_solve += nb_it
			
		trgBasis[:, itr] = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
			gMemSlfA,cellsA,chi_inv_coeff,P,srcBasis[:, itr])

		# Orthogonalize
		print("Start of MGS in basic \n ")
		gramSchmidtHarm!(alpha_0,alpha_1,xi,P_0,trgBasis, srcBasis, bCoeffs1, bCoeffs2, gMemSlfN,
			gMemSlfA,cellsA,chi_inv_coeff,P,itr, tol_MGS)
		print("End of MGS in basic \n ")

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
			return real(theta),srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,eigenvalues,eigenvectors,nb_it_vals_basic_for_restart,nb_it_total_bicgstab_solve
			# println(real(theta))
		end
		# Eigenvalue tolerance check
		if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
			if nb_it_eigval_conv == 5
				print("Basic for restrart algo converged off eigval tolerance \n")
				return real(theta),srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,eigenvalues,eigenvectors,nb_it_vals_basic_for_restart,nb_it_total_bicgstab_solve
			end 
			nb_it_eigval_conv += 1
		end
		previous_eigval = theta
		nb_it_vals_basic_for_restart += 1
	end
 
	print("Didn't converge off tolerance for basic for restart program. 
		Atteined max set number of iterations \n")
	return real(theta),srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,eigenvalues,eigenvectors,nb_it_vals_basic_for_restart,nb_it_total_bicgstab_solve
end

function jacDavRitzHarm_restart(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
	P,innerLoopDim::Integer,restartDim::Integer,
		tol_MGS::Float64,tol_conv::Float64,tol_eigval::Float64, 
			tol_bicgstab::Float64)::Tuple{Float64,Float64,Float64} # ::Float64

	dim = cellsA[1]*cellsA[2]*cellsA[3]*3

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
	
	restart_resVec = Vector{ComplexF64}(undef, dim)
	restart_hRitzTrg = Vector{ComplexF64}(undef, dim)
	restart_hRitzSrc = Vector{ComplexF64}(undef, dim)
	restart_trgBasis = Array{ComplexF64}(undef, dim, restartDim)
	restart_srcBasis = Array{ComplexF64}(undef, dim, restartDim)
	restart_kMat = zeros(ComplexF64, dim, restartDim)
	restart_theta = 0 

	# Change of basis matrix
	u_matrix = Array{ComplexF64}(undef, innerLoopDim, restartDim)


	# Matrices memory initialization
	trgBasis = Array{ComplexF64}(undef, dim, dim)
	srcBasis = Array{ComplexF64}(undef, dim, dim)
	kMat = zeros(ComplexF64, dim, dim)
	# Vectors memory initialization
	resVec = Vector{ComplexF64}(undef, dim)
	hRitzTrg = Vector{ComplexF64}(undef, dim)
	hRitzSrc = Vector{ComplexF64}(undef, dim)
	bCoeffs1 = Vector{ComplexF64}(undef, dim)
	bCoeffs2 = Vector{ComplexF64}(undef, dim)
	# Set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# Normalize starting vector
	# nrm = BLAS.nrm2(vecDim, view(srcBasis,:,1), 1) # norm(vk)
	nrm = BLAS.nrm2(dim, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# Algorithm initialization
	# trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
	trgBasis[:, 1] = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
		gMemSlfA,cellsA,chi_inv_coeff,P,srcBasis[:, 1])

	nrm = BLAS.nrm2(dim, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# Representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(dim, view(trgBasis, :, 1), 1,
		view(srcBasis, :, 1), 1) # Kk
	# Ritz value
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
	for it in 1:2000 # restartDim # Need to think this over 
		eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
		# Inner loop
		if it == 1
			print("Start of basic HRitz for restart \n ")
			theta,srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,eigenvalues,eigenvectors,nb_it_basic_for_restart,nb_it_bicgstab_solve=
				jacDavRitzHarm_basic_for_restart(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,
				innerLoopDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
			print("End of basic HRitz for restart \n ")
			nb_it_restart += nb_it_basic_for_restart
			nb_it_total_bicgstab_solve += nb_it_bicgstab_solve

		else # Essentially for it > 1
			# Before we restart, we will create a new version of everything 
			resVec = Vector{ComplexF64}(undef, dim)
			resVec = restart_resVec
			hRitzTrg = Vector{ComplexF64}(undef, dim)
			hRitzTrg = restart_hRitzTrg
			hRitzSrc = Vector{ComplexF64}(undef, dim)
			hRitzSrc = restart_hRitzSrc

			eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
			eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)

			
			bCoeffs1 = Vector{ComplexF64}(undef, dim)
			bCoeffs2 = Vector{ComplexF64}(undef, dim)
			
			trgBasis = Array{ComplexF64}(undef, dim, dim) # dims[1],dims[2]
			trgBasis[:,1:restartDim] = restart_trgBasis

			srcBasis = Array{ComplexF64}(undef, dim, dim) # dims[1],dims[2]
			srcBasis[:,1:restartDim] = restart_srcBasis

			theta = restart_theta

			kMat = zeros(ComplexF64, dim, dim)
			kMat[1:restartDim,1:restartDim] = restart_kMat
		
			for itr in restartDim+1: innerLoopDim # -restartDim # Need to determine when this for loops stops 
				# depending on how much memory the laptop can take before crashing.
				prjCoeff = BLAS.dotc(dim, hRitzTrg, 1, hRitzSrc, 1)
				# Calculate Jacobi-Davidson direction
				print("Start of bicgstab in restart \n ")
				srcBasis[:, itr],nb_it = harmonic_ritz_cg_operator(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
					P, theta, hRitzTrg, hRitzSrc, prjCoeff, resVec, tol_bicgstab)
				# srcBasis[:, itr], nb_it = harmonic_ritz_bicgstab_operator(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
				# 	P, theta, hRitzTrg, hRitzSrc, prjCoeff, resVec, tol_bicgstab)
				print("Enf of bicgstab in restart \n ")
				nb_it_total_bicgstab_solve += nb_it 

				trgBasis[:, itr] = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
					gMemSlfA,cellsA,chi_inv_coeff,P,srcBasis[:, itr])
					# A_v_product(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
					# 	P,alpha_coeff,srcBasis[:, itr])
				# opt * srcBasis[:, itr]

				# Orthogonalize
				print("Start of 1rst MGS in basic \n ")
				gramSchmidtHarm!(alpha_0,alpha_1,xi,P_0,trgBasis, srcBasis, bCoeffs1, bCoeffs2, gMemSlfN,
					gMemSlfA,cellsA,chi_inv_coeff,P,itr, tol_MGS)
				print("End of 1rst MGS in restart \n ")				

				# Update inverse representation of opt^{-1} in trgBasis
				kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
					view(srcBasis, :, itr))
				# Assuming opt^{-1} Hermitian matrix
				kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])
				
				# kMat is not Hermitian so let's make it Hermitian by adding its 
				# adjoint and dividing by 2
				kMat[1 : itr, 1 : itr] = (kMat[1 : itr, 1 : itr].+adjoint(kMat[1 : itr, 1 : itr]))./2

				# Eigenvalue decomposition, largest real eigenvalue last.
				# should replace by BLAS operation
				eigSys = eigen(view(kMat, 1 : itr, 1 : itr))

				eigenvectors[1:itr,1:itr] = eigSys.vectors
				eigenvalues[1:itr] = eigSys.values

				# Update Ritz vector
				"""
				- We want the largest eigenvalue in absolue value since 
				it will give the smallest eigenvalue that is closest to 0. 
				"""

				if abs.(eigSys.values[end]) > abs.(eigSys.values[1])
					if_val = 1
					theta = 1/eigSys.values[end]
					hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, end])
					hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, end])
				else
					if_val = 2
					theta = 1/eigSys.values[1]
					hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
					hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
				end	

		
				# Update residual vector
				resVec = (theta * hRitzSrc) .- hRitzTrg
		
				# Direction vector tolerance check
				if norm(resVec) < tol_conv
					return real(theta), nb_it_restart
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
		Create change of basis matrix: u_matrix
		# Create trgBasis and srcBasis for restart
		"""
		for i = 1:restartDim

			# Creation of new trgBasis and srcBasis matrices 
			if abs.(eigenvalues[end]) > abs.(eigenvalues[1]) 
				u_matrix[1:innerLoopDim,i] = eigenvectors[:, end]
				restart_trgBasis[:,i] = trgBasis[:, 1 : innerLoopDim] * (eigenvectors[:, end])
				restart_srcBasis[:,i] = srcBasis[:, 1 : innerLoopDim] * (eigenvectors[:, end])
				pop!(eigenvalues) # This removes the last element
				eigenvectors = eigenvectors[:, 1:end-1] # This removes the last column
			else # abs.(eigSys.values[end]) < abs.(eigSys.values[1])
				u_matrix[1:innerLoopDim,i] = eigenvectors[:, 1]
				restart_trgBasis[:,i] = trgBasis[:, 1 : innerLoopDim] * (eigenvectors[:, 1])
				restart_srcBasis[:,i] = srcBasis[:, 1 : innerLoopDim] * (eigenvectors[:, 1])
				popfirst!(eigenvalues) # This removes the first element
				eigenvectors = eigenvectors[:, 2:end] # This removes the first column 
			end
		end

		# Orthogonalize
		print("Start of 2nd MGS in restart \n ")
		gramSchmidtHarm!(alpha_0,alpha_1,xi,P_0,restart_trgBasis, restart_srcBasis, bCoeffs1, bCoeffs2, gMemSlfN,
					gMemSlfA,cellsA,chi_inv_coeff,P,restartDim, tol_MGS)
		print("Enf of 2nd MGS in restart \n ")

		"""
		Create kMat for restart
		"""
		restart_kMat = adjoint(u_matrix)*kMat[1 : innerLoopDim, 1 : innerLoopDim]*u_matrix

		# kMat is not Hermitian so let's make it Hermitian by adding its 
		# adjoint and dividing by 2
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


# perform Gram-Schmidt on target basis, adjusting source basis accordingly
function gramSchmidtHarm!(alpha_0,alpha_1,xi,P_0,trgBasis::Array{T}, srcBasis::Array{T},
	bCoeffs1::Vector{T}, bCoeffs2::Vector{T}, gMemSlfN,gMemSlfA,cellsA,
		chi_inv_coeff,P, n::Integer,tol::Float64) where T <: Number
	
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
		# Remove projection into existing basis
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

		# G operator product 
		# trgBasis[:, n] = opt * srcBasis[:, n]
		trgBasis[:, n] = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
			gMemSlfA,cellsA,chi_inv_coeff,P,srcBasis[:, n])

		gramSchmidtHarm!(alpha_0,alpha_1,xi,P_0,trgBasis, srcBasis, bCoeffs1, bCoeffs2,
			gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P, n, tol)
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

# pseudo-projections for harmonic Ritz vector calculations
@inline function harmVec(dim::Integer, pTrg::Vector{T}, pSrc::Vector{T},
	prjCoeff::Number, sVec::Array{T})::Array{T} where T <: Number
	return sVec .- ((BLAS.dotc(dim, pTrg, 1, sVec, 1) / prjCoeff) .* pSrc)
end

# This is a biconjugate gradient program without a preconditioner.
# m is the maximum number of iterations
function harmonic_ritz_bicgstab_operator(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
	P, theta, hTrg, hSrc, prjC, b, tol_bicgstab)
	dim = cellsA[1]*cellsA[2]*cellsA[3]*3
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b
    rho_m1 = alpha = omega_m1 = 1
    for k in 1 : 1000
        rho_k = conj.(transpose(r0))*r_m1
        # Bêta calculation
        # First term
        first_term = rho_k/rho_m1
        # Second term
        second_term = alpha/omega_m1
        # Calculation
        beta = first_term*second_term
        pk = r_m1 .+ beta.*(p_m1-omega_m1.*v_m1)
        pkPrj = harmVec(dim, hTrg, hSrc, prjC, pk)
		A_pkPrj = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
		gMemSlfA,cellsA,chi_inv_coeff,P,pkPrj)
		vk = harmVec(dim, hTrg, hSrc, prjC, A_pkPrj .- (theta .* pkPrj))

        # alpha calculation
        # Bottom term
        bottom_term = conj.(transpose(r0))*vk
        # Calculation
        alpha = rho_k / bottom_term
        h = xk_m1 + alpha.*pk
        s = r_m1 - alpha.*vk
        sPrj = harmVec(dim, hTrg, hSrc, prjC, s)
		A_sPrj = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
			gMemSlfA,cellsA,chi_inv_coeff,P,sPrj)
		t = harmVec(dim, hTrg, hSrc, prjC,A_sPrj .- (theta .* sPrj))

        # omega_k calculation
        # Top term
        ts = conj.(transpose(t))*s
        # Bottom term
        tt = conj.(transpose(t))*t
        # Calculation
        omega_k = ts ./ tt
        xk_m1 = h + omega_k.*s
        r_old = r_m1
        r_m1 = s-omega_k.*t
		rho_m1 = rho_k
		if norm(r_m1) < tol_bicgstab
			return xk_m1,k # k is essentially the number of iterations
			# to reach the chosen tolerance 
        end
    end
	print("Didn't converge off bicgstab tolerance \n")
    return xk_m1,k # k is essentially the number of iterations 
end

function harmonic_ritz_cg_operator(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
	P, theta, hTrg, hSrc, prjC, b, tol_cg)
    # tol = 1e-5 # The program terminates once 
    # there is an r for which its norm is smaller
    # than the chosen tolerance. 
	dim = cellsA[1]*cellsA[2]*cellsA[3]*3
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
        A_pkPrj = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
			gMemSlfA,cellsA,chi_inv_coeff,P,pkPrj)
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

end
