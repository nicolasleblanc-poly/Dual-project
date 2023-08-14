using LinearAlgebra, Random, Base.Threads, Plots
function jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt::Array{ComplexF64}, 
	vecDim::Integer, repDim::Integer, innerLoopDim::Integer,tol_MGS::Float64,
	tol_conv::Float64,tol_eigval::Float64,tol_bicgstab::Float64,solver_method)::Tuple{Float64, Float64}
# function jacDavRitzHarm_basic(trgBasis::Array{ComplexF64}, 
# 	srcBasis::Array{ComplexF64}, kMat::Array{ComplexF64}, opt::Array{ComplexF64}, 
# 	vecDim::Integer, repDim::Integer, innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64)::Float64
	### Memory initialization
	resVec = Vector{ComplexF64}(undef, vecDim)
	hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	hRitzSrc = Vector{ComplexF64}(undef, vecDim)
	bCoeffs1 = Vector{ComplexF64}(undef, repDim)
	bCoeffs2 = Vector{ComplexF64}(undef, repDim)
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
	# print("size(hRitzSrc) ", size(hRitzSrc), "\n")

	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

	previous_eigval = theta
	nb_it_vals_basic = 0.0
	nb_it_eigval_conv = 0.0
	
	# Code for if we just want the inner loop, so with no restart  
	for itr in 2 : innerLoopDim  # Need to determine when this for loops stops 
		# Depending on how much memory the laptop can take before crashing.
		prjCoeff = BLAS.dotc(vecDim, hRitzTrg, 1, hRitzSrc, 1)
		# Calculate Jacobi-Davidson direction

		if solver_method == "bicgstab"
			srcBasis[:, itr] = bicgstab_matrix(opt, theta, hRitzTrg,
				hRitzSrc, prjCoeff, resVec, tol_bicgstab)
		elseif solver_method == "direct"
			# # Test method 
			# # uk_tilde = bicgstab_matrix(opt, theta, hRitzTrg,
			# # 	hRitzSrc, prjCoeff, hRitzTrg, tol_bicgstab)
			# uk_tilde = opt\hRitzSrc # Try to use bicgstab to solve for this 
			# # rk_tilde = bicgstab_matrix(opt, theta, hRitzTrg,
			# # 	hRitzSrc, prjCoeff, hRitzSrc, tol_bicgstab)
			# rk_tilde = opt\hRitzTrg # Try to use bicgstab to solve for this

			# OG method 
			# uk_tilde = bicgstab_matrix(opt, theta, hRitzTrg,
			# 	hRitzSrc, prjCoeff, hRitzTrg, tol_bicgstab)
			uk_tilde = (opt-theta*I)\hRitzTrg # Try to use bicgstab to solve for this 
			# rk_tilde = bicgstab_matrix(opt, theta, hRitzTrg,
			# 	hRitzSrc, prjCoeff, hRitzSrc, tol_bicgstab)
			rk_tilde = (opt-theta*I)\hRitzSrc # Try to use bicgstab to solve for this 
			epsilon = (adjoint(hRitzTrg)*uk_tilde)/(adjoint(hRitzSrc)*rk_tilde)
			srcBasis[:, itr] = epsilon*uk_tilde-rk_tilde
		else
			print("Didn't enter a correct solver method \n")
		end
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
			return real(theta),nb_it_vals_basic
			# println(real(theta))
		end
		# Eigenvalue tolerance check
		# print("previous_eigval ", previous_eigval, "\n")
		# print("theta ", theta, "\n")
		# print("(real(theta) - real(previous_eigval))/real(previous_eigval) ", 
		# 	(real(theta) - real(previous_eigval))/real(previous_eigval), "\n")
		if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
			if nb_it_eigval_conv == 5
				print("Basic algo converged off eigval tolerance \n")
				return real(theta),nb_it_vals_basic
			end 
			nb_it_eigval_conv += 1
		end
		previous_eigval = theta
		nb_it_vals_basic += 1
		# print("norm(resVec) basic program ", norm(resVec),"\n")
	end
 
	print("Didn't converge off tolerance for basic program. 
		Atteined max set number of iterations \n")
	return real(theta),nb_it_vals_basic
end


function jacDavRitzHarm_basic_for_restart(trgBasis, srcBasis, kMat, opt::Array{ComplexF64}, 
	vecDim::Integer, repDim::Integer, innerLoopDim::Integer,tol_MGS::Float64,
	tol_conv::Float64,tol_eigval::Float64,tol_bicgstab::Float64,solver_method)
# function jacDavRitzHarm_basic(trgBasis::Array{ComplexF64}, 
# 	srcBasis::Array{ComplexF64}, kMat::Array{ComplexF64}, opt::Array{ComplexF64}, 
# 	vecDim::Integer, repDim::Integer, innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64)::Float64
	### Memory initialization
	resVec = Vector{ComplexF64}(undef, vecDim)
	hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	hRitzSrc = Vector{ComplexF64}(undef, vecDim)
	bCoeffs1 = Vector{ComplexF64}(undef, repDim)
	bCoeffs2 = Vector{ComplexF64}(undef, repDim)
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
	# print("size(hRitzSrc) ", size(hRitzSrc), "\n")

	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

	previous_eigval = theta
	nb_it_vals_basic_for_restart = 0.0
	nb_it_eigval_conv = 0.0

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	# eigenvectors = Array{ComplexF64}(undef, innerLoopDim,restartDim)
	eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
	
	# Code for if we just want the inner loop, so with no restart  
	for itr in 2 : innerLoopDim  # Need to determine when this for loops stops 
		# Depending on how much memory the laptop can take before crashing.
		prjCoeff = BLAS.dotc(vecDim, hRitzTrg, 1, hRitzSrc, 1)
		# Calculate Jacobi-Davidson direction
		if solver_method == "bicgstab"
			srcBasis[:, itr] = bicgstab_matrix(opt, theta, hRitzTrg,
				hRitzSrc, prjCoeff, resVec, tol_bicgstab)
		elseif solver_method == "direct"
			# uk_tilde = bicgstab_matrix(opt, theta, hRitzTrg,
			# 	hRitzSrc, prjCoeff, hRitzTrg, tol_bicgstab)
			uk_tilde = (opt-theta*I)\hRitzTrg # Try to use bicgstab to solve for this 
			# rk_tilde = bicgstab_matrix(opt, theta, hRitzTrg,
			# 	hRitzSrc, prjCoeff, hRitzSrc, tol_bicgstab)
			rk_tilde = (opt-theta*I)\hRitzSrc # Try to use bicgstab to solve for this 
			epsilon = (adjoint(hRitzTrg)*uk_tilde)/(adjoint(hRitzSrc)*rk_tilde)
			srcBasis[:, itr] = epsilon*uk_tilde-rk_tilde
		else
			print("Didn't enter a correct solver method \n")
		end
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
			return real(theta),srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,eigenvalues,eigenvectors,nb_it_vals_basic_for_restart
			# println(real(theta))
		end
		# Eigenvalue tolerance check
		if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
			if nb_it_eigval_conv == 5
				print("Basic for restrart algo converged off eigval tolerance \n")
				return real(theta),srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,eigenvalues,eigenvectors,nb_it_vals_basic_for_restart
			end 
			nb_it_eigval_conv += 1
		end
		previous_eigval = theta
		nb_it_vals_basic_for_restart += 1
		# print("norm(resVec) basic program ", norm(resVec),"\n")
	end
 
	print("Didn't converge off tolerance for basic for restart program. 
		Atteined max set number of iterations \n")
	return real(theta),srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,eigenvalues,eigenvectors,nb_it_vals_basic_for_restart
end

function jacDavRitzHarm_restart(trgBasis::Array{ComplexF64}, 
	srcBasis::Array{ComplexF64}, kMat::Array{ComplexF64}, 
		opt::Array{ComplexF64}, vecDim::Integer, repDim::Integer, 
			innerLoopDim::Integer,restartDim::Integer,tol_MGS::Float64,
				tol_conv::Float64,tol_eigval::Float64,tol_bicgstab::Float64,
					solver_method)::Tuple{Float64, Float64}

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	# eigenvectors = Array{ComplexF64}(undef, innerLoopDim,restartDim)
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

	### memory initialization
	resVec = Vector{ComplexF64}(undef, vecDim)
	hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	hRitzSrc = Vector{ComplexF64}(undef, vecDim)
	bCoeffs1 = Vector{ComplexF64}(undef, repDim)
	bCoeffs2 = Vector{ComplexF64}(undef, repDim)
	# set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	### algorithm initialization
	trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
	nrm = BLAS.nrm2(vecDim, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# representation of opt^{-1} in trgBasis
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

	if_val = 0

	previous_eigval = theta
	nb_it_restart = 0.0
	nb_it_eigval_conv = 0.0
	
	# innerLoopDim = Int(repDim/4)

	# Code with restart
	# Outer loop
	for it in 1: 1000 # restartDim # Need to think this over 
		eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
		# Inner loop

		if it == 1
			theta,srcBasis,trgBasis,kMat,resVec,hRitzSrc,hRitzTrg,eigenvalues,eigenvectors,nb_it_basic_for_restart=
				jacDavRitzHarm_basic_for_restart(trgBasis, srcBasis, kMat, opt, 
				vecDim, vecDim, innerLoopDim, tol_MGS, tol_conv,tol_eigval,tol_bicgstab,solver_method)
			nb_it_restart += nb_it_basic_for_restart
		
		else # Essentially for it > 1
			# print("Second loop \n")
			# Before we restart, we will create a new version of everything 
			resVec = Vector{ComplexF64}(undef, vecDim)
			resVec = restart_resVec
			hRitzTrg = Vector{ComplexF64}(undef, vecDim)
			hRitzTrg = restart_hRitzTrg
			hRitzSrc = Vector{ComplexF64}(undef, vecDim)
			hRitzSrc = restart_hRitzSrc

			eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
			eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)

			bCoeffs1 = Vector{ComplexF64}(undef, repDim)
			bCoeffs2 = Vector{ComplexF64}(undef, repDim)

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
				# Calculate Jacobi-Davidson direction
				if solver_method == "bicgstab"
					srcBasis[:, itr] = bicgstab_matrix(opt, theta, hRitzTrg,
						hRitzSrc, prjCoeff, resVec, tol_bicgstab)
				elseif solver_method == "direct"
					# uk_tilde = bicgstab_matrix(opt, theta, hRitzTrg,
					# 	hRitzSrc, prjCoeff, hRitzTrg, tol_bicgstab)
					uk_tilde = (opt-theta*I)\hRitzTrg # Try to use bicgstab to solve for this 
					# rk_tilde = bicgstab_matrix(opt, theta, hRitzTrg,
					# 	hRitzSrc, prjCoeff, hRitzSrc, tol_bicgstab)
					rk_tilde = (opt-theta*I)\hRitzSrc # Try to use bicgstab to solve for this 
					epsilon = (adjoint(hRitzTrg)*uk_tilde)/(adjoint(hRitzSrc)*rk_tilde)
					srcBasis[:, itr] = epsilon*uk_tilde-rk_tilde
				else
					print("Didn't enter a correct solver method \n")
				end
				trgBasis[:, itr] = opt * srcBasis[:, itr]

				# orthogonalize
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

				# print("kMat 2 ", kMat, "\n")

				# Let's check if kMat is Hermitian by calling the ishermitian
				# function on it
				# print("Test if kMat is hermitian ", 
				# 	ishermitian(kMat[1 : itr, 1 : itr]), "\n")	
				# (opt .+ adjoint(opt)) ./ 2

				# Eigenvalue decomposition, largest real eigenvalue last.
				# Should replace by BLAS operation
				eigSys = eigen(view(kMat, 1 : itr, 1 : itr))

				eigenvectors[1:itr,1:itr] = eigSys.vectors
				eigenvalues[1:itr] = eigSys.values

				# Update Ritz vector
				"""
				- We want the largest eigenvalue in absolue value since 
				when we do 1/the largest eigenvalue it will give the smallest 
				eigenvalue that is closest to 0. 
				"""

				if abs.(eigSys.values[end]) > abs.(eigSys.values[1])
					# print("If statement 1.1 \n")
					if_val = 1
					theta = 1/eigSys.values[end]
					hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, end])
					hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, end])
				else
					# print("If statement 1.2 \n")
					if_val = 2
					theta = 1/eigSys.values[1]
					hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
					hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
				end	

				# Update residual vector
				resVec = (theta * hRitzSrc) .- hRitzTrg
		
				# Direction vector tolerance check 
				if norm(resVec) < tol_conv
					print("Restart algo converged off resVec tolerance \n")
					# print("Converged off tolerance \n")
					# print("norm(resVec) ", norm(resVec), "\n")
					return real(theta),nb_it_restart
					# println(real(theta))
				end
				# Eigenvalue tolerance check
				if abs((real(theta) - real(previous_eigval))/real(previous_eigval)) < tol_eigval
					if nb_it_eigval_conv == 5
						print("Restart algo converged off eigval tolerance \n")
						return real(theta),nb_it_restart
					end 
					nb_it_eigval_conv += 1
				end
				previous_eigval = theta
				nb_it_restart += 1
			end
		end


		# if it == restartDim
		# 	print("Didn't converge off tolerance for restart program. 
		# 		Atteined max set number of iterations \n")
		# 	return real(theta),nb_it_restart
		# end

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
		# for i in range(0,1,restartDim)
		for i = 1:restartDim
			# print("eigenvalues[1] ", eigenvalues[1], "\n")
			# print("eigenvalues[end] ", eigenvalues[end], "\n")
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

		# orthogonalize
		gramSchmidtHarm!(restart_trgBasis, restart_srcBasis, bCoeffs1, bCoeffs2, opt,
		restartDim, tol_MGS)

		"""
		Create kMat for restart using change of basis matrix: u_matrix
		# Create kMat for restart using scrcBasis and trgBasis
		"""
		
		restart_kMat = adjoint(u_matrix)*kMat[1 : innerLoopDim, 1 : innerLoopDim]*u_matrix
		# Let's check if kMat is Hermitian by calling the ishermitian
		# function on it
		# print("Test if restart_kMat is hermitian ", 
		# 	ishermitian(restart_kMat[1 : restartDim, 1 : restartDim]), "\n")	
		# (opt .+ adjoint(opt)) ./ 2

		# kMat is not Hermitian so let's make it Hermitian by adding its 
		# adjoint and dividing by 2
		restart_kMat = (restart_kMat.+adjoint(restart_kMat))./2
		# print("Test if restart_kMat is hermitian ", 
		# 	ishermitian(restart_kMat[1 : restartDim, 1 : restartDim]), "\n")	

		# Other way of calculating restart_kMat
		# for i = 1:restartDim
		# 	if i == 1
		# 		# representation of opt^{-1} in trgBasis
		# 		restart_kMat[1,1] = BLAS.dotc(vecDim, view(restart_trgBasis, :, 1), 1,
		# 		view(restart_srcBasis, :, 1), 1)
		# 	else # Essentially if i != 1 or i > 1 since i starts at 1 and can't be smaller than 1
		# 		# update inverse representation of opt^{-1} in trgBasis
		# 		restart_kMat[1 : i, i] = BLAS.gemv('C', view(restart_trgBasis, :, 1 : i),
		# 		view(restart_srcBasis, :, i))
		# 		# assuming opt^{-1} Hermitian matrix
		# 		restart_kMat[i, 1 : (i - 1)] = conj(restart_kMat[1 : (i-1), i])
		# 	end
		# 	# print("restart_kMat[1,1] ", restart_kMat[1,1], "\n")
		# end

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
	return real(theta),nb_it_restart
	
end


# perform Gram-Schmidt on target basis, adjusting source basis accordingly
function gramSchmidtHarm!(trgBasis, srcBasis,
	bCoeffs1::Vector{T}, bCoeffs2::Vector{T}, opt::Array{T}, n::Integer,
	tol::Float64) where T <: Number
# function gramSchmidtHarm!(trgBasis::Array{T}, srcBasis::Array{T},
# 	bCoeffs1::Vector{T}, bCoeffs2::Vector{T}, opt::Array{T}, n::Integer,
# 	tol::Float64) where T <: Number
	# dimension of vector space
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
function bicgstab_matrix(A, theta, hTrg, hSrc, prjC, b, tol_bicgstab)
	dim = size(A)[1]
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    # tol = 1e-4a
    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b
    rho_m1 = alpha = omega_m1 = 1
    # for k in 1:length(b)
    # We don't want a perfect solve, should fix this though
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
        vk = harmVec(dim, hTrg, hSrc, prjC,
        	A * pkPrj .- (theta .* pkPrj))
        # (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, pk)
        # Alpha calculation
        # Bottom term
        bottom_term = conj.(transpose(r0))*vk
        # Calculation
        alpha = rho_k / bottom_term
        h = xk_m1 + alpha.*pk
        # If h is accurate enough, then set xk=h and quantity
        # What does accurate enough mean?
        s = r_m1 - alpha.*vk
        bPrj = harmVec(dim, hTrg, hSrc, prjC, b)
        t = harmVec(dim, hTrg, hSrc, prjC,
        	A * bPrj .- (theta .* bPrj))
        # (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, s)
        # Omega_k calculation
        # Top term
        ts = conj.(transpose(t))*s
        # Bottom term
        tt = conj.(transpose(t))*t
        # Calculation
        omega_k = ts ./ tt
        xk_m1 = h + omega_k.*s
        r_old = r_m1
        r_m1 = s-omega_k.*t
        # print("conj.(transpose(r_m1))*r_m1  ", conj.(transpose(r_m1))*r_m1 , "\n")
        # if real((conj.(transpose(r_m1))*r_m1)[1]) < tol
        if norm(r_m1)-norm(r_old) < tol_bicgstab
            # print("bicgstab break \n")
            # print("real((conj.(transpose(r_m1))*r_m1)[1])",real((conj.(transpose(r_m1))*r_m1)[1]),"\n")
            # break
			return xk_m1
			# Number of iterations to reach a tolerances
        end
    end
	print("Didn't converge off bicgstab tolerance \n")
    return xk_m1
end

### testing
# opt = [8.0 + im*0.0  -3.0 + im*0.0  2.0 + im*0.0;
# 	-1.0 + im*0.0  3.0 + im*0.0  -1.0 + im*0.0;
# 	1.0 + im*0.0  -1.0 + im*0.0  4.0 + im*0.0]

# For RND tests 
# sz = 256
# sz = 50
# sz = 15
# opt = Array{ComplexF64}(undef,sz,sz)
# rand!(opt)

# Double check bicgstab
# Check norm of srcBasis and trgBasis

# For tests
# opt = [2.0 + im*0.0  -2.0 + im*0.0  0.0 + im*0.0;
# 	-1.0 + im*0.0  3.0 + im*0.0  -1.0 + im*0.0;
# 	0.0 + im*0.0  -1.0 + im*0.0  4.0 + im*0.0]
# opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# trueEigSys = eigen(opt)
# minEigPos = argmin(abs.(trueEigSys.values))
# julia_min_eigval = trueEigSys.values[minEigPos]

# dims = size(opt)

# bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# trgBasis = zeros(ComplexF64, dims[1], dims[2])
# srcBasis = zeros(ComplexF64, dims[1], dims[2])
# kMat = zeros(ComplexF64, dims[2], dims[2])


# innerLoopDim = 10
# restartDim = 2

# sz = 32
# opt = Array{ComplexF64}(undef,sz,sz)

operatorDim_vals = [50,100,150,200]
opt = Array{ComplexF64}(undef,operatorDim_vals[1],operatorDim_vals[1])
rand!(opt)
opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
trueEigSys = eigen(opt)
minEigPos = argmin(abs.(trueEigSys.values))
julia_min_eigval = trueEigSys.values[minEigPos]

dims = size(opt)
bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
kMat = zeros(ComplexF64, dims[2], dims[2])


innerLoopDim = 25
restartDim = 3
tol_MGS = 1.0e-12
tol_conv = 1.0e-6
tol_eigval = 1.0e-9
tol_bicgstab_vals = [1e-2,1e-3,1e-4,1e-5,1e-6]

bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
bicgstab_tol_nb_it_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
bicgstab_tol_nb_it_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))

solver_method = "bicgstab"
# solver_method = "direct"

"""
1. 
"""
tol_bicgstab= tol_bicgstab_vals[1]
first_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
first_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
first_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
first_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
first_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		first_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	first_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	first_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	first_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	first_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("first_bicgstab_tol_eigvals_basic ", first_bicgstab_tol_eigvals_basic, "\n")
print("first_bicgstab_tol_nb_it_basic ", first_bicgstab_tol_nb_it_basic, "\n")
print("first_bicgstab_tol_eigvals_restart ", first_bicgstab_tol_eigvals_restart, "\n")
print("first_bicgstab_tol_nb_it_restart  ", first_bicgstab_tol_nb_it_restart , "\n")
print("first_bicgstab_julia_min_eigvals ", first_bicgstab_julia_min_eigvals, "\n")



"""
2. 
"""
tol_bicgstab= tol_bicgstab_vals[2]
second_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
second_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
second_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
second_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
second_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		second_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	second_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	second_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	second_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	second_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("second_bicgstab_tol_eigvals_basic ", second_bicgstab_tol_eigvals_basic, "\n")
print("second_bicgstab_tol_nb_it_basic ", second_bicgstab_tol_nb_it_basic, "\n")
print("second_bicgstab_tol_eigvals_restart ", second_bicgstab_tol_eigvals_restart, "\n")
print("second_bicgstab_tol_nb_it_restart  ", second_bicgstab_tol_nb_it_restart , "\n")
print("second_bicgstab_julia_min_eigvals ", second_bicgstab_julia_min_eigvals, "\n")


"""
3. 
"""
tol_bicgstab= tol_bicgstab_vals[3]
third_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
third_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
third_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
third_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
third_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		third_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	third_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	third_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	third_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	third_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("third_bicgstab_tol_eigvals_basic ", third_bicgstab_tol_eigvals_basic, "\n")
print("third_bicgstab_tol_nb_it_basic ", third_bicgstab_tol_nb_it_basic, "\n")
print("third_bicgstab_tol_eigvals_restart ", third_bicgstab_tol_eigvals_restart, "\n")
print("third_bicgstab_tol_nb_it_restart  ", third_bicgstab_tol_nb_it_restart , "\n")
print("third_bicgstab_julia_min_eigvals ", third_bicgstab_julia_min_eigvals, "\n")


"""
4. 
"""
tol_bicgstab= tol_bicgstab_vals[4]
fourth_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
fourth_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
fourth_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
fourth_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
fourth_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		fourth_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	fourth_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	fourth_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	fourth_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	fourth_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("fourth_bicgstab_tol_eigvals_basic ", fourth_bicgstab_tol_eigvals_basic, "\n")
print("fourth_bicgstab_tol_nb_it_basic ", fourth_bicgstab_tol_nb_it_basic, "\n")
print("fourth_bicgstab_tol_eigvals_restart ", fourth_bicgstab_tol_eigvals_restart, "\n")
print("fourth_bicgstab_tol_nb_it_restart  ", fourth_bicgstab_tol_nb_it_restart , "\n")
print("fourth_bicgstab_julia_min_eigvals ", fourth_bicgstab_julia_min_eigvals, "\n")


"""
5. 
"""
tol_bicgstab= tol_bicgstab_vals[5]
fifth_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
fifth_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
fifth_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
fifth_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
fifth_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		fifth_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	fifth_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	fifth_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	fifth_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	fifth_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("fifth_bicgstab_tol_eigvals_basic ", fifth_bicgstab_tol_eigvals_basic, "\n")
print("fifth_bicgstab_tol_nb_it_basic ", fifth_bicgstab_tol_nb_it_basic, "\n")
print("fifth_bicgstab_tol_eigvals_restart ", fifth_bicgstab_tol_eigvals_restart, "\n")
print("fifth_bicgstab_tol_nb_it_restart  ", fifth_bicgstab_tol_nb_it_restart , "\n")
print("fifth_bicgstab_julia_min_eigvals ", fifth_bicgstab_julia_min_eigvals, "\n")

"""
6. Direct solve
"""
innerLoopDim = 25
restartDim = 3
tol_bicgstab = 1e-9
operatorDim_vals = [50,100,150,200]
# opt = Array{ComplexF64}(undef,50,50)
# rand!(opt)
# opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# opt += 8*I # To avoid singularities
# trueEigSys = eigen(opt)
# minEigPos = argmin(abs.(trueEigSys.values))
# julia_min_eigval = trueEigSys.values[minEigPos]
# direct_solve_julia_min_eigval = julia_min_eigval

dims = size(opt)
bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
kMat = zeros(ComplexF64, dims[2], dims[2])

solver_method = "direct"  
# solver_method = "bicgstab"
tol_bicgstab = tol_bicgstab_vals[1]
six_direct_solve_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
six_direct_solve_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
six_direct_solve_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
six_direct_solve_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
six_direct_solve_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
# print("Start of basic program \n")
# eigval_basic,nb_it_basic  = 
# 	jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 		dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 			tol_bicgstab,solver_method)
# print("Start of restart program \n")
# eigval_restart,nb_it_restart  = 
# 	jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 		dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 			tol_bicgstab,solver_method)
# print("eigval_basic ", eigval_basic, "\n")
# print("nb_it_basic ", nb_it_basic, "\n")
# print("eigval_restart ", eigval_restart, "\n")
# print("nb_it_restart ", nb_it_restart, "\n")
# print("direct_solve_julia_min_eigval ", direct_solve_julia_min_eigval, "\n")


@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		# opt += 8*I # To avoid singularities
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		six_direct_solve_julia_min_eigvals[index] = julia_min_eigval
		print("trueEigSys.values ", trueEigSys.values,"\n")

		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	six_direct_solve_eigvals_basic[index] = sum_eigval_basic/loop_nb
	six_direct_solve_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	six_direct_solve_eigvals_restart[index] = sum_eigval_restart/loop_nb
	six_direct_solve_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("six_direct_solve_eigvals_basic ", six_direct_solve_eigvals_basic, "\n")
print("six_direct_solve_nb_it_basic ", six_direct_solve_nb_it_basic, "\n")
print("six_direct_solve_eigvals_restart ", six_direct_solve_eigvals_restart, "\n")
print("six_direct_solve_nb_it_restart  ", six_direct_solve_nb_it_restart , "\n")
print("six_direct_solve_julia_min_eigvals ", six_direct_solve_julia_min_eigvals, "\n")


innerLoopDim = 25
restartDim = 10
tol_MGS = 1.0e-12
tol_conv = 1.0e-6
tol_eigval = 1.0e-9
tol_bicgstab_vals = [1e-2,1e-3,1e-4,1e-5,1e-6]


bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
bicgstab_tol_nb_it_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
bicgstab_tol_nb_it_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))

solver_method = "bicgstab"
# solver_method = "direct"

"""
7. Same as 1-6 but with a different value for restartDim 
"""
tol_bicgstab= tol_bicgstab_vals[1]
seven_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
seven_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
seven_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
seven_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
seven_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		seven_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	seven_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	seven_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	seven_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	seven_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("seven_bicgstab_tol_eigvals_basic ", seven_bicgstab_tol_eigvals_basic, "\n")
print("seven_bicgstab_tol_nb_it_basic ", seven_bicgstab_tol_nb_it_basic, "\n")
print("seven_bicgstab_tol_eigvals_restart ", seven_bicgstab_tol_eigvals_restart, "\n")
print("seven_bicgstab_tol_nb_it_restart  ", seven_bicgstab_tol_nb_it_restart , "\n")
print("seven_bicgstab_julia_min_eigvals ", seven_bicgstab_julia_min_eigvals, "\n")



"""
8. 
"""
tol_bicgstab= tol_bicgstab_vals[2]
eight_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
eight_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
eight_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
eight_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
eight_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		eight_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	eight_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	eight_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	eight_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	eight_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("eight_bicgstab_tol_eigvals_basic ", eight_bicgstab_tol_eigvals_basic, "\n")
print("eight_bicgstab_tol_nb_it_basic ", eight_bicgstab_tol_nb_it_basic, "\n")
print("eight_bicgstab_tol_eigvals_restart ", eight_bicgstab_tol_eigvals_restart, "\n")
print("eight_bicgstab_tol_nb_it_restart  ", eight_bicgstab_tol_nb_it_restart , "\n")
print("eight_bicgstab_julia_min_eigvals ", eight_bicgstab_julia_min_eigvals, "\n")


"""
9.
"""
tol_bicgstab= tol_bicgstab_vals[3]
nine_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
nine_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
nine_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
nine_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
nine_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		nine_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	nine_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	nine_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	nine_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	nine_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("nine_bicgstab_tol_eigvals_basic ", nine_bicgstab_tol_eigvals_basic, "\n")
print("nine_bicgstab_tol_nb_it_basic ", nine_bicgstab_tol_nb_it_basic, "\n")
print("nine_bicgstab_tol_eigvals_restart ", nine_bicgstab_tol_eigvals_restart, "\n")
print("nine_bicgstab_tol_nb_it_restart  ", nine_bicgstab_tol_nb_it_restart , "\n")
print("nine_bicgstab_julia_min_eigvals ", nine_bicgstab_julia_min_eigvals, "\n")


"""
10.
"""
tol_bicgstab= tol_bicgstab_vals[4]
ten_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
ten_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
ten_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
ten_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
ten_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		fourth_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	ten_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	ten_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	ten_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	ten_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("ten_bicgstab_tol_eigvals_basic ", ten_bicgstab_tol_eigvals_basic, "\n")
print("ten_bicgstab_tol_nb_it_basic ", ten_bicgstab_tol_nb_it_basic, "\n")
print("ten_bicgstab_tol_eigvals_restart ", ten_bicgstab_tol_eigvals_restart, "\n")
print("ten_bicgstab_tol_nb_it_restart  ", ten_bicgstab_tol_nb_it_restart , "\n")
print("ten_bicgstab_julia_min_eigvals ", ten_bicgstab_julia_min_eigvals, "\n")


"""
11. 
"""
tol_bicgstab= tol_bicgstab_vals[5]
eleven_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
eleven_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
eleven_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
eleven_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
eleven_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		eleven_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	eleven_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	eleven_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	eleven_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	eleven_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("eleven_bicgstab_tol_eigvals_basic ", eleven_bicgstab_tol_eigvals_basic, "\n")
print("eleven_bicgstab_tol_nb_it_basic ", eleven_bicgstab_tol_nb_it_basic, "\n")
print("eleven_bicgstab_tol_eigvals_restart ", eleven_bicgstab_tol_eigvals_restart, "\n")
print("eleven_bicgstab_tol_nb_it_restart  ", eleven_bicgstab_tol_nb_it_restart , "\n")
print("eleven_bicgstab_julia_min_eigvals ", eleven_bicgstab_julia_min_eigvals, "\n")


"""
12. Second direct solve. 
"""
innerLoopDim = 25
restartDim = 10
tol_bicgstab = 1e-9
operatorDim_vals = [50,100,150,200]
# opt = Array{ComplexF64}(undef,50,50)
# rand!(opt)
# opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# opt += 8*I # To avoid singularities
# trueEigSys = eigen(opt)
# minEigPos = argmin(abs.(trueEigSys.values))
# julia_min_eigval = trueEigSys.values[minEigPos]
# direct_solve_julia_min_eigval = julia_min_eigval

dims = size(opt)
bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
kMat = zeros(ComplexF64, dims[2], dims[2])

solver_method = "direct"  
# solver_method = "bicgstab"
tol_bicgstab = tol_bicgstab_vals[1]
twelve_direct_solve_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
twelve_direct_solve_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
twelve_direct_solve_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
twelve_direct_solve_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
twelve_direct_solve_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10

@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		# opt += 8*I # To avoid singularities
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		twelve_direct_solve_julia_min_eigvals[index] = julia_min_eigval
		print("trueEigSys.values ", trueEigSys.values,"\n")

		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	twelve_direct_solve_eigvals_basic[index] = sum_eigval_basic/loop_nb
	twelve_direct_solve_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	twelve_direct_solve_eigvals_restart[index] = sum_eigval_restart/loop_nb
	twelve_direct_solve_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end

innerLoopDim = 25
restartDim = 20
tol_MGS = 1.0e-12
tol_conv = 1.0e-6
tol_eigval = 1.0e-9
tol_bicgstab_vals = [1e-2,1e-3,1e-4,1e-5,1e-6]


bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
bicgstab_tol_nb_it_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
bicgstab_tol_nb_it_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))

solver_method = "bicgstab"
# solver_method = "direct"

"""
13. Same as 1-6 but with a different value for restartDim 
"""
tol_bicgstab= tol_bicgstab_vals[1]
thirteen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
thirteen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
thirteen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
thirteen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
thirteen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		thirteen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	thirteen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	thirteen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	thirteen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	thirteen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("thirteen_bicgstab_tol_eigvals_basic ", thirteen_bicgstab_tol_eigvals_basic, "\n")
print("thirteen_bicgstab_tol_nb_it_basic ", thirteen_bicgstab_tol_nb_it_basic, "\n")
print("thirteen_bicgstab_tol_eigvals_restart ", thirteen_bicgstab_tol_eigvals_restart, "\n")
print("thirteen_bicgstab_tol_nb_it_restart  ", thirteen_bicgstab_tol_nb_it_restart , "\n")
print("thirteen_bicgstab_julia_min_eigvals ", thirteen_bicgstab_julia_min_eigvals, "\n")



"""
14. 
"""
tol_bicgstab= tol_bicgstab_vals[2]
fourteen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
fourteen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
fourteen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
fourteen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
fourteen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		fourteen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	fourteen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	fourteen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	fourteen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	fourteen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("fourteen_bicgstab_tol_eigvals_basic ", fourteen_bicgstab_tol_eigvals_basic, "\n")
print("fourteen_bicgstab_tol_nb_it_basic ", fourteen_bicgstab_tol_nb_it_basic, "\n")
print("fourteen_bicgstab_tol_eigvals_restart ", fourteen_bicgstab_tol_eigvals_restart, "\n")
print("fourteen_bicgstab_tol_nb_it_restart  ", fourteen_bicgstab_tol_nb_it_restart , "\n")
print("fourteen_bicgstab_julia_min_eigvals ", fourteen_bicgstab_julia_min_eigvals, "\n")


"""
15.
"""
tol_bicgstab= tol_bicgstab_vals[3]
fifteen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
fifteen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
fifteen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
fifteen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
fifteen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		fifteen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	fifteen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	fifteen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	fifteen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	fifteen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("fifteen_bicgstab_tol_eigvals_basic ", fifteen_bicgstab_tol_eigvals_basic, "\n")
print("fifteen_bicgstab_tol_nb_it_basic ", fifteen_bicgstab_tol_nb_it_basic, "\n")
print("fifteen_bicgstab_tol_eigvals_restart ", fifteen_bicgstab_tol_eigvals_restart, "\n")
print("fifteen_bicgstab_tol_nb_it_restart  ", fifteen_bicgstab_tol_nb_it_restart , "\n")
print("fifteen_bicgstab_julia_min_eigvals ", fifteen_bicgstab_julia_min_eigvals, "\n")


"""
16.
"""
tol_bicgstab= tol_bicgstab_vals[4]
sixteen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
sixteen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
sixteen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
sixteen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
sixteen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		sixteen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	sixteen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	sixteen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	sixteen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	sixteen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("sixteen_bicgstab_tol_eigvals_basic ", sixteen_bicgstab_tol_eigvals_basic, "\n")
print("sixteen_bicgstab_tol_nb_it_basic ", sixteen_bicgstab_tol_nb_it_basic, "\n")
print("sixteen_bicgstab_tol_eigvals_restart ", sixteen_bicgstab_tol_eigvals_restart, "\n")
print("sixteen_bicgstab_tol_nb_it_restart  ", sixteen_bicgstab_tol_nb_it_restart , "\n")
print("sixteen_bicgstab_julia_min_eigvals ", sixteen_bicgstab_julia_min_eigvals, "\n")


"""
17. 
"""
tol_bicgstab= tol_bicgstab_vals[5]
seventeen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
seventeen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
seventeen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
seventeen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
seventeen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10
@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		seventeen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	seventeen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
	seventeen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	seventeen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
	seventeen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end
print("seventeen_bicgstab_tol_eigvals_basic ", seventeen_bicgstab_tol_eigvals_basic, "\n")
print("seventeen_bicgstab_tol_nb_it_basic ", seventeen_bicgstab_tol_nb_it_basic, "\n")
print("seventeen_bicgstab_tol_eigvals_restart ", seventeen_bicgstab_tol_eigvals_restart, "\n")
print("seventeen_bicgstab_tol_nb_it_restart  ", seventeen_bicgstab_tol_nb_it_restart , "\n")
print("seventeen_bicgstab_julia_min_eigvals ", seventeen_bicgstab_julia_min_eigvals, "\n")


"""
18. Second direct solve. 
"""
innerLoopDim = 25
restartDim = 20
tol_bicgstab = 1e-9
operatorDim_vals = [50,100,150,200]
# opt = Array{ComplexF64}(undef,50,50)
# rand!(opt)
# opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# opt += 8*I # To avoid singularities
# trueEigSys = eigen(opt)
# minEigPos = argmin(abs.(trueEigSys.values))
# julia_min_eigval = trueEigSys.values[minEigPos]
# direct_solve_julia_min_eigval = julia_min_eigval

dims = size(opt)
bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
kMat = zeros(ComplexF64, dims[2], dims[2])

solver_method = "direct"  
# solver_method = "bicgstab"
tol_bicgstab = tol_bicgstab_vals[1]
eighteen_direct_solve_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
eighteen_direct_solve_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
eighteen_direct_solve_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
eighteen_direct_solve_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
eighteen_direct_solve_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
loop_nb = 10

@threads for index = 1:length(operatorDim_vals)
	sum_eigval_basic = 0.0
	sum_nb_it_basic = 0.0
	sum_eigval_restart = 0.0
	sum_nb_it_restart = 0.0
	for i = 1:loop_nb
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)
		# opt += 8*I # To avoid singularities
		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		julia_min_eigval = trueEigSys.values[minEigPos]
		eighteen_direct_solve_julia_min_eigvals[index] = julia_min_eigval
		print("trueEigSys.values ", trueEigSys.values,"\n")

		
		dims = size(opt)
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])

		print("Start of basic program \n")
		eigval_basic,nb_it_basic  = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,solver_method)
		sum_eigval_basic += eigval_basic 
		sum_nb_it_basic += nb_it_basic 
		sum_eigval_restart += eigval_restart
		sum_nb_it_restart += nb_it_restart 
	end
	eighteen_direct_solve_eigvals_basic[index] = sum_eigval_basic/loop_nb
	eighteen_direct_solve_nb_it_basic[index] = sum_nb_it_basic/loop_nb
	eighteen_direct_solve_eigvals_restart[index] = sum_eigval_restart/loop_nb
	eighteen_direct_solve_nb_it_restart[index] = sum_nb_it_restart/loop_nb
end

print("eighteen_direct_solve_eigvals_basic ", eighteen_direct_solve_eigvals_basic, "\n")
print("eighteen_direct_solve_nb_it_basic ", eighteen_direct_solve_nb_it_basic, "\n")
print("eighteen_direct_solve_eigvals_restart ", eighteen_direct_solve_eigvals_restart, "\n")
print("eighteen_direct_solve_nb_it_restart  ", eighteen_direct_solve_nb_it_restart , "\n")
print("eighteen_direct_solve_julia_min_eigvals ", eighteen_direct_solve_julia_min_eigvals, "\n")

print("twelve_direct_solve_eigvals_basic ", twelve_direct_solve_eigvals_basic, "\n")
print("twelve_direct_solve_nb_it_basic ", twelve_direct_solve_nb_it_basic, "\n")
print("twelve_direct_solve_eigvals_restart ", twelve_direct_solve_eigvals_restart, "\n")
print("twelve_direct_solve_nb_it_restart  ", twelve_direct_solve_nb_it_restart , "\n")
print("twelve_direct_solve_julia_min_eigvals ", twelve_direct_solve_julia_min_eigvals, "\n")


print("six_direct_solve_eigvals_basic ", six_direct_solve_eigvals_basic, "\n")
print("six_direct_solve_nb_it_basic ", six_direct_solve_nb_it_basic, "\n")
print("six_direct_solve_eigvals_restart ", six_direct_solve_eigvals_restart, "\n")
print("six_direct_solve_nb_it_restart  ", six_direct_solve_nb_it_restart , "\n")
print("six_direct_solve_julia_min_eigvals ", six_direct_solve_julia_min_eigvals, "\n")



# innerLoopDim = 25
# restartDim = 3
# tol_MGS = 1.0e-12
# tol_conv_vals = [1e-2,1e-3,1e-4,1e-5,1e-6]
# tol_eigval = 1.0e-9
# tol_bicgstab = 1.0e-6

# bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_nb_it_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_nb_it_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))

# solver_method = "bicgstab"
# # solver_method = "direct"

# """
# 13. Same as 1-6 but with different tol_conv values instead of bicgstab_tol 
# 	values
# """
# tol_conv = tol_conv_vals[1]
# thirteen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# thirteen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# thirteen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# thirteen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# thirteen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		thirteen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	thirteen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	thirteen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	thirteen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	thirteen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("thirteen_bicgstab_tol_eigvals_basic ", thirteen_bicgstab_tol_eigvals_basic, "\n")
# print("thirteen_bicgstab_tol_nb_it_basic ", thirteen_bicgstab_tol_nb_it_basic, "\n")
# print("thirteen_bicgstab_tol_eigvals_restart ", thirteen_bicgstab_tol_eigvals_restart, "\n")
# print("thirteen_bicgstab_tol_nb_it_restart  ", thirteen_bicgstab_tol_nb_it_restart , "\n")
# print("thirteen_bicgstab_julia_min_eigvals ", thirteen_bicgstab_julia_min_eigvals, "\n")



# """
# 14. 
# """
# tol_conv = tol_conv_vals[2]
# fourteen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# fourteen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# fourteen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# fourteen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# fourteen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		fourteen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	fourteen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	fourteen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	fourteen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	fourteen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("fourteen_bicgstab_tol_eigvals_basic ", fourteen_bicgstab_tol_eigvals_basic, "\n")
# print("fourteen_bicgstab_tol_nb_it_basic ", fourteen_bicgstab_tol_nb_it_basic, "\n")
# print("fourteen_bicgstab_tol_eigvals_restart ", fourteen_bicgstab_tol_eigvals_restart, "\n")
# print("fourteen_bicgstab_tol_nb_it_restart  ", fourteen_bicgstab_tol_nb_it_restart , "\n")
# print("fourteen_bicgstab_julia_min_eigvals ", fourteen_bicgstab_julia_min_eigvals, "\n")


# """
# 15.
# """
# tol_conv = tol_conv_vals[3]
# fifteen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# fifteen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# fifteen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# fifteen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# fifteen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		fifteen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	fifteen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	fifteen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	fifteen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	fifteen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("fifteen_bicgstab_tol_eigvals_basic ", fifteen_bicgstab_tol_eigvals_basic, "\n")
# print("fifteen_bicgstab_tol_nb_it_basic ", fifteen_bicgstab_tol_nb_it_basic, "\n")
# print("fifteen_bicgstab_tol_eigvals_restart ", fifteen_bicgstab_tol_eigvals_restart, "\n")
# print("fifteen_bicgstab_tol_nb_it_restart  ", fifteen_bicgstab_tol_nb_it_restart , "\n")
# print("fifteen_bicgstab_julia_min_eigvals ", fifteen_bicgstab_julia_min_eigvals, "\n")


# """
# 16.
# """
# tol_conv = tol_conv_vals[4]
# sixteen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# sixteen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# sixteen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# sixteen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# sixteen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		sixteen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	sixteen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	sixteen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	sixteen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	sixteen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("sixteen_bicgstab_tol_eigvals_basic ", sixteen_bicgstab_tol_eigvals_basic, "\n")
# print("sixteen_bicgstab_tol_nb_it_basic ", sixteen_bicgstab_tol_nb_it_basic, "\n")
# print("sixteen_bicgstab_tol_eigvals_restart ", sixteen_bicgstab_tol_eigvals_restart, "\n")
# print("sixteen_bicgstab_tol_nb_it_restart  ", sixteen_bicgstab_tol_nb_it_restart , "\n")
# print("sixteen_bicgstab_julia_min_eigvals ", sixteen_bicgstab_julia_min_eigvals, "\n")


# """
# 17. 
# """
# tol_conv = tol_conv_vals[5]
# seventeen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# seventeen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# seventeen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# seventeen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# seventeen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		seventeen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	seventeen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	seventeen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	seventeen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	seventeen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("seventeen_bicgstab_tol_eigvals_basic ", seventeen_bicgstab_tol_eigvals_basic, "\n")
# print("seventeen_bicgstab_tol_nb_it_basic ", seventeen_bicgstab_tol_nb_it_basic, "\n")
# print("seventeen_bicgstab_tol_eigvals_restart ", seventeen_bicgstab_tol_eigvals_restart, "\n")
# print("seventeen_bicgstab_tol_nb_it_restart  ", seventeen_bicgstab_tol_nb_it_restart , "\n")
# print("seventeen_bicgstab_julia_min_eigvals ", seventeen_bicgstab_julia_min_eigvals, "\n")

# innerLoopDim = 25
# restartDim = 3
# tol_MGS = 1.0e-12
# tol_conv = 1e-6
# tol_eigval_vals = [1e-4,1e-5,1e-7,1e-9]
# tol_bicgstab = 1.0e-6

# bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_nb_it_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_nb_it_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))

# solver_method = "bicgstab"
# # solver_method = "direct"

# """
# 18. Same as 1-6 but with different tol_eigval values instead of bicgstab_tol 
# 	values
# """
# tol_eigval = tol_eigval_vals[1]
# eighteen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# eighteen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# eighteen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# eighteen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# eighteen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		eighteen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	eighteen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	eighteen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	eighteen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	eighteen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("eighteen_bicgstab_tol_eigvals_basic ", eighteen_bicgstab_tol_eigvals_basic, "\n")
# print("eighteen_bicgstab_tol_nb_it_basic ", eighteen_bicgstab_tol_nb_it_basic, "\n")
# print("eighteen_bicgstab_tol_eigvals_restart ", eighteen_bicgstab_tol_eigvals_restart, "\n")
# print("eighteen_bicgstab_tol_nb_it_restart  ", eighteen_bicgstab_tol_nb_it_restart , "\n")
# print("eighteen_bicgstab_julia_min_eigvals ", eighteen_bicgstab_julia_min_eigvals, "\n")



# """
# 19. 
# """
# tol_eigval = tol_eigval_vals[2]
# nineteen_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# nineteen_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# nineteen_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# nineteen_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# nineteen_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		nineteen_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	nineteen_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	nineteen_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	nineteen_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	nineteen_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("nineteen_bicgstab_tol_eigvals_basic ", nineteen_bicgstab_tol_eigvals_basic, "\n")
# print("nineteen_bicgstab_tol_nb_it_basic ", nineteen_bicgstab_tol_nb_it_basic, "\n")
# print("nineteen_bicgstab_tol_eigvals_restart ", nineteen_bicgstab_tol_eigvals_restart, "\n")
# print("nineteen_bicgstab_tol_nb_it_restart  ", nineteen_bicgstab_tol_nb_it_restart , "\n")
# print("nineteen_bicgstab_julia_min_eigvals ", nineteen_bicgstab_julia_min_eigvals, "\n")


# """
# 20.
# """
# tol_eigval = tol_eigval_vals[3]
# twenty_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twenty_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twenty_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twenty_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twenty_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		twenty_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	twenty_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	twenty_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	twenty_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	twenty_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("twenty_bicgstab_tol_eigvals_basic ", twenty_bicgstab_tol_eigvals_basic, "\n")
# print("twenty_bicgstab_tol_nb_it_basic ", twenty_bicgstab_tol_nb_it_basic, "\n")
# print("twenty_bicgstab_tol_eigvals_restart ", twenty_bicgstab_tol_eigvals_restart, "\n")
# print("twenty_bicgstab_tol_nb_it_restart  ", twenty_bicgstab_tol_nb_it_restart , "\n")
# print("twenty_bicgstab_julia_min_eigvals ", twenty_bicgstab_julia_min_eigvals, "\n")


# """
# 21.
# """
# tol_eigval = tol_eigval_vals[4]
# twentyone_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentyone_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentyone_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentyone_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentyone_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		twentyone_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	twentyone_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	twentyone_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	twentyone_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	twentyone_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("twentyone_bicgstab_tol_eigvals_basic ", twentyone_bicgstab_tol_eigvals_basic, "\n")
# print("twentyone_bicgstab_tol_nb_it_basic ", twentyone_bicgstab_tol_nb_it_basic, "\n")
# print("twentyone_bicgstab_tol_eigvals_restart ", twentyone_bicgstab_tol_eigvals_restart, "\n")
# print("twentyone_bicgstab_tol_nb_it_restart  ", twentyone_bicgstab_tol_nb_it_restart , "\n")
# print("twentyone_bicgstab_julia_min_eigvals ", twentyone_bicgstab_julia_min_eigvals, "\n")


# innerLoopDim = 25
# restartDim = 20
# tol_MGS = 1.0e-12
# tol_conv_vals = [1e-2,1e-3,1e-4,1e-5,1e-6]
# tol_eigval = 1.0e-9
# tol_bicgstab = 1.0e-6

# bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_nb_it_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_nb_it_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))

# solver_method = "bicgstab"
# # solver_method = "direct"

# """
# 22. Same as 1-6 but with different tol_conv values instead of bicgstab_tol 
# 	values
# """
# tol_conv = tol_conv_vals[1]
# twentytwo_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentytwo_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentytwo_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentytwo_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentytwo_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		twentytwo_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	twentytwo_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	twentytwo_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	twentytwo_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	twentytwo_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("twentytwo_bicgstab_tol_eigvals_basic ", twentytwo_bicgstab_tol_eigvals_basic, "\n")
# print("twentytwo_bicgstab_tol_nb_it_basic ", twentytwo_bicgstab_tol_nb_it_basic, "\n")
# print("twentytwo_bicgstab_tol_eigvals_restart ", twentytwo_bicgstab_tol_eigvals_restart, "\n")
# print("twentytwo_bicgstab_tol_nb_it_restart  ", twentytwo_bicgstab_tol_nb_it_restart , "\n")
# print("twentytwo_bicgstab_julia_min_eigvals ", twentytwo_bicgstab_julia_min_eigvals, "\n")



# """
# 23. 
# """
# tol_conv = tol_conv_vals[2]
# twentythree_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentythree_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentythree_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentythree_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentythree_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		twentythree_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	twentythree_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	twentythree_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	twentythree_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	twentythree_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("twentythree_bicgstab_tol_eigvals_basic ", twentythree_bicgstab_tol_eigvals_basic, "\n")
# print("twentythree_bicgstab_tol_nb_it_basic ", twentythree_bicgstab_tol_nb_it_basic, "\n")
# print("twentythree_bicgstab_tol_eigvals_restart ", twentythree_bicgstab_tol_eigvals_restart, "\n")
# print("twentythree_bicgstab_tol_nb_it_restart  ", twentythree_bicgstab_tol_nb_it_restart , "\n")
# print("twentythree_bicgstab_julia_min_eigvals ", twentythree_bicgstab_julia_min_eigvals, "\n")


# """
# 24.
# """
# tol_conv = tol_conv_vals[3]
# twentyfour_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentyfour_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentyfour_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentyfour_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentyfour_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		twentyfour_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	twentyfour_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	twentyfour_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	twentyfour_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	twentyfour_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("twentyfour_bicgstab_tol_eigvals_basic ", twentyfour_bicgstab_tol_eigvals_basic, "\n")
# print("twentyfour_bicgstab_tol_nb_it_basic ", twentyfour_bicgstab_tol_nb_it_basic, "\n")
# print("twentyfour_bicgstab_tol_eigvals_restart ", twentyfour_bicgstab_tol_eigvals_restart, "\n")
# print("twentyfour_bicgstab_tol_nb_it_restart  ", twentyfour_bicgstab_tol_nb_it_restart , "\n")
# print("twentyfour_bicgstab_julia_min_eigvals ", twentyfour_bicgstab_julia_min_eigvals, "\n")


# """
# 25.
# """
# tol_conv = tol_conv_vals[4]
# twentyfive_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentyfive_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentyfive_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentyfive_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentyfive_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		twentyfive_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	twentyfive_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	twentyfive_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	twentyfive_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	twentyfive_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("twentyfive_bicgstab_tol_eigvals_basic ", twentyfive_bicgstab_tol_eigvals_basic, "\n")
# print("twentyfive_bicgstab_tol_nb_it_basic ", twentyfive_bicgstab_tol_nb_it_basic, "\n")
# print("twentyfive_bicgstab_tol_eigvals_restart ", twentyfive_bicgstab_tol_eigvals_restart, "\n")
# print("twentyfive_bicgstab_tol_nb_it_restart  ", twentyfive_bicgstab_tol_nb_it_restart , "\n")
# print("twentyfive_bicgstab_julia_min_eigvals ", twentyfive_bicgstab_julia_min_eigvals, "\n")


# """
# 26. 
# """
# tol_conv = tol_conv_vals[5]
# twentysix_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentysix_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentysix_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentysix_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentysix_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		twentysix_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	twentysix_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	twentysix_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	twentysix_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	twentysix_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("twentysix_bicgstab_tol_eigvals_basic ", twentysix_bicgstab_tol_eigvals_basic, "\n")
# print("twentysix_bicgstab_tol_nb_it_basic ", twentysix_bicgstab_tol_nb_it_basic, "\n")
# print("twentysix_bicgstab_tol_eigvals_restart ", twentysix_bicgstab_tol_eigvals_restart, "\n")
# print("twentysix_bicgstab_tol_nb_it_restart  ", twentysix_bicgstab_tol_nb_it_restart , "\n")
# print("twentysix_bicgstab_julia_min_eigvals ", twentysix_bicgstab_julia_min_eigvals, "\n")


# innerLoopDim = 25
# restartDim = 10
# tol_MGS = 1.0e-12
# tol_conv_vals = [1e-2,1e-3,1e-4,1e-5,1e-6]
# tol_eigval = 1.0e-9
# tol_bicgstab = 1.0e-6

# bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_nb_it_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
# bicgstab_tol_nb_it_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))

# solver_method = "bicgstab"
# # solver_method = "direct"

# """
# 27. Same as 1-6 but with different tol_conv values instead of bicgstab_tol 
# 	values
# """
# tol_conv = tol_conv_vals[1]
# twentyseven_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentyseven_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentyseven_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentyseven_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentyseven_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		twentyseven_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	twentyseven_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	twentyseven_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	twentyseven_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	twentyseven_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("twentyseven_bicgstab_tol_eigvals_basic ", twentyseven_bicgstab_tol_eigvals_basic, "\n")
# print("twentyseven_bicgstab_tol_nb_it_basic ", twentyseven_bicgstab_tol_nb_it_basic, "\n")
# print("twentyseven_bicgstab_tol_eigvals_restart ", twentyseven_bicgstab_tol_eigvals_restart, "\n")
# print("twentyseven_bicgstab_tol_nb_it_restart  ", twentyseven_bicgstab_tol_nb_it_restart , "\n")
# print("twentyseven_bicgstab_julia_min_eigvals ", twentyseven_bicgstab_julia_min_eigvals, "\n")



# """
# 28. 
# """
# tol_conv = tol_conv_vals[2]
# twentyeight_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentyeight_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentyeight_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentyeight_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentyeight_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		twentyeight_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	twentyeight_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	twentyeight_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	twentyeight_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	twentyeight_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("twentyeight_bicgstab_tol_eigvals_basic ", twentyeight_bicgstab_tol_eigvals_basic, "\n")
# print("twentyeight_bicgstab_tol_nb_it_basic ", twentyeight_bicgstab_tol_nb_it_basic, "\n")
# print("twentyeight_bicgstab_tol_eigvals_restart ", twentyeight_bicgstab_tol_eigvals_restart, "\n")
# print("twentyeight_bicgstab_tol_nb_it_restart  ", twentyeight_bicgstab_tol_nb_it_restart , "\n")
# print("twentyeight_bicgstab_julia_min_eigvals ", twentyeight_bicgstab_julia_min_eigvals, "\n")


# """
# 29.
# """
# tol_conv = tol_conv_vals[3]
# twentynine_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentynine_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# twentynine_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentynine_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# twentynine_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		twentynine_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	twentynine_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	twentynine_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	twentynine_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	twentynine_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("twentynine_bicgstab_tol_eigvals_basic ", twentynine_bicgstab_tol_eigvals_basic, "\n")
# print("twentynine_bicgstab_tol_nb_it_basic ", twentynine_bicgstab_tol_nb_it_basic, "\n")
# print("twentynine_bicgstab_tol_eigvals_restart ", twentynine_bicgstab_tol_eigvals_restart, "\n")
# print("twentynine_bicgstab_tol_nb_it_restart  ", twentynine_bicgstab_tol_nb_it_restart , "\n")
# print("twentynine_bicgstab_julia_min_eigvals ", twentynine_bicgstab_julia_min_eigvals, "\n")


# """
# 30.
# """
# tol_conv = tol_conv_vals[4]
# thirty_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# thirty_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# thirty_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# thirty_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# thirty_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		thirty_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	thirty_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	thirty_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	thirty_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	thirty_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("thirty_bicgstab_tol_eigvals_basic ", thirty_bicgstab_tol_eigvals_basic, "\n")
# print("thirty_bicgstab_tol_nb_it_basic ", thirty_bicgstab_tol_nb_it_basic, "\n")
# print("thirty_bicgstab_tol_eigvals_restart ", thirty_bicgstab_tol_eigvals_restart, "\n")
# print("thirty_bicgstab_tol_nb_it_restart  ", thirty_bicgstab_tol_nb_it_restart , "\n")
# print("thirty_bicgstab_julia_min_eigvals ", thirty_bicgstab_julia_min_eigvals, "\n")


# """
# 31. 
# """
# tol_conv = tol_conv_vals[5]
# thirtyone_bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
# thirtyone_bicgstab_tol_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
# thirtyone_bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
# thirtyone_bicgstab_tol_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
# thirtyone_bicgstab_julia_min_eigvals = Vector{Float64}(undef, length(operatorDim_vals))
# loop_nb = 10
# @threads for index = 1:length(operatorDim_vals)
# 	sum_eigval_basic = 0.0
# 	sum_nb_it_basic = 0.0
# 	sum_eigval_restart = 0.0
# 	sum_nb_it_restart = 0.0
# 	for i = 1:loop_nb
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)
# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		julia_min_eigval = trueEigSys.values[minEigPos]
# 		thirtyone_bicgstab_julia_min_eigvals[index] = julia_min_eigval
		
# 		dims = size(opt)
# 		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 		kMat = zeros(ComplexF64, dims[2], dims[2])

# 		print("Start of basic program \n")
# 		eigval_basic,nb_it_basic  = 
# 			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		print("Start of restart program \n")
# 		eigval_restart,nb_it_restart  = 
# 			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 					tol_bicgstab,solver_method)
# 		sum_eigval_basic += eigval_basic 
# 		sum_nb_it_basic += nb_it_basic 
# 		sum_eigval_restart += eigval_restart
# 		sum_nb_it_restart += nb_it_restart 
# 	end
# 	thirtyone_bicgstab_tol_eigvals_basic[index] = sum_eigval_basic/loop_nb
# 	thirtyone_bicgstab_tol_nb_it_basic[index] = sum_nb_it_basic/loop_nb
# 	thirtyone_bicgstab_tol_eigvals_restart[index] = sum_eigval_restart/loop_nb
# 	thirtyone_bicgstab_tol_nb_it_restart[index] = sum_nb_it_restart/loop_nb
# end
# print("thirtyone_bicgstab_tol_eigvals_basic ", thirtyone_bicgstab_tol_eigvals_basic, "\n")
# print("thirtyone_bicgstab_tol_nb_it_basic ", thirtyone_bicgstab_tol_nb_it_basic, "\n")
# print("v_bicgstab_tol_eigvals_restart ", thirtyone_bicgstab_tol_eigvals_restart, "\n")
# print("thirtyone_bicgstab_tol_nb_it_restart  ", thirtyone_bicgstab_tol_nb_it_restart , "\n")
# print("thirtyone_bicgstab_julia_min_eigvals ", thirtyone_bicgstab_julia_min_eigvals, "\n")




# # Graph of the number of iterations vs operator size for different 
# # bicgstab tolerances for the restart Harmonic Ritz program.
# plot(operatorDim_vals, [first_bicgstab_tol_nb_it_restart second_bicgstab_tol_nb_it_restart third_bicgstab_tol_nb_it_restart fourth_bicgstab_tol_nb_it_restart fifth_bicgstab_tol_nb_it_restart six_direct_solve_nb_it_restart seven_bicgstab_tol_nb_it_restart eight_bicgstab_tol_nb_it_restart nine_bicgstab_tol_nb_it_restart ten_bicgstab_tol_nb_it_restart eleven_bicgstab_tol_nb_it_restart twelve_direct_solve_nb_it_restart thirteen_bicgstab_tol_nb_it_restart fourteen_bicgstab_tol_nb_it_restart fifteen_bicgstab_tol_nb_it_restart sixteen_bicgstab_tol_nb_it_restart seventeen_bicgstab_tol_nb_it_restart eighteen_bicgstab_tol_nb_it_restart nineteen_bicgstab_tol_nb_it_restart twenty_bicgstab_tol_nb_it_restart twenty_bicgstab_tol_nb_it_restart twentyone_bicgstab_tol_nb_it_restart twentytwo_bicgstab_tol_nb_it_restart twentythree_bicgstab_tol_nb_it_restart twentyfour_bicgstab_tol_nb_it_restart twentyfive_bicgstab_tol_nb_it_restart twentysix_bicgstab_tol_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
# 	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 or 10",
# 		label=["bicgstab_tol=1e-2,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6" "direct solve,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-2,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6" "direct solve,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6,eigval_tol=1e-4" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6,eigval_tol=1e-5" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6,eigval_tol=1e-7" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6,eigval_tol=1e-9" "bicgstab_tol=1e-2,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=20,conv_tol=1e-6"]
# 		, legend = :outertopleft) # 
# 		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
# plot!(size=(2500,1000))
# xlabel!("Operator size")
# ylabel!("Number of iterations to converge")
# savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/May 31st graphs/nb_it_vs_operator_size.png")


# Graph of the number of iterations vs operator size for different 
# bicgstab tolerances for the restart Harmonic Ritz program.
# Graphs 1-6 -> restartDim = 3 and conv_tol = 1e-6

print("size(first_bicgstab_tol_nb_it_restart)",size(first_bicgstab_tol_nb_it_restart),"\n")
print("size(second_bicgstab_tol_nb_it_restart)",size(second_bicgstab_tol_nb_it_restart),"\n")
print("size(third_bicgstab_tol_nb_it_restart)",size(third_bicgstab_tol_nb_it_restart),"\n")
print("size(fourth_bicgstab_tol_nb_it_restart)",size(fourth_bicgstab_tol_nb_it_restart),"\n")
print("size(fifth_bicgstab_tol_nb_it_restart)",size(fifth_bicgstab_tol_nb_it_restart),"\n")
print("size(six_direct_solve_nb_it_restart)",size(six_direct_solve_nb_it_restart),"\n")
print("size(seven_bicgstab_tol_nb_it_restart)",size(seven_bicgstab_tol_nb_it_restart),"\n")
print("size(eight_bicgstab_tol_nb_it_restart)",size(eight_bicgstab_tol_nb_it_restart),"\n")
print("size(nine_bicgstab_tol_nb_it_restart)",size(nine_bicgstab_tol_nb_it_restart),"\n")
print("size(ten_bicgstab_tol_nb_it_restart)",size(ten_bicgstab_tol_nb_it_restart),"\n")
print("size(eleven_bicgstab_tol_nb_it_restart)",size(eleven_bicgstab_tol_nb_it_restart),"\n")
print("size(twelve_direct_solve_nb_it_restart)",size(twelve_direct_solve_nb_it_restart),"\n")
print("size(thirteen_bicgstab_tol_nb_it_restart)",size(thirteen_bicgstab_tol_nb_it_restart),"\n")
print("size(fourteen_bicgstab_tol_nb_it_restart)",size(fourteen_bicgstab_tol_nb_it_restart),"\n")
print("size(fifteen_bicgstab_tol_nb_it_restart)",size(fifteen_bicgstab_tol_nb_it_restart),"\n")
print("size(sixteen_bicgstab_tol_nb_it_restart)",size(sixteen_bicgstab_tol_nb_it_restart),"\n")
print("size(seventeen_bicgstab_tol_nb_it_restart)",size(seventeen_bicgstab_tol_nb_it_restart),"\n")
print("size(eighteen_bicgstab_tol_eigvals_restart)",size(eighteen_direct_solve_nb_it_restart),"\n")


plot(operatorDim_vals, [first_bicgstab_tol_nb_it_restart second_bicgstab_tol_nb_it_restart third_bicgstab_tol_nb_it_restart fourth_bicgstab_tol_nb_it_restart fifth_bicgstab_tol_nb_it_restart six_direct_solve_nb_it_restart seven_bicgstab_tol_nb_it_restart eight_bicgstab_tol_nb_it_restart nine_bicgstab_tol_nb_it_restart ten_bicgstab_tol_nb_it_restart eleven_bicgstab_tol_nb_it_restart twelve_direct_solve_nb_it_restart thirteen_bicgstab_tol_nb_it_restart fourteen_bicgstab_tol_nb_it_restart fifteen_bicgstab_tol_nb_it_restart sixteen_bicgstab_tol_nb_it_restart seventeen_bicgstab_tol_nb_it_restart eighteen_direct_solve_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 ",
		label=["bicgstab_tol=1e-2,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6" "direct solve,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-2,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6" "direct solve,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-2,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=20,conv_tol=1e-6" "direct solve,restartDim=20,conv_tol=1e-6"]
		, legend = :outertopleft) # 
		# , linecolor=[:red,:blue,:cyan,:green,:indigo,:midnightblue,:purple,:plum,:orchid,:maroon,:pink,:tan1,:brown,:gray47,:black,:gold,:yellow,:salmon]
		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
plot!(size=(2500,1000))
xlabel!("Operator size")
ylabel!("Number of iterations to converge")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/May 31st graphs/nb_it_vs_operator_size_DiffBicgstabTolerances_innerLoopDim25_DiffRestartDim3_10_20.png")


plot(operatorDim_vals, [first_bicgstab_tol_nb_it_restart second_bicgstab_tol_nb_it_restart third_bicgstab_tol_nb_it_restart fourth_bicgstab_tol_nb_it_restart fifth_bicgstab_tol_nb_it_restart six_direct_solve_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 ",
		label=["bicgstab_tol=1e-2,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6" "direct solve,restartDim=3,conv_tol=1e-6"]
		, legend = :outertopleft) # 
		# , linecolor=[:red,:blue,:cyan,:green,:indigo,:midnightblue,:purple,:plum,:orchid,:maroon,:pink,:tan1,:brown,:gray47,:black,:gold,:yellow,:salmon]
		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
plot!(size=(2500,1000))
xlabel!("Operator size")
ylabel!("Number of iterations to converge")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/May 31st graphs/nb_it_vs_operator_size_DiffBicgstabTolerances_innerLoopDim25_DiffRestartDim3.png")

plot(operatorDim_vals, [seven_bicgstab_tol_nb_it_restart eight_bicgstab_tol_nb_it_restart nine_bicgstab_tol_nb_it_restart ten_bicgstab_tol_nb_it_restart eleven_bicgstab_tol_nb_it_restart twelve_direct_solve_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 ",
		label=["bicgstab_tol=1e-2,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6" "direct solve,restartDim=10,conv_tol=1e-6"]
		, legend = :outertopleft) # 
		# , linecolor=[:red,:blue,:cyan,:green,:indigo,:midnightblue,:purple,:plum,:orchid,:maroon,:pink,:tan1,:brown,:gray47,:black,:gold,:yellow,:salmon]
		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
plot!(size=(2500,1000))
xlabel!("Operator size")
ylabel!("Number of iterations to converge")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/May 31st graphs/nb_it_vs_operator_size_DiffBicgstabTolerances_innerLoopDim25_DiffRestartDim10.png")

plot(operatorDim_vals, [thirteen_bicgstab_tol_nb_it_restart fourteen_bicgstab_tol_nb_it_restart fifteen_bicgstab_tol_nb_it_restart sixteen_bicgstab_tol_nb_it_restart seventeen_bicgstab_tol_nb_it_restart eighteen_direct_solve_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 ",
		label=["bicgstab_tol=1e-2,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=20,conv_tol=1e-6" "direct solve,restartDim=20,conv_tol=1e-6"]
		, legend = :outertopleft) # 
		# , linecolor=[:red,:blue,:cyan,:green,:indigo,:midnightblue,:purple,:plum,:orchid,:maroon,:pink,:tan1,:brown,:gray47,:black,:gold,:yellow,:salmon]
		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
plot!(size=(2500,1000))
xlabel!("Operator size")
ylabel!("Number of iterations to converge")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/May 31st graphs/nb_it_vs_operator_size_DiffBicgstabTolerances_innerLoopDim25_DiffRestartDim20.png")


# # Graph of the number of iterations vs operator size for different 
# # bicgstab tolerances for the restart Harmonic Ritz program.
# # Graphs 7-12 -> restartDim = 10 and conv_tol = 1e-6
# plot(operatorDim_vals, [seven_bicgstab_tol_nb_it_restart eight_bicgstab_tol_nb_it_restart nine_bicgstab_tol_nb_it_restart ten_bicgstab_tol_nb_it_restart eleven_bicgstab_tol_nb_it_restart twelve_direct_solve_nb_it_restart ] , #  1e-4,1e-5,1e-7,1e-9
# 	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 or 10",
# 		label=["bicgstab_tol=1e-2,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6" "direct solve,restartDim=10,conv_tol=1e-6"]
# 		, legend = :outertopleft) # 
# 		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
# plot!(size=(2500,1000))
# xlabel!("Operator size")
# ylabel!("Number of iterations to converge")
# savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/May 31st graphs/nb_it_vs_operator_size_DiffBicgstabTolerances_innerLoopDim25_restartDim10.png")


# # Graph of the number of iterations vs operator size for different 
# # residual vector (resVec) convergence tolerances for the restart Harmonic Ritz program.
# # Graphs 13-17 -> bicgstab_tol = 1e-6 and restartDim = 3
# plot(operatorDim_vals, [thirteen_bicgstab_tol_nb_it_restart fourteen_bicgstab_tol_nb_it_restart fifteen_bicgstab_tol_nb_it_restart sixteen_bicgstab_tol_nb_it_restart seventeen_bicgstab_tol_nb_it_restart ] , #  1e-4,1e-5,1e-7,1e-9
# 	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 or 10",
# 		label=["bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6"]
# 		, legend = :outertopleft) # 
# 		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
# plot!(size=(2500,1000))
# xlabel!("Operator size")
# ylabel!("Number of iterations to converge")
# savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/May 31st graphs/nb_it_vs_operator_size_DiffConvTolerances_BicgstabTol1e-6_restartDim3.png")


# # Graph of the number of iterations vs operator size for different 
# # bicgstab tolerances for the restart Harmonic Ritz program.
# # Graphs 18-> bicgstab_tol=1e-6, restartDim=3 and conv_tol=1e-6
# plot(operatorDim_vals, [eighteen_bicgstab_tol_nb_it_restart nineteen_bicgstab_tol_nb_it_restart twenty_bicgstab_tol_nb_it_restart twenty_bicgstab_tol_nb_it_restart twentyone_bicgstab_tol_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
# 	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 or 10",
# 		label=["bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6,eigval_tol=1e-4" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6,eigval_tol=1e-5" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6,eigval_tol=1e-7" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6,eigval_tol=1e-9"]
# 		, legend = :outertopleft) # 
# 		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
# plot!(size=(2500,1000))
# xlabel!("Operator size")
# ylabel!("Number of iterations to converge")
# savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/May 31st graphs/nb_it_vs_operator_size_DiffEigvalTolerances_BicgstabTol1e-6_restartDim3.png")


# # Graph of the number of iterations vs operator size for different 
# # bicgstab tolerances for the restart Harmonic Ritz program.
# # Graphs 22-26 -> bicgstab_tol = 1e-6 and restartDim = 20
# plot(operatorDim_vals, [twentytwo_bicgstab_tol_nb_it_restart twentythree_bicgstab_tol_nb_it_restart twentyfour_bicgstab_tol_nb_it_restart twentyfive_bicgstab_tol_nb_it_restart twentysix_bicgstab_tol_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
# 	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 or 10",
# 		label=["bicgstab_tol=1e-2,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=20,conv_tol=1e-6"]
# 		, legend = :outertopleft) # 
# 		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
# plot!(size=(2500,1000))
# xlabel!("Operator size")
# ylabel!("Number of iterations to converge")
# savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/May 31st graphs/nb_it_vs_operator_size_DiffConvTolerances_BicgstabTol1e-6_restartDim20.png")


# # Graph of the number of iterations vs operator size for different 
# # bicgstab tolerances for the restart Harmonic Ritz program.
# plot(operatorDim_vals, [first_bicgstab_tol_nb_it_restart second_bicgstab_tol_nb_it_restart third_bicgstab_tol_nb_it_restart fourth_bicgstab_tol_nb_it_restart fifth_bicgstab_tol_nb_it_restart six_direct_solve_nb_it_restart seven_bicgstab_tol_nb_it_restart eight_bicgstab_tol_nb_it_restart nine_bicgstab_tol_nb_it_restart ten_bicgstab_tol_nb_it_restart eleven_bicgstab_tol_nb_it_restart twelve_direct_solve_nb_it_restart  twentytwo_bicgstab_tol_nb_it_restart twentythree_bicgstab_tol_nb_it_restart twentyfour_bicgstab_tol_nb_it_restart twentyfive_bicgstab_tol_nb_it_restart twentysix_bicgstab_tol_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
# 	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 or 10",
# 		label=["bicgstab_tol=1e-2,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6" "direct solve,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-2,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6" "direct solve,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-2,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=20,conv_tol=1e-6"]
# 		, legend = :outertopleft) # 
# 		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
# plot!(size=(2500,1000))
# xlabel!("Operator size")
# ylabel!("Number of iterations to converge")
# savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/May 31st graphs/nb_it_vs_operator_size_ComparisonOf3RestartSizes.png")



# # tol_bicgstab = 1.0e-12
# tol_bicgstab_vals = [1e-2,1e-3,1e-4,1e-5,1e-6] # ,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12 
# nb_it_vals_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
# nb_it_vals_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))
# eigval_julia_vals = Vector{Float64}(undef, length(tol_bicgstab_vals))
# eigval_basic_vals = Vector{Float64}(undef, length(tol_bicgstab_vals))
# eigval_restart_vals = Vector{Float64}(undef, length(tol_bicgstab_vals))
# eigval_sums_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
# eigval_sums_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
# nb_it_sums_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
# nb_it_sums_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))

# all_julia_eigvals = Vector{Vector}(undef, length(operatorDim_vals))

# # print("all_julia_eigvals ", all_julia_eigvals, "\n")
# # print("all_julia_eigvals[2] ", all_julia_eigvals[2], "\n")

# nb_test_loops = 10

# bicgstab_tol_eigval_sums_basic = Vector{Vector}(undef, nb_test_loops)
# bicgstab_tol_eigval_sums_restart = Vector{Vector}(undef, nb_test_loops)
# bicgstab_tol_nb_it_sums_basic = Vector{Vector}(undef, nb_test_loops)
# bicgstab_tol_nb_it_sums_restart = Vector{Vector}(undef, nb_test_loops)

# # print("bicgstab_tol_eigval_sums_basic ", bicgstab_tol_eigval_sums_basic, "\n")
# # print("bicgstab_tol_eigval_sums_basic[2] ", bicgstab_tol_eigval_sums_basic[2], "\n")

# for loops = 1:nb_test_loops
# 	@threads for index = 1:length(operatorDim_vals)
# 		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
# 		rand!(opt)

# 		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
# 		trueEigSys = eigen(opt)
# 		minEigPos = argmin(abs.(trueEigSys.values))
# 		eigval_julia_vals[index] = trueEigSys.values[minEigPos]

# 		all_julia_eigvals[index] = trueEigSys.values

# 		dims = size(opt)

# 		@threads for index = 1:length(tol_bicgstab_vals)
# 			print("index ", index, "\n")
# 			bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
# 			bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
# 			trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 			srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
# 			kMat = zeros(ComplexF64, dims[2], dims[2])
# 			print("Start of basic program \n")
# 			eigval_basic_vals[index],nb_it_vals_basic[index] = 
# 				jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
# 					dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_eigval,tol_bicgstab_vals[index])
# 			print("Start of restart program \n")
# 			eigval_restart_vals[index], nb_it_vals_restart[index] = 
# 				jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 					dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 						tol_bicgstab_vals[index])
# 			print("eigval_restart_vals[index] ", eigval_restart_vals[index], "\n")
# 			print("nb_it_vals_restart[index] ", nb_it_vals_restart[index], "\n")
# 		end 
# 		@threads for index = 1:length(tol_bicgstab_vals)
# 			eigval_sums_basic[index] += eigval_basic_vals[index]
# 			eigval_sums_restart[index] += eigval_restart_vals[index]
# 			nb_it_sums_basic[index] += nb_it_vals_basic[index]
# 			nb_it_sums_restart[index] += nb_it_vals_restart[index]
# 		end 
# 	end
# 	bicgstab_tol_eigval_sums_basic[loops] = eigval_sums_basic
# 	bicgstab_tol_eigval_sums_restart[loops] = eigval_sums_restart
# 	bicgstab_tol_nb_it_sums_basic[loops] = nb_it_sums_basic
# 	bicgstab_tol_nb_it_sums_restart[loops] = nb_it_sums_restart
# end
# print("eigval_sums_basic ", eigval_sums_basic, "\n")
# print("eigval_sums_restart ", eigval_sums_restart, "\n")
# print("nb_it_sums_basic ", nb_it_sums_basic, "\n")
# print("nb_it_sums_restart ", nb_it_sums_restart, "\n")

# print("bicgstab_tol_eigval_sums_basic ", bicgstab_tol_eigval_sums_basic, "\n")
# print("bicgstab_tol_eigval_sums_restart ", bicgstab_tol_eigval_sums_restart, "\n")
# print("bicgstab_tol_nb_it_sums_basic ", bicgstab_tol_nb_it_sums_basic, "\n")
# print("bicgstab_tol_nb_it_sums_restart ", bicgstab_tol_nb_it_sums_restart, "\n")

# avg_eigvals_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
# avg_eigvals_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
# avg_nb_it_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
# avg_nb_it_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
# @threads for index = 1:length(tol_bicgstab_vals)
# 	@threads for index = 1:length(tol_bicgstab_vals)
# 		avg_eigvals_basic[index] += eigval_basic_vals[index]/length(eigval_basic_vals)
# 		avg_eigvals_restart[index] += eigval_restart_vals[index]/length(eigval_restart_vals)
# 		avg_nb_it_basic[index] += nb_it_vals_basic[index]/length(nb_it_vals_basic)
# 		avg_nb_it_restart[index] += nb_it_vals_restart[index]/length(nb_it_vals_restart)
# 	end 
# end

# rel_diff_eigval_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
# rel_diff_eigval_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))

# @threads for loops = 1:nb_test_loops
# 	@threads for index = 1:length(tol_bicgstab_vals)
# 		rel_diff_eigval_basic[index] += abs((avg_eigvals_basic[index]-eigval_julia_vals[loops])/eigval_julia_vals[loops])*100
# 		rel_diff_eigval_restart[index] += abs((avg_eigvals_restart[index]-eigval_julia_vals[loops])/eigval_julia_vals[loops])*100
# 	end 
# end

# print("tol_bicgstab_vals", tol_bicgstab_vals, "\n")

# print("Basic algo - HarmonicRitz number of iterations is ", avg_nb_it_basic, "\n")
# print("Restart algo - HarmonicRitz number of iterations is ", avg_nb_it_restart, "\n")

# print("No restart - HarmonicRitz smallest positive eigenvalue is ", avg_eigvals_basic, "\n")
# print("Restart - HarmonicRitz smallest positive eigenvalue is ", avg_eigvals_restart, "\n")
# println("Julia smallest positive eigenvalue is ", eigval_julia_vals,"\n")

# print("Basic algo - HarmonicRitz relative difference between eigvals are ", rel_diff_eigval_basic, "\n")
# print("Restart algo - HarmonicRitz relative difference between eigvals are  ", rel_diff_eigval_restart, "\n")

# println("Julia eigenvalues ", trueEigSys.values,"\n")


# """ 1. Number of iterations vs bicgstab tolerance graphs """
# # Graph of the number of iterations to converge for the basic Harmonic Ritz 
# # program.
# plot(tol_bicgstab_vals,avg_nb_it_basic,title="Number of iterations vs bicgstab tolerance \n basic Harm Ritz")
# xlabel!("Bicgstab tolerance values")
# ylabel!("Number of iterations")
# savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing bicgstab tolerance/tol_bicgstab_basic_vs_nb_it.png")

# # Graph of the number of iterations to converge for the restart Harmonic Ritz 
# # program.
# plot(tol_bicgstab_vals,avg_nb_it_restart,title="Number of iterations vs bicgstab tolerance \n restart Harm Ritz")
# xlabel!("Bicgstab tolerance values")
# ylabel!("Number of iterations")
# savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing bicgstab tolerance/tol_bicgstab_restart_vs_nb_it.png")

# """ 2. Relative difference vs bicgstab tolerance graphs """
# # Graph of the relative different between the eigvals vs bicgstab tolerance 
# # for the basic Harmonic Ritz program.
# plot(tol_bicgstab_vals,rel_diff_eigval_basic,title="Eigvals relative different vs bicgstab tolerance \n basic Harm Ritz")
# xlabel!("Bicgstab tolerance values")
# ylabel!("Eigvals relative difference with Julia solve")
# savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing bicgstab tolerance/tol_bicgstab_basic_vs_eigvals_relative_diff.png")

# # Graph of the relative different between the eigvals vs bicgstab tolerance 
# # for the restart Harmonic Ritz program.
# plot(tol_bicgstab_vals,rel_diff_eigval_restart,title="Eigvals relative different vs bicgstab tolerance \n restart Harm Ritz")
# xlabel!("Bicgstab tolerance values")
# ylabel!("Number of iterations")
# savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing bicgstab tolerance/tol_bicgstab_restart_vs_eigvals_relative_diff.png")
