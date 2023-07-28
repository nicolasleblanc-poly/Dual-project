module Davidson_Operator_HarmonicRitz_module_May16_NbIt_vs_RestartSize_Graph

using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, Av_product_Davidson, Plots

export jacDavRitzHarm_restart, jacDavRitzHarm_basic

# jacDavRitzHarm_restart(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
# 	P,alpha,innerLoopDim::Integer,restartDim::Integer,dims::Integer,tol::Float64)::Float64

# jacDavRitzHarm_restart(trgBasis::Array{ComplexF64}, 
# 	srcBasis::Array{ComplexF64}, kMat::Array{ComplexF64}, 
# 	opt::Array{ComplexF64}, vecDim::Integer, repDim::Integer, 
# 	innerLoopDim::Integer,restartDim::Integer,tol_MGS::Float64,tol_conv::Float64)::Float64
function jacDavRitzHarm_basic(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,alpha,
	innerLoopDim::Integer,dims::Integer,tol_MGS::Float64,tol_conv::Float64,tol_eigval::Float64)::Tuple{Float64, Int32}
	# Matrices memory initialization
	trgBasis = Array{ComplexF64}(undef, dims, dims)
	srcBasis = Array{ComplexF64}(undef, dims, dims)
	kMat = zeros(ComplexF64, dims, dims)

	nb_it_vals_basic = 0

	# set starting vector
	### memory initialization
	resVec = Vector{ComplexF64}(undef, dims)
	hRitzTrg = Vector{ComplexF64}(undef, dims)
	hRitzSrc = Vector{ComplexF64}(undef, dims)
	bCoeffs1 = Vector{ComplexF64}(undef, dims)
	bCoeffs2 = Vector{ComplexF64}(undef, dims)
	# set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# normalize starting vector
	nrm = BLAS.nrm2(dims, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	### algorithm initialization
	# trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
	trgBasis[:, 1] = A_v_product(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
		P,alpha,srcBasis[:, 1])
	nrm = BLAS.nrm2(dims, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(dims, view(trgBasis, :, 1), 1,
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

	previous_eigval = 0
	
	# Code for if we just want the inner loop, so with no restart  
	for itr in 2 : innerLoopDim  # Need to determine when this for loops stops 
		# depending on how much memory the laptop can take before crashing.
		prjCoeff = BLAS.dotc(dims, hRitzTrg, 1, hRitzSrc, 1)
		# calculate Jacobi-Davidson direction
		srcBasis[:, itr] = bad_bicgstab_matrix(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
				P,alpha, theta, hRitzTrg, hRitzSrc, prjCoeff, resVec)
			# bad_bicgstab_matrix(opt, theta, hRitzTrg,
			# 	hRitzSrc, prjCoeff, resVec)
		trgBasis[:, itr] = A_v_product(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
			P,alpha,srcBasis[:, itr])
			# opt * srcBasis[:, itr]
		# orthogonalize
		gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, gMemSlfN,
			gMemSlfA,cellsA,chi_inv_coeff,P,alpha,itr, tol_MGS)
		# srcBasis[:, itr] = bad_bicgstab_matrix(opt, theta, hRitzTrg,
		# 	hRitzSrc, prjCoeff, resVec)
		# trgBasis[:, itr] = opt * srcBasis[:, itr]
		# # orthogonalize
		# gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, opt,
		# 	itr, tol_MGS)

		# update inverse representation of opt^{-1} in trgBasis
		kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
			view(srcBasis, :, itr))
		# assuming opt^{-1} Hermitian matrix
		kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])
		# eigenvalue decomposition, largest real eigenvalue last.
		# should replace by BLAS operation
		eigSys = eigen(view(kMat, 1 : itr, 1 : itr))
	

		# update Ritz vector
		if abs.(eigSys.values[end]) > abs.(eigSys.values[1])
			theta = 1/eigSys.values[end]
			hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, end])
			hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, end])
		else
			theta = 1/eigSys.values[1]
			hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
			hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
		end	

		# update residual vector
		resVec = (theta * hRitzSrc) .- hRitzTrg
 
		# Direction vector tolerance check 
		if norm(resVec) < tol_conv
			# print("Converged off tolerance \n")
			# print("norm(resVec) ", norm(resVec), "\n")
			return real(theta), real(nb_it_vals_basic)
			# println(real(theta))
		end
		# Eigenvalue tolerance check
		if real(theta) - real(previous_eigval) < tol_eigval
			print("Converged off tolerance \n")
			return real(theta), real(nb_it_vals_basic)
		end
		# Eigenvalue tolerance check
			# if real(theta) - real(previous_eigval) < tol_eigval
				# print("Converged off tolerance \n")
			# 	print("Converged off tolerance \n")
			# 	return real(theta), real(nb_it_vals_restart)
		# end
		previous_eigval = theta
		# print("norm(resVec) basic program ", norm(resVec),"\n")
		nb_it_vals_basic += 1
		print("nb_it_vals_basic ", nb_it_vals_basic, "\n")
	end
 
	print("Didn't converge off tolerance for basic program. 
		Atteined max set number of iterations \n")
	return real(theta), real(nb_it_vals_basic)
end


# function jacDavRitzHarm_restart(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
# 	P,alpha,trgBasis::Array{ComplexF64},srcBasis::Array{ComplexF64}, 
# 	kMat::Array{ComplexF64}, vecDim::Integer, repDim::Integer, 
# 	restartLoopDim::Integer,innerLoopDim::Integer,tol::Float64)::Float64
function jacDavRitzHarm_restart(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
	P,alpha,innerLoopDim::Integer,restartDim::Integer,dims::Integer,
	tol_MGS::Float64,tol_conv::Float64,tol_eigval::Float64)::Tuple{Float64, Int32}

	nb_it_vals_restart = 0

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	# eigenvectors = Array{ComplexF64}(undef, innerLoopDim,restartDim)
	eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
	
	restart_resVec = Vector{ComplexF64}(undef, dims)
	restart_hRitzTrg = Vector{ComplexF64}(undef, dims)
	restart_hRitzSrc = Vector{ComplexF64}(undef, dims)
	restart_trgBasis = Array{ComplexF64}(undef, dims, restartDim)
	restart_srcBasis = Array{ComplexF64}(undef, dims, restartDim)
	restart_kMat = zeros(ComplexF64, dims, restartDim)
	restart_theta = 0 

	# Matrices memory initialization
	trgBasis = Array{ComplexF64}(undef, dims, dims)
	srcBasis = Array{ComplexF64}(undef, dims, dims)
	kMat = zeros(ComplexF64, dims, dims)
	# restart_srcBasis = Array{ComplexF64}(undef, numberRestartVals, numberRestartVals)
	# restart_srcBasis = Vector{ComplexF64}(undef, dims)
	# Vectors memory initialization
	resVec = Vector{ComplexF64}(undef, dims)
	hRitzTrg = Vector{ComplexF64}(undef, dims)
	hRitzSrc = Vector{ComplexF64}(undef, dims)
	bCoeffs1 = Vector{ComplexF64}(undef, dims)
	bCoeffs2 = Vector{ComplexF64}(undef, dims)
	# set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# normalize starting vector
	# nrm = BLAS.nrm2(vecDim, view(srcBasis,:,1), 1) # norm(vk)
	nrm = BLAS.nrm2(dims, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	### algorithm initialization
	# trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
	trgBasis[:, 1] = A_v_product(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
		P,alpha,srcBasis[:, 1])

	nrm = BLAS.nrm2(dims, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(dims, view(trgBasis, :, 1), 1,
		view(srcBasis, :, 1), 1) # Kk
	# Ritz value
	eigPos = 1
	theta = 1 / kMat[1,1] # eigenvalue 
	# Ritz vectors
	hRitzTrg[:] = trgBasis[:, 1] # hk = wk 
	hRitzSrc[:] = srcBasis[:, 1] # fk = vk
	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

	previous_eigval = 0

	# Code with restart
	# Outer loop
	for it in 1:restartDim # Need to think this over 
		eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
		# Inner loop
		if it > 1
			# Before we restart, we will create a new version of everything 
			resVec = Vector{ComplexF64}(undef, dims)
			resVec = restart_resVec
			hRitzTrg = Vector{ComplexF64}(undef, dims)
			hRitzTrg = restart_hRitzTrg
			hRitzSrc = Vector{ComplexF64}(undef, dims)
			hRitzSrc = restart_hRitzSrc
			
			bCoeffs1 = Vector{ComplexF64}(undef, dims)
			bCoeffs2 = Vector{ComplexF64}(undef, dims)
			
			trgBasis = Array{ComplexF64}(undef, dims, dims) # dims[1],dims[2]
			copyto!(trgBasis[:,1:restartDim],restart_trgBasis)
			
			srcBasis = Array{ComplexF64}(undef, dims, dims) # dims[1],dims[2]
			copyto!(srcBasis[:,1:restartDim],restart_srcBasis)

			theta = restart_theta

			kMat = zeros(ComplexF64, dims, dims)
			# print("size(restart_kMat) ", size(restart_kMat), "\n")
			# print("size(kMat) ", size(kMat), "\n")
			# kMat[1:restartDim,1:restartDim] = restart_kMat 
			kMat[:,1:restartDim] = restart_kMat
		
			for itr in restartDim+1: innerLoopDim # -restartDim # Need to determine when this for loops stops 
				# depending on how much memory the laptop can take before crashing.
				prjCoeff = BLAS.dotc(dims, hRitzTrg, 1, hRitzSrc, 1)
				# calculate Jacobi-Davidson direction
				srcBasis[:, itr] = bad_bicgstab_matrix(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
				P,alpha, theta, hRitzTrg, hRitzSrc, prjCoeff, resVec)
				# bad_bicgstab_matrix(opt, theta, hRitzTrg,
				# 	hRitzSrc, prjCoeff, resVec)
				trgBasis[:, itr] = A_v_product(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
					P,alpha,srcBasis[:, itr])
				# opt * srcBasis[:, itr]
				# orthogonalize
				gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, gMemSlfN,
					gMemSlfA,cellsA,chi_inv_coeff,P,alpha,itr, tol_MGS)
				# gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, opt,
				# 	itr, tol_MGS)

				print("srcBasis 2 ", srcBasis, "\n")
				print("trgBasis 2 ", trgBasis, "\n")					

				# update inverse representation of opt^{-1} in trgBasis
				kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
					view(srcBasis, :, itr))
				# assuming opt^{-1} Hermitian matrix
				kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])

				print("kMat 2 ", kMat, "\n")

				# eigenvalue decomposition, largest real eigenvalue last.
				# should replace by BLAS operation
				eigSys = eigen(view(kMat, 1 : itr, 1 : itr))

				eigenvectors[1:itr,1:itr] = eigSys.vectors
				eigenvalues[1:itr] = eigSys.values
		
				print("itr ", itr, "\n")
				print("abs.(eigSys.values[end]) ", abs.(eigSys.values[end]), "\n")
				print("abs.(eigSys.values[1]) ", abs.(eigSys.values[1]), "\n")
				print("eigSys.values ", eigSys.values, "\n")

				# update Ritz vector
				"""
				- We want the largest eigenvalue in absolue value since 
				it will give the smallest eigenvalue that is closest to 0. 
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

		
				# update residual vector
				resVec = (theta * hRitzSrc) .- hRitzTrg
		
				# Direction vector tolerance check 
				if norm(resVec) < tol_conv
					# print("Converged off tolerance \n")
					# print("norm(resVec) ", norm(resVec), "\n")
					print("Converged off tolerance \n")
					return real(theta), real(nb_it_vals_restart)
					# println(real(theta))
				end
				# Eigenvalue tolerance check
				# if real(theta) - real(previous_eigval) < tol_eigval
					# print("Converged off tolerance \n")
				# 	print("Converged off tolerance \n")
				# 	return real(theta), real(nb_it_vals_restart)
				# end
				previous_eigval = theta
				# print("norm(resVec) restart program ", norm(resVec),"\n")
				# eigenvectors[:,1:itr] = eigSys.vectors
				# eigenvalues[1:itr] = eigSys.values
			end
		
		# Essentially for the case when it = 0
		else
			print("First Loop \n")
			for itr in 2 : innerLoopDim # Need to determine when this for loops stops 
				# depending on how much memory the laptop can take before crashing.
				prjCoeff = BLAS.dotc(dims, hRitzTrg, 1, hRitzSrc, 1)
				# calculate Jacobi-Davidson direction
				srcBasis[:, itr] = bad_bicgstab_matrix(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
					P,alpha, theta, hRitzTrg, hRitzSrc, prjCoeff, resVec)
				# bad_bicgstab_matrix(opt, theta, hRitzTrg,
				# 	hRitzSrc, prjCoeff, resVec)
				trgBasis[:, itr] = A_v_product(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
					P,alpha,srcBasis[:, itr])
				# opt * srcBasis[:, itr]

				# orthogonalize
				# gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, opt,
				# 	itr, tol_MGS)
				gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, gMemSlfN,
					gMemSlfA,cellsA,chi_inv_coeff,P,alpha,itr, tol_MGS)

				# print("srcBasis 1 ", srcBasis, "\n")
				# print("trgBasis 1 ", trgBasis, "\n")

				# update inverse representation of opt^{-1} in trgBasis
				kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
					view(srcBasis, :, itr))
				# assuming opt^{-1} Hermitian matrix
				kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])

				print("kMat 1 ", kMat, "\n")
				
				# eigenvalue decomposition, largest real eigenvalue last.
				# should replace by BLAS operation
				eigSys = eigen(view(kMat, 1 : itr, 1 : itr))
				# print("size(eigSys.vectors) ", size(eigSys.vectors), "\n")

				eigenvectors[1:itr,1:itr] = eigSys.vectors
				eigenvalues[1:itr] = eigSys.values
		
				# print("itr ", itr, "\n")
				# print("abs.(eigSys.values[end]) ", abs.(eigSys.values[end]), "\n")
				# print("abs.(eigSys.values[1]) ", abs.(eigSys.values[1]), "\n")
				# print("eigSys.values ", eigSys.values, "\n")

				# update Ritz vector
				if abs.(eigSys.values[end]) > abs.(eigSys.values[1])
					# print("If statement 2.1 \n")
					# if_val = 1
					theta = 1/eigSys.values[end]
					hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, end])
					hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, end])
				else
					# print("If statement 2.2 \n")
					# if_val = 2
					theta = 1/eigSys.values[1]
					hRitzTrg[:] = trgBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
					hRitzSrc[:] = srcBasis[:, 1 : itr] * (eigSys.vectors[:, 1])
				end	

				# update residual vector
				resVec = (theta * hRitzSrc) .- hRitzTrg
		
				# Direction vector tolerance check 
				if norm(resVec) < tol_conv
					# print("Converged off tolerance \n")
					# print("norm(resVec) ", norm(resVec), "\n")
					print("Converged off tolerance \n")
					return real(theta), real(nb_it_vals_restart)
					# println(real(theta))
				end
				# Eigenvalue tolerance check
				# if real(theta) - real(previous_eigval) < tol_eigval
					# print("Converged off tolerance \n")
				# 	return real(theta), real(nb_it_vals_restart)
				# end
				previous_eigval = theta
						# print("norm(resVec) restart program ", norm(resVec),"\n")
			end
		end

		"""
		Time to save important stuff before restarting
		"""

		print("eigenvalues out of loop ", eigenvalues, "\n")

		"""
		Create trgBasis and srcBasis for restart
		"""
		# for i in range(0,1,restartDim)
		for i = 1:restartDim
			print("eigenvalues[1] ", eigenvalues[1], "\n")
			print("eigenvalues[end] ", eigenvalues[end], "\n")
			# Creation of new trgBasis and srcBasis matrices 
			if abs.(eigenvalues[end]) > abs.(eigenvalues[1]) 
				restart_trgBasis[:,i] = trgBasis[:, 1 : innerLoopDim] * (eigenvectors[:, end])
				restart_srcBasis[:,i] = srcBasis[:, 1 : innerLoopDim] * (eigenvectors[:, end])
				pop!(eigenvalues) # This removes the last element
			else # abs.(eigSys.values[end]) < abs.(eigSys.values[1])
				restart_trgBasis[:,i] = trgBasis[:, 1 : innerLoopDim] * (eigenvectors[:, 1])
				restart_srcBasis[:,i] = srcBasis[:, 1 : innerLoopDim] * (eigenvectors[:, 1])
				popfirst!(eigenvalues) # This removes the first element
			end
			print("eigenvalues restart ", eigenvalues, "\n")

			# # update inverse representation of opt^{-1} in trgBasis
			# kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
			# view(srcBasis, :, itr))
			# # assuming opt^{-1} Hermitian matrix
			# kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])
		end

		"""
		Create kMat for restart
		"""
		for i = 1:restartDim
			if i == 1
				# representation of opt^{-1} in trgBasis
				restart_kMat[1,1] = BLAS.dotc(dims, view(restart_trgBasis, :, 1), 1,
				view(restart_srcBasis, :, 1), 1)
			else # Essentially if i != 1 or i > 1 since i starts at 1 and can't be smaller than 1
				# update inverse representation of opt^{-1} in trgBasis
				restart_kMat[1 : i, i] = BLAS.gemv('C', view(restart_trgBasis, :, 1 : i),
				view(restart_srcBasis, :, i))
				# assuming opt^{-1} Hermitian matrix
				restart_kMat[i, 1 : (i - 1)] = conj(restart_kMat[1 : (i-1), i])
			end
			# print("restart_kMat[1,1] ", restart_kMat[1,1], "\n")
		end

		# print("restart_trgBasis ", restart_trgBasis, "\n")
		# print("restart_srcBasis ", restart_srcBasis, "\n")
		# print("restart_kMat ", restart_kMat, "\n")

		"""
		Save other stuff for restart 
		"""
		
		# copyto !(b, a)
		# a is what we want to copy 
		# b is the destination of storage of the copy 

		restart_resVec = resVec
		restart_hRitzTrg = hRitzTrg
		restart_hRitzSrc = hRitzSrc

		# copyto!(restart_resVec,resVec)
		# copyto!(restart_hRitzTrg,hRitzTrg)
		# copyto!(restart_hRitzSrc,hRitzSrc)

		# print("restart_resVec ", restart_resVec, "\n")
		# print("restart_hRitzTrg ", restart_hRitzTrg, "\n")
		# print("restart_hRitzSrc ", restart_hRitzSrc, "\n")

		restart_theta = theta
		# print("restart_theta ", restart_theta, "\n")

		# restart_trgBasis = trgBasis[:,innerLoopDim-restartDim+1:innerLoopDim]
		# restart_kMat = kMat[innerLoopDim-restartDim+1:innerLoopDim
		# 		,innerLoopDim-restartDim+1:innerLoopDim]

		# restart_kMat = kMat[:,innerLoopDim-restartDim+1:innerLoopDim]
		# print("restart_kMat ", restart_kMat, "\n")

		# Once we have ran out of memory, we want to restart the inner loop 
		# but not with random starting vectors and matrices but with the ones
		# from the last inner loop iteration that was done before running out 
		# of memory. 

		nb_it_vals_restart += 1
		print("nb_it_vals_restart ", nb_it_vals_restart, "\n")

	end	
	print("Didn't converge off tolerance. Atteined max set number of iterations \n")
	print("real(theta) ", real(theta), "\n")
	return real(theta), real(nb_it_vals_restart)
end


# perform Gram-Schmidt on target basis, adjusting source basis accordingly
function gramSchmidtHarm!(trgBasis::Array{T}, srcBasis::Array{T},
	bCoeffs1::Vector{T}, bCoeffs2::Vector{T}, gMemSlfN,gMemSlfA,cellsA,
		chi_inv_coeff,P,alpha, n::Integer,tol::Float64) where T <: Number
	# dimension of vector space
	dim = size(trgBasis)[1]
	# initialize projection norm
	prjNrm = 1.0
	# initialize projection coefficient memory
	bCoeffs1[1:(n-1)] .= 0.0 + im*0.0
	# check that basis does not exceed dimension
	if n > dim
		error("Requested basis size exceeds dimension of vector space.")
	end
	# norm of proposed vector
	nrm = BLAS.nrm2(dim, view(trgBasis,:,n), 1)
	# renormalize new vector
	trgBasis[:,n] = trgBasis[:,n] ./ nrm
	srcBasis[:,n] = srcBasis[:,n] ./ nrm
	# guarded orthogonalization
	while prjNrm > (tol * 100) && abs(nrm) > tol
		### remove projection into existing basis
 		# calculate projection coefficients
 		BLAS.gemv!('C', 1.0 + im*0.0, view(trgBasis, :, 1:(n-1)),
 			view(trgBasis, :, n), 0.0 + im*0.0,
 			view(bCoeffs2, 1:(n -1)))
 		# remove projection coefficients
 		BLAS.gemv!('N', -1.0 + im*0.0, view(trgBasis, :, 1:(n-1)),
 			view(bCoeffs2, 1:(n -1)), 1.0 + im*0.0,
 			view(trgBasis, :, n))
 		# update total projection coefficients
 		bCoeffs1 .= bCoeffs2 .+ bCoeffs1
 		# calculate projection norm
 		prjNrm = BLAS.nrm2(n-1, bCoeffs2, 1)
 	end
 	# remaining norm after removing projections
 	nrm = BLAS.nrm2(dim, view(trgBasis,:,n), 1)
	# check that remaining vector is sufficiently large
	if abs(nrm) < tol
		# switch to random search direction
		rand!(view(srcBasis, :, n))

		# G operator product 
		# trgBasis[:, n] = opt * srcBasis[:, n]
		trgBasis[:, n] = A_v_product(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
		P,alpha,srcBasis[:, n])

		gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2,
			gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,alpha, n, tol)
	else
		# renormalize
		trgBasis[:,n] = trgBasis[:,n] ./ nrm
		srcBasis[:,n] = srcBasis[:,n] ./ nrm
		bCoeffs1 .= bCoeffs1 ./ nrm
		# remove projections from source vector
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
function bad_bicgstab_matrix(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
	P,alpha, theta, hTrg, hSrc, prjC, b)
	# dim = size(A)[1]
	dim = cellsA[1]*cellsA[2]*cellsA[3]*3
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    # tol = 1e-4
    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b
    rho_m1 = alpha = omega_m1 = 1
    # for k in 1:length(b)
    # we don't want a perfect solve, should fix this though
    for k in 1 : 16
        rho_k = conj.(transpose(r0))*r_m1
        # beta calculation
        # First term
        first_term = rho_k/rho_m1
        # Second term
        second_term = alpha/omega_m1
        # Calculation
        beta = first_term*second_term
        pk = r_m1 .+ beta.*(p_m1-omega_m1.*v_m1)
        pkPrj = harmVec(dim, hTrg, hSrc, prjC, pk)
		A_pkPrj = A_v_product(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
		P,alpha,pkPrj)	
        # vk = harmVec(dim, hTrg, hSrc, prjC,
        # 	A * pkPrj .- (theta .* pkPrj))
		vk = harmVec(dim, hTrg, hSrc, prjC,
		A_pkPrj .- (theta .* pkPrj))

        # (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, pk)
        # alpha calculation
        # Bottom term
        bottom_term = conj.(transpose(r0))*vk
        # Calculation
        alpha = rho_k / bottom_term
        h = xk_m1 + alpha.*pk
        # If h is accurate enough, then set xk=h and quantity
        # What does accurate enough mean?
        s = r_m1 - alpha.*vk
        bPrj = harmVec(dim, hTrg, hSrc, prjC, b)
		A_bPrj = A_v_product(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,
		P,alpha,bPrj)	
		
		# t = harmVec(dim, hTrg, hSrc, prjC,
		# A * bPrj .- (theta .* bPrj))
		t = harmVec(dim, hTrg, hSrc, prjC,
        	A_bPrj .- (theta .* bPrj))

        # (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, s)
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
        # print("conj.(transpose(r_m1))*r_m1  ", conj.(transpose(r_m1))*r_m1 , "\n")
        # if real((conj.(transpose(r_m1))*r_m1)[1]) < tol
        # # if norm(r_m1)-norm(r_old) < tol
        #     print("bicgstab break \n")
        #     print("real((conj.(transpose(r_m1))*r_m1)[1])",real((conj.(transpose(r_m1))*r_m1)[1]),"\n")
        #     break
        # end
    end
    return xk_m1
end
end 
