module Restart_Ritz_jacobiDavidson_operator

export ritz_bicgstab_operator,jacDavRitz_basic,jacDavRitz_basic_for_restart,
	jacDavRitz_restart

using LinearAlgebra, Random, product

@inline function projVec(dim::Integer, pVec::Vector{T}, sVec::Array{T})::Array{T} where T <: Number

	return sVec .- (BLAS.dotc(dim, pVec, 1, sVec, 1) .* pVec)
end

# This is a biconjugate gradient program without a preconditioner. 
# m is the maximum number of iterations
function ritz_bicgstab_operator(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,
    chi_inv_coeff,P, theta, u, b, tol_bicgstab)
    # ritz_bicgstab_operator(A, theta, u, b, tol_bicgstab)
    
	# dim = size(A)[1]
    
    dim = cellsA[1]*cellsA[2]*cellsA[3]*3
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    # tol = 1e-4
    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b 
    rho_m1 = alpha = omega_m1 = 1
    # for k in 1:length(b)
    # we don't want a perfect solve, should fix this though
	k = 0
    for k in 1 : 1000
        rho_k = conj.(transpose(r0))*r_m1  
        # beta calculation
        # First term 
        first_term = rho_k/rho_m1
        # Second term 
        second_term = alpha/omega_m1
        # Calculation 
        beta = first_term*second_term
        pk = r_m1 .+ beta.*(p_m1-omega_m1.*v_m1)
        pkPrj = projVec(dim, u, pk)
        A_pkPrj = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
		    gMemSlfA,cellsA,chi_inv_coeff,P,pkPrj)
		vk = projVec(dim, u, A_pkPrj .- (theta .* pkPrj))
        # vk = projVec(dim, u, A * pkPrj .- (theta .* pkPrj))

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
        bPrj = projVec(dim, u, b) 
        A_bPrj = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
			gMemSlfA,cellsA,chi_inv_coeff,P,bPrj)
		t = projVec(dim, u, A_bPrj .- (theta .* bPrj))
        # t = projVec(dim, u, A * bPrj .- (theta .* bPrj))

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
        if norm(r_m1)-norm(r_old) < tol_bicgstab
            # print("bicgstab break \n")
            # print("real((conj.(transpose(r_m1))*r_m1)[1])",real((conj.(transpose(r_m1))*r_m1)[1]),"\n")
            # break
			return xk_m1,k # k is essentially the number of iterations 
			# Number of iterations to reach a tolerances
        end
    end
    return xk_m1,k # k is essentially the number of iterations 
end

function gramSchmidt!(basis::Array{T}, n::Integer, tol::Float64) where T <: Number

	# dimension of vector space
	dim = size(basis)[1]
	# orthogonality check
	prjNrm = 1.0;
	# check that basis does not exceed dimension
	if n > dim

		error("Requested basis size exceeds dimension of vector space.")
	end
	# norm calculation
	nrm = BLAS.nrm2(dim, view(basis,:,n), 1)
	# renormalize new vector
	basis[:,n] = basis[:,n] ./ nrm
	nrm = BLAS.nrm2(dim, view(basis,:,n), 1)
	# guarded orthogonalization
	while prjNrm > (tol * 100) && abs(nrm) > tol
		# remove projection into existing basis
 		BLAS.gemv!('N', -1.0 + im*0.0, view(basis, :, 1:(n-1)), 
 			BLAS.gemv('C', view(basis, :, 1:(n-1)), view(basis, :, n)), 
 			1.0 + im*0.0, view(basis, :, n)) 
 		# recalculate the norm
 		nrm = BLAS.nrm2(dim, view(basis,:,n), 1) 
 		# calculate projection norm
 		prjNrm = BLAS.nrm2(n-1, BLAS.gemv('C', 
 			view(basis, :, 1:(n-1)), view(basis, :, n)), 1) 
 	end
	# check that remaining vector is sufficiently large
	if abs(nrm) < tol
		# switch to random basis vector
		rand!(view(basis, :, n))
		gramSchmidt!(basis, n, tol)		
	else
		# renormalize orthogonalized vector
		basis[:,n] = basis[:,n] ./ nrm
	end 
end

function jacDavRitz_basic(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64,
			tol_eigval::Float64,tol_bicgstab::Float64)::Tuple{Float64,Float64,
				Float64}
    # jacDavRitz_basic(opt,innerLoopDim::Integer,tol_MGS::Float64,tol_conv::Float64,
    # tol_eigval::Float64,tol_bicgstab::Float64)::Tuple{Float64,Float64,
    #     Float64}
                

	# dims = size(opt)
	# vecDim = dims[1]
    dim = cellsA[1]*cellsA[2]*cellsA[3]*3

	basis = Array{ComplexF64}(undef, dim, dim)
	hesse = zeros(ComplexF64, dim, dim)

	### memory initialization
	outVec = Vector{ComplexF64}(undef, dim)
	resVec = Vector{ComplexF64}(undef, dim)
	ritzVec = Vector{ComplexF64}(undef, dim)
	# set starting vector
	rand!(view(basis, :, 1))
	# normalize starting vector
	nrm = BLAS.nrm2(dim, view(basis,:,1), 1)
	basis[:, 1] = basis[:, 1] ./ nrm
	### algorithm initialization
	outVec = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
        gMemSlfA,cellsA,chi_inv_coeff,P,basis[:, 1])
    # outVec = opt * basis[:, 1] 


	# Hessenberg matrix
	hesse[1,1] = BLAS.dotc(dim, view(basis, :, 1), 1, outVec, 1) 
	# Ritz value
	theta = hesse[1,1] 
	# Ritz vector
	ritzVec[:] = basis[:, 1]
	# Negative residual vector
	resVec = (theta .* ritzVec) .- outVec

	previous_eigval = theta
	nb_it_vals_basic = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0

	# for itr in 2 : repDim
	for itr in 2 : innerLoopDim

		# Jacobi-Davidson direction
		basis[:, itr],nb_it = ritz_bicgstab_operator(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P, theta, ritzVec, resVec, 
			tol_bicgstab)
		nb_it_total_bicgstab_solve += nb_it

		# orthogonalize
		gramSchmidt!(basis, itr, tol_MGS)
		# new image
        outVec = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
            gMemSlfA,cellsA,chi_inv_coeff,P,basis[:, itr])
		# outVec = opt * basis[:, itr] 
		
        # Update Hessenberg
		hesse[1 : itr, itr] = BLAS.gemv('C', view(basis, :, 1 : itr), outVec)
		hesse[itr, 1 : (itr - 1)] = conj(hesse[1 : (itr - 1), itr])
		# eigenvalue decomposition, largest real eigenvalue last. 
		# should replace by BLAS operation
		eigSys = eigen(view(hesse, 1 : itr, 1 : itr)) 
		# update Ritz vector
		theta = eigSys.values[end]
		ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, end])
        outVec = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
            gMemSlfA,cellsA,chi_inv_coeff,P,ritzVec)
		# outVec = opt * ritzVec
		# update residual vector
		resVec = (theta * ritzVec) .- outVec

		# Direction vector tolerance check 
		if norm(resVec) < tol_conv
			print("Basic algo converged off resVec tolerance \n")
			# print("norm(resVec) ", norm(resVec), "\n")
			return real(theta),nb_it_vals_basic,nb_it_total_bicgstab_solve #,ritzVec
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
				return real(theta),nb_it_vals_basic,nb_it_total_bicgstab_solve
			end # ,ritzVec
			nb_it_eigval_conv += 1
		end
		previous_eigval = theta
		nb_it_vals_basic += 1
	end
	print("Didn't converge off tolerance for basic program. 
		Atteined max set number of iterations \n")
	return (real(theta),nb_it_vals_basic,nb_it_total_bicgstab_solve) # ,ritzVec
end

function jacDavRitz_basic_for_restart(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,
    cellsA,chi_inv_coeff,P,innerLoopDim::Integer,tol_MGS::Float64,
        tol_conv::Float64,tol_eigval::Float64,tol_bicgstab::Float64)		

	# dims = size(opt)
	# vecDim = dims[1]
    dim = cellsA[1]*cellsA[2]*cellsA[3]*3

	basis = Array{ComplexF64}(undef, dim, dim)
	hesse = zeros(ComplexF64, dim, dim)

	### memory initialization
	outVec = Vector{ComplexF64}(undef, dim)
	resVec = Vector{ComplexF64}(undef, dim)
	ritzVec = Vector{ComplexF64}(undef, dim)
	# set starting vector
	rand!(view(basis, :, 1))
	# normalize starting vector
	nrm = BLAS.nrm2(dim, view(basis,:,1), 1)
	basis[:, 1] = basis[:, 1] ./ nrm
	### algorithm initialization
    outVec = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
        gMemSlfA,cellsA,chi_inv_coeff,P,basis[:, 1])
	# outVec = opt * basis[:, 1] 
	# Hessenberg matrix
	hesse[1,1] = BLAS.dotc(dim, view(basis, :, 1), 1, outVec, 1) 
	# Ritz value
	theta = hesse[1,1] 
	# Ritz vector
	ritzVec[:] = basis[:, 1]
	# Negative residual vector
	resVec = (theta .* ritzVec) .- outVec

	previous_eigval = theta
	nb_it_vals_basic_for_restart = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	# eigenvectors = Array{ComplexF64}(undef, innerLoopDim,restartDim)
	eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
	

	# for itr in 2 : repDim

	# for it in 1:1000
	for itr in 2 : innerLoopDim

		# Jacobi-Davidson direction
		basis[:, itr],nb_it = ritz_bicgstab_operator(alpha_0,alpha_1,xi,P_0,
            gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P, theta, ritzVec, resVec, 
			    tol_bicgstab)
		nb_it_total_bicgstab_solve += nb_it

		# orthogonalize
		gramSchmidt!(basis, itr, tol_MGS)
		# new image
        outVec = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
            gMemSlfA,cellsA,chi_inv_coeff,P,basis[:, itr])
		# outVec = opt * basis[:, itr] 

		print("outVec ", outVec, "\n")
		print("size(outVec) ", size(outVec), "\n")
		# print("outVec ", outVec, "\n")

		# Update Hessenberg
		# hesse[1 : itr, itr] = BLAS.gemv('C', view(basis, :, 1 : itr), outVec)
		hesse[1 : itr, itr] = adjoint(view(basis, :, 1 : itr))*outVec
		hesse[itr, 1 : (itr - 1)] = conj(hesse[1 : (itr - 1), itr])
		# eigenvalue decomposition, largest real eigenvalue last. 
		# should replace by BLAS operation
		eigSys = eigen(view(hesse, 1 : itr, 1 : itr)) 
		
		eigenvectors[1:itr,1:itr] = eigSys.vectors 
		eigenvalues[1:itr] = eigSys.values
		
		# update Ritz vector
		theta = eigSys.values[end]
		ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, end])
        outVec = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
            gMemSlfA,cellsA,chi_inv_coeff,P,ritzVec)
		# outVec = opt * ritzVec

		# Update residual vector
		resVec = (theta * ritzVec) .- outVec

		# Direction vector tolerance check 
		if norm(resVec) < tol_conv
			print("Basic algo converged off resVec tolerance \n")
			# print("norm(resVec) ", norm(resVec), "\n")
			return (real(theta),basis,hesse,outVec,ritzVec,resVec,eigenvalues,
				eigenvectors,nb_it_vals_basic_for_restart,nb_it_total_bicgstab_solve)
				# ,ritzVec
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
				return (real(theta),basis,hesse,outVec,ritzVec,resVec,eigenvalues,
					eigenvectors,nb_it_vals_basic_for_restart,nb_it_total_bicgstab_solve)
					# ,ritzVec
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

function jacDavRitz_restart(alpha_0,alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,
    chi_inv_coeff,P,innerLoopDim::Integer,restartDim::Integer,tol_MGS::Float64,
        tol_conv::Float64, tol_eigval::Float64,
        tol_bicgstab::Float64)::Tuple{Float64,Float64,Float64} 

	# dims = size(opt)
	# vecDim = dims[1]
    dim = cellsA[1]*cellsA[2]*cellsA[3]*3

	basis = Array{ComplexF64}(undef, dim, dim)
	hesse = zeros(ComplexF64, dim, dim)

	eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
	# eigenvectors = Array{ComplexF64}(undef, innerLoopDim,restartDim)
	eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
	
	restart_outVec = Vector{ComplexF64}(undef, dim)
	restart_resVec = Vector{ComplexF64}(undef, dim)
	restart_ritzVec = Vector{ComplexF64}(undef, dim)
	restart_basis = Array{ComplexF64}(undef, dim, restartDim)
	restart_hesse = Array{ComplexF64}(undef, dim, restartDim)
	restart_theta = 0 
	# Change of basis matrix
	u_matrix = Array{ComplexF64}(undef, innerLoopDim, restartDim)

	### memory initialization
	outVec = Vector{ComplexF64}(undef, dim)
	resVec = Vector{ComplexF64}(undef, dim)
	ritzVec = Vector{ComplexF64}(undef, dim)
	# set starting vector
	rand!(view(basis, :, 1))
	# normalize starting vector
	nrm = BLAS.nrm2(dim, view(basis,:,1), 1)
	basis[:, 1] = basis[:, 1] ./ nrm
	### algorithm initialization
    outVec = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
        gMemSlfA,cellsA,chi_inv_coeff,P,basis[:, 1])
	# outVec = opt * basis[:, 1] 

	# Hessenberg matrix
	hesse[1,1] = BLAS.dotc(dim, view(basis, :, 1), 1, outVec, 1) 
	# Ritz value
	theta = hesse[1,1] 
	# Ritz vector
	ritzVec[:] = basis[:, 1]
	# Negative residual vector
	resVec = (theta .* ritzVec) .- outVec

	previous_eigval = theta
	nb_it_restart = 0.0
	nb_it_eigval_conv = 0.0
	nb_it_total_bicgstab_solve = 0.0

	# for itr in 2 : repDim
	for it in 1 : 1000
		eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)
		# Inner loop

		if it == 1
		# 	real(theta),basis,hesse,outVec,ritzVec,resVec,eigenvalues,
		# eigenvectors,nb_it_vals_basic_for_restart,nb_it_total_bicgstab_solve
			theta,basis,hesse,outVec,ritzVec,resVec,eigenvalues,
				eigenvectors,nb_it_vals_basic_for_restart,
					nb_it_bicgstab_solve = jacDavRitz_basic_for_restart(alpha_0,
                        alpha_1,xi,P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P, 
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

			eigenvectors = Array{ComplexF64}(undef, innerLoopDim,innerLoopDim)
			eigenvalues = Vector{ComplexF64}(undef, innerLoopDim)

			basis = Array{ComplexF64}(undef, dim, dim)
			basis[:,1:restartDim] = restart_basis

			hesse = Array{ComplexF64}(undef, dim, dim)
			hesse[1:restartDim,1:restartDim] = restart_hesse

			theta = restart_theta

			for itr in restartDim+1:innerLoopDim
				# Jacobi-Davidson direction
				basis[:, itr],nb_it = ritz_bicgstab_operator(alpha_0,alpha_1,xi,
                    P_0,gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P, theta,ritzVec, 
                        resVec, tol_bicgstab)
				nb_it_total_bicgstab_solve += nb_it

				# Orthogonalize
				gramSchmidt!(basis, itr, tol_MGS)
				# New image
                outVec = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
					gMemSlfA,cellsA,chi_inv_coeff,P,basis[:, itr])
				# outVec = opt * basis[:, itr] 

				# Update Hessenberg
				# hesse[1 : itr, itr] = BLAS.gemv('C', view(basis, :, 1 : itr), outVec)
				hesse[1 : itr, itr] = adjoint(view(basis, :, 1 : itr))*outVec
				hesse[itr, 1 : (itr - 1)] = conj(hesse[1 : (itr - 1), itr])
				# eigenvalue decomposition, largest real eigenvalue last. 
				# should replace by BLAS operation
				eigSys = eigen(view(hesse, 1 : itr, 1 : itr)) 

				eigenvectors[1:itr,1:itr] = eigSys.vectors
				eigenvalues[1:itr] = eigSys.values

				# update Ritz vector
				if abs.(eigSys.values[end]) > abs.(eigSys.values[1])
					theta = eigSys.values[end]
					ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, end])
				else # For the case abs.(eigSys.values[end]) < abs.(eigSys.values[1])
					theta = eigSys.values[1]
					ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, 1])	
				end 
				
				outVec = green_vect_prod_pade(alpha_0,alpha_1,xi,P_0,gMemSlfN,
					gMemSlfA,cellsA,chi_inv_coeff,P,ritzVec)
                # outVec = opt * ritzVec

				# Update residual vector
				resVec = (theta * ritzVec) .- outVec

				# Direction vector tolerance check 
				if norm(resVec) < tol_conv
					print("Basic algo converged off resVec tolerance \n")
					# print("norm(resVec) ", norm(resVec), "\n")
					return real(theta),nb_it_restart,nb_it_total_bicgstab_solve
					# ,ritzVec
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
			# print("eigenvalues[1] ", eigenvalues[1], "\n")
			# print("eigenvalues[end] ", eigenvalues[end], "\n")
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
		# orthogonalize
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


# test
# opt = [2.0 + im*0.0  -2.0 + im*0.0  0.0 + im*0.0;
# 	-1.0 + im*0.0  3.0 + im*0.0  -1.0 + im*0.0;
# 	0.0 + im*0.0  -1.0 + im*0.0  4.0 + im*0.0]

# RND matrix
# sz = 10
# opt = Array{ComplexF64}(undef,sz,sz)
# rand!(opt)
# opt[:,:] = (opt .+ conj(transpose(opt))) ./ 2

# dims = size(opt)

# basis = Array{ComplexF64}(undef, dims[1], dims[2])
# hesse = zeros(ComplexF64, dims[2], dims[2])


# innerLoopDim = 5
# restartDim = 1
# tol_MGS = 1.0e-12
# tol_conv = 1.0e-12
# tol_eigval = 1.0e-9
# tol_bicgstab = 1e-6

# (val_basic, vec_basic) = jacDavRitz_basic(opt,
# 	innerLoopDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
# print("Ritz eigval basic ", val_basic, "\n")
# # print("Ritz eigvec basic ", vec_basic, "\n")
# (val_restart, vec_restart) = jacDavRitz_restart(opt,
# 	innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bicgstab)
# print("Ritz eigval restart ", val_restart, "\n")
# # print("Ritz eigvec restart ", vec_restart, "\n")

# trueEig = eigen(opt) 
# julia_eigval = trueEig.values[end]
# julia_eigvec = trueEig.vectors[:,end]
# print("Julia largest eigval ", julia_eigval, "\n")
# print("Julia largest eigvec ", julia_eigvec, "\n")

end