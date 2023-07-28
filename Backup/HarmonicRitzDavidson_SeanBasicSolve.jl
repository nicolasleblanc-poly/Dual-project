using LinearAlgebra, Random
function jacDavRitzHarm(trgBasis::Array{ComplexF64}, srcBasis::Array{ComplexF64}, 
	kMat::Array{ComplexF64}, opt::Array{ComplexF64}, vecDim::Integer, 
	repDim::Integer, tol_bicgstab::Float64, tol_MGS::Float64,max_nb_it::Int64)::Float64
	### memory initialization
	resVec = Vector{ComplexF64}(undef, vecDim)
	hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	hRitzSrc = Vector{ComplexF64}(undef, vecDim)
	bCoeffs1 = Vector{ComplexF64}(undef, repDim)
	bCoeffs2 = Vector{ComplexF64}(undef, repDim)
	# set starting vector
	rand!(view(srcBasis, :, 1))
	# normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(srcBasis,:,1), 1)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm
	### algorithm initialization
	trgBasis[:, 1] = opt * srcBasis[:, 1]
	nrm = BLAS.nrm2(vecDim, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm
	# representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(vecDim, view(trgBasis, :, 1), 1,
		view(srcBasis, :, 1), 1)
	# Ritz value
	eigPos = 1
	theta = 1 / kMat[1,1]
	# Ritz vectors
	hRitzTrg[:] = trgBasis[:, 1]
	hRitzSrc[:] = srcBasis[:, 1]
	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg
	for itr in 2 : repDim
		prjCoeff = BLAS.dotc(vecDim, hRitzTrg, 1, hRitzSrc, 1)
		# calculate Jacobi-Davidson direction
		srcBasis[:, itr] = bicgstab_matrix(opt, theta, hRitzTrg,
			hRitzSrc, prjCoeff, resVec,tol_bicgstab,max_nb_it)
		trgBasis[:, itr] = opt * srcBasis[:, itr]
		# orthogonalize
		gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, opt,
			itr, tol_MGS)
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

		# Tolerance check here
		if norm(resVec) < 1e-4 # tol_conv
			print("Converged off tolerance \n")
			return real(theta) 
			# println(real(theta))
		end

		# # add tolerance check here
		# if mod(itr,32) == 0
			
		# 	println(real(theta))
		# end
	end
	return real(theta)
end
# perform Gram-Schmidt on target basis, adjusting source basis accordingly
function gramSchmidtHarm!(trgBasis::Array{T}, srcBasis::Array{T},
	bCoeffs1::Vector{T}, bCoeffs2::Vector{T}, opt::Array{T}, n::Integer,
	tol::Float64) where T <: Number
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
		trgBasis[:, n] = opt * srcBasis[:, n]
		gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2,
			opt, n, tol)
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

# New july 13th
# function bicgstab_matrix(A, theta, hTrg, hSrc, prjC, b,tol_bicgstab,max_nb_it)
# 	dim = size(A)[1]
#     xk = Vector{Float64}(undef, dim)
#     rand!(xk)
#     rk = b-A*xk
#     r0_hat = Vector{Float64}(undef, dim)
#     rand!(r0_hat)
#     print("dot(r0_hat,rk_m1) ", dot(r0_hat,rk),"\n")
#     pk = rk
#     alpha = 1
#     nb_it = 0

# 	for k in 1:max_nb_it
#         # alpha calculation
# 		pkPrj = harmVec(dim, hTrg, hSrc, prjC, pk)
#         A_pk = harmVec(dim, hTrg, hSrc, prjC,A * pkPrj .- (theta .* pkPrj))
#         # A_pk = A*pk
#         alpha_top_term = dot(r0_hat,rk)
#         alpha_bottom_term = dot((A_pk),r0_hat)
#         alpha = alpha_top_term/alpha_bottom_term

#         s = rk-alpha*A_pk

#         # omega calculation 
# 		sPrj = harmVec(dim, hTrg, hSrc, prjC, s)
#         A_s = harmVec(dim, hTrg, hSrc, prjC,
#         	A * sPrj .- (theta .* sPrj))
#         # A_s = A*s
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
#     # print("xk ", xk, "\n")
#     # print("nb_it ", nb_it, "\n")
#     # print("Reached max number of iterations \n")
#     # print("Restart a new function call \n")
#     # xk = bicgstab_matrix(A, theta, hTrg, hSrc, prjC, b,tol_bicgstab,max_nb_it)
#     return xk

# end

# Test
function bicgstab_matrix(A, theta, hTrg, hSrc, prjC, b,tol_bicgstab,max_nb_it)
	dim = size(A)[1]
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b 
    rho_m1 = alpha = omega_m1 = 1
	k = 0
    for k in 1 : 1000
        rho_k = dot(r0,r_m1) # conj.(transpose(r0))*r_m1  
        # Bêta calculation
        # First term 
        first_term = rho_k/rho_m1
        # Second term 
        second_term = alpha/omega_m1
        # Calculation 
        beta = first_term*second_term
        pk = r_m1 .+ beta.*(p_m1-omega_m1.*v_m1)
        pkPrj = harmVec(dim, hTrg, hSrc, prjC, pk)
        vk = harmVec(dim, hTrg, hSrc, prjC,A * pkPrj .- (theta .* pkPrj))
        # alpha calculation
        # Bottom term
        bottom_term = dot(r0,vk) # conj.(transpose(r0))*vk
        # Calculation 
        alpha = rho_k / bottom_term 
        h = xk_m1 + alpha.*pk
        s = r_m1 - alpha.*vk
        sPrj = harmVec(dim, hTrg, hSrc, prjC, s)
        t = harmVec(dim, hTrg, hSrc, prjC,A * sPrj .- (theta .* sPrj))
        # omega_k calculation 
        # Top term 
        ts = dot(t,s) # conj.(transpose(t))*s
        # Bottom term
        tt = dot(t,t) # conj.(transpose(t))*t
        # Calculation 
        omega_k = ts ./ tt
        xk_m1 = h + omega_k.*s
        r_old = r_m1
        r_m1 = s-omega_k.*t
		rho_m1 = rho_k
		print("norm(r_m1) ", norm(r_m1), "\n")
        if norm(r_m1) < tol_bicgstab
			return xk_m1 # k is essentially the number of iterations 
			# to reach the chosen tolerance
        end
    end
	print("Didn't converge off tolerence \n")
    return xk_m1 # k is essentially the number of iterations 

end

# Super old 
# function bicgstab_matrix(A, theta, hTrg, hSrc, prjC, b,tol_bicgstab,max_nb_it)
#     dim = length(b)
# 	max_nb_it = 10000
# 	tol_bicgstab = 1e-4
#     xk_m1 = Vector{Float64}(undef, dim)
#     rand!(xk_m1)
#     rk_m1 = b-A*xk_m1
#     r0_hat = Vector{Float64}(undef, dim)
#     rand!(r0_hat)
#     # print("dot(r0_hat,rk_m1) ", dot(r0_hat,rk_m1),"\n")
#     pk_m1 = rk_m1
#     alpha = 1
#     nb_it = 0
#     for k in 1:max_nb_it
#         # alpha calculation 
#         # A_pk_m1 = A*pk_m1
#         pkPrj = harmVec(dim, hTrg, hSrc, prjC, pk_m1)
#         A_pk_m1 = harmVec(dim, hTrg, hSrc, prjC,A * pkPrj .- (theta .* pkPrj))

#         alpha_top_term = dot(r0_hat,rk_m1)
#         alpha_bottom_term = dot((A_pk_m1),r0_hat)
#         alpha = alpha_top_term/alpha_bottom_term

#         s = rk_m1-alpha*A_pk_m1

#         # omega calculation 
#         # A_s = A*s
# 		sPrj = harmVec(dim, hTrg, hSrc, prjC, s)
#         A_s = harmVec(dim, hTrg, hSrc, prjC, A * sPrj .- (theta .* sPrj))
#         omega_top_term = dot(A_s,s)
#         omega_bottom_term = dot(A_s,A_s)
#         omega = omega_top_term/omega_bottom_term

#         xk_m1 = xk_m1 + alpha*pk_m1 + omega*s

#         rk = s - omega*A_s

#         # if norm(rk)/norm(b) < tol_bicgstab
# 		print("norm(rk) ", norm(rk), "\n")
#         if norm(rk) < tol_bicgstab
# 			# print("rk ", rk, "\n")
#             print("norm(rk) ", norm(rk), "\n")
# 			return xk_m1 #,nb_it 
#         end
#         # bêta calculation 
#         beta = (alpha/omega)*(dot(rk,r0_hat)/dot(rk_m1,r0_hat))
#         pk_m1 = rk + beta*(pk_m1-omega*A_pk_m1)
#         rk_m1 = rk
#         nb_it += 1
#     end
#     # print("xk_m1 ", xk_m1, "\n")
#     # print("nb_it ", nb_it, "\n")
#     print("Reached max number of iterations \n")
#     # print("Restart a new function call \n")
#     # xk_m1,nb_it = bicgstab_matrix(A, theta, hTrg, hSrc, prjC, b)
#     return xk_m1 #,nb_it
# end

### testing
# opt = [8.0 + im*0.0  -3.0 + im*0.0  2.0 + im*0.0;
# 	-1.0 + im*0.0  3.0 + im*0.0  -1.0 + im*0.0;
# 	1.0 + im*0.0  -1.0 + im*0.0  4.0 + im*0.0]
sz = 256
tol_MGS = 1e-9
tol_bicgstab = 1e-4
tol_cg = 1e-4
max_nb_it = 1000
opt = Array{ComplexF64}(undef,sz,sz)
rand!(opt)
opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
trueEigSys = eigen(opt)
minEigPos = argmin(abs.(trueEigSys.values))
minEig = trueEigSys.values[minEigPos]
dims = size(opt)
# print("dims[1] ", dims[1], "\n")
# print("dims[2] ", dims[2], "\n")
bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
kMat = zeros(ComplexF64, dims[2], dims[2])
# val = jacDavRitzHarm(trgBasis, srcBasis, kMat, opt, dims[1], dims[2], 1.0e-6)
val = jacDavRitzHarm(trgBasis, srcBasis, kMat, opt, dims[1], 256, tol_bicgstab,tol_MGS,max_nb_it)
print("Harmonic Ritz eigenvalue closest to 0 is ", val, "\n")
println("The Julia smallest eigenvalue is ", minEig,".")
