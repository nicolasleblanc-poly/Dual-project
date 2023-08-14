"""
This module contains code to test the convergence of the harmonic 
Ritz algorithm for different operator sizes with varying 
bicgstab tolerances. The same graphs could be done for 
different restart sizes, residual vector tolerances, 
eigenvalue convergence tolerances, etc. 

Author: Nicolas Leblanc
"""

using LinearAlgebra, Random, Base.Threads, Plots, 
	Restart_Ritz_jacobiDavidson,Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson

# Setup 
operatorDim_vals = [50,100,150,200]
opt = Array{ComplexF64}(undef,operatorDim_vals[1],operatorDim_vals[1])
rand!(opt)
opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
trueEigSys = eigen(opt)
minEigPos = argmin(abs.(trueEigSys.values))
julia_min_eigval = trueEigSys.values[minEigPos]
max_nb_it = 1000
innerLoopDim = 25
restartDim = 3
tol_MGS = 1.0e-12
tol_conv = 1.0e-6
tol_eigval = 1.0e-9
tol_bicgstab_vals = [1e-2,1e-3,1e-4,1e-5,1e-6]

# Memory allocation 
dims = size(opt)
bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
kMat = zeros(ComplexF64, dims[2], dims[2])
bicgstab_tol_eigvals_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
bicgstab_tol_nb_it_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
bicgstab_tol_eigvals_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
bicgstab_tol_nb_it_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))

basis_solver = "bicgstab"
# basis_solver = "direct"

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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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

basis_solver = "direct"  
# basis_solver = "bicgstab"
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
# 			tol_bicgstab,basis_solver)
# print("Start of restart program \n")
# eigval_restart,nb_it_restart  = 
# 	jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
# 		dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
# 			tol_bicgstab,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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

basis_solver = "bicgstab"
# basis_solver = "direct"

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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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

basis_solver = "direct"  
# basis_solver = "bicgstab"
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
			jacDavRitzHarm_basic(opt,innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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

basis_solver = "bicgstab"
# basis_solver = "direct"

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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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

basis_solver = "direct"  
# basis_solver = "bicgstab"
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
			jacDavRitzHarm_basic(opt, innerLoopDim, tol_MGS, tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("Start of restart program \n")
		eigval_restart,nb_it_restart  = 
			jacDavRitzHarm_restart(opt,innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
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
savefig("Pade_Ritz_HarmonicRitz_algorithms/Combining Padé, Ritz and harmonic Ritz/Matrix version/Graphs/Nb of iterations to conv vers operator size for different bicgstab tolerences/nb_it_vs_operator_size_DiffBicgstabTolerances_innerLoopDim25_DiffRestartDim3_10_20.png")


plot(operatorDim_vals, [first_bicgstab_tol_nb_it_restart second_bicgstab_tol_nb_it_restart third_bicgstab_tol_nb_it_restart fourth_bicgstab_tol_nb_it_restart fifth_bicgstab_tol_nb_it_restart six_direct_solve_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 ",
		label=["bicgstab_tol=1e-2,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=3,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=3,conv_tol=1e-6" "direct solve,restartDim=3,conv_tol=1e-6"]
		, legend = :outertopleft) # 
		# , linecolor=[:red,:blue,:cyan,:green,:indigo,:midnightblue,:purple,:plum,:orchid,:maroon,:pink,:tan1,:brown,:gray47,:black,:gold,:yellow,:salmon]
		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
plot!(size=(2500,1000))
xlabel!("Operator size")
ylabel!("Number of iterations to converge")
savefig("Pade_Ritz_HarmonicRitz_algorithms/Combining Padé, Ritz and harmonic Ritz/Matrix version/Graphs/Nb of iterations to conv vers operator size for different bicgstab tolerences/nb_it_vs_operator_size_DiffBicgstabTolerances_innerLoopDim25_DiffRestartDim3.png")

plot(operatorDim_vals, [seven_bicgstab_tol_nb_it_restart eight_bicgstab_tol_nb_it_restart nine_bicgstab_tol_nb_it_restart ten_bicgstab_tol_nb_it_restart eleven_bicgstab_tol_nb_it_restart twelve_direct_solve_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 ",
		label=["bicgstab_tol=1e-2,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=10,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6" "direct solve,restartDim=10,conv_tol=1e-6"]
		, legend = :outertopleft) # 
		# , linecolor=[:red,:blue,:cyan,:green,:indigo,:midnightblue,:purple,:plum,:orchid,:maroon,:pink,:tan1,:brown,:gray47,:black,:gold,:yellow,:salmon]
		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
plot!(size=(2500,1000))
xlabel!("Operator size")
ylabel!("Number of iterations to converge")
savefig("Pade_Ritz_HarmonicRitz_algorithms/Combining Padé, Ritz and harmonic Ritz/Matrix version/Graphs/Nb of iterations to conv vers operator size for different bicgstab tolerences/nb_it_vs_operator_size_DiffBicgstabTolerances_innerLoopDim25_DiffRestartDim10.png")

plot(operatorDim_vals, [thirteen_bicgstab_tol_nb_it_restart fourteen_bicgstab_tol_nb_it_restart fifteen_bicgstab_tol_nb_it_restart sixteen_bicgstab_tol_nb_it_restart seventeen_bicgstab_tol_nb_it_restart eighteen_direct_solve_nb_it_restart] , #  1e-4,1e-5,1e-7,1e-9
	title="Number of iterations to converge vs operator size \n for different bicgstab tolerances-restart Harm Ritz \n using innerLoopDim=25 and restartDim=3 ",
		label=["bicgstab_tol=1e-2,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-3,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-4,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-5,restartDim=20,conv_tol=1e-6" "bicgstab_tol=1e-6,restartDim=20,conv_tol=1e-6" "direct solve,restartDim=20,conv_tol=1e-6"]
		, legend = :outertopleft) # 
		# , linecolor=[:red,:blue,:cyan,:green,:indigo,:midnightblue,:purple,:plum,:orchid,:maroon,:pink,:tan1,:brown,:gray47,:black,:gold,:yellow,:salmon]
		# "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-2" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-3" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-4" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-5" "bicgstab_tol=1e-6,restartDim=10,conv_tol=1e-6"
plot!(size=(2500,1000))
xlabel!("Operator size")
ylabel!("Number of iterations to converge")
savefig("Pade_Ritz_HarmonicRitz_algorithms/Combining Padé, Ritz and harmonic Ritz/Matrix version/Graphs/Nb of iterations to conv vers operator size for different bicgstab tolerences/nb_it_vs_operator_size_DiffBicgstabTolerances_innerLoopDim25_DiffRestartDim20.png")

