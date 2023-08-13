using LinearAlgebra, Random, Base.Threads, Plots, Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson

"""
1. For different bicgstab tolerences
"""
# Varibles setup
sz = 15
opt = Array{ComplexF64}(undef,sz,sz)
rand!(opt)
opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
trueEigSys = eigen(opt)
minEigPos = argmin(abs.(trueEigSys.values))
julia_min_eigval = trueEigSys.values[minEigPos]
dims = size(opt)
innerLoopDim = 10
restartDim = 8 
max_nb_it = 1000
tol_MGS = 1.0e-12
tol_conv = 1.0e-6
tol_eigval = 1.0e-9
basis_solver = "bicgstab"
tol_bicgstab_vals = [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12] 

# Memory allocation
nb_it_vals_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
nb_it_vals_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))
eigval_julia_vals = Vector{Float64}(undef, length(tol_bicgstab_vals))
eigval_basic_vals = Vector{Float64}(undef, length(tol_bicgstab_vals))
eigval_restart_vals = Vector{Float64}(undef, length(tol_bicgstab_vals))
eigval_sums_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
eigval_sums_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
nb_it_sums_basic = Vector{Int32}(undef, length(tol_bicgstab_vals))
nb_it_sums_restart = Vector{Int32}(undef, length(tol_bicgstab_vals))
for loops = 1:10
	@threads for index = 1:length(tol_bicgstab_vals)
		print("index ", index, "\n")
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])
		print("Start of basic program \n")
		eigval_basic_vals[index],nb_it_vals_basic[index] = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_bicgstab_vals[index], max_nb_it,
                    basis_solver)
		print("Start of restart program \n")
		eigval_restart_vals[index], nb_it_vals_restart[index] = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab_vals[index],max_nb_it,basis_solver)
		print("eigval_restart_vals[index] ", eigval_restart_vals[index], "\n")
		print("nb_it_vals_restart[index] ", nb_it_vals_restart[index], "\n")
	end 
	@threads for index = 1:length(tol_bicgstab_vals)
		eigval_sums_basic[index] += eigval_basic_vals[index]
		eigval_sums_restart[index] += eigval_restart_vals[index]
		nb_it_sums_basic[index] += nb_it_vals_basic[index]
		nb_it_sums_restart[index] += nb_it_vals_restart[index]
	end 
end

avg_eigvals_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
avg_eigvals_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
avg_nb_it_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
avg_nb_it_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
@threads for index = 1:length(tol_bicgstab_vals)
	avg_eigvals_basic[index] += eigval_basic_vals[index]/length(eigval_basic_vals)
	avg_eigvals_restart[index] += eigval_restart_vals[index]/length(eigval_restart_vals)
	avg_nb_it_basic[index] += nb_it_vals_basic[index]/length(nb_it_vals_basic)
	avg_nb_it_restart[index] += nb_it_vals_restart[index]/length(nb_it_vals_restart)
end 

rel_diff_eigval_basic = Vector{Float64}(undef, length(tol_bicgstab_vals))
rel_diff_eigval_restart = Vector{Float64}(undef, length(tol_bicgstab_vals))
@threads for index = 1:length(tol_bicgstab_vals)
	rel_diff_eigval_basic[index] += abs((avg_eigvals_basic[index]-julia_min_eigval)/julia_min_eigval)*100
	rel_diff_eigval_restart[index] += abs((avg_eigvals_restart[index]-julia_min_eigval)/julia_min_eigval)*100
end 

print("tol_bicgstab_vals", tol_bicgstab_vals, "\n")

print("Basic algo - HarmonicRitz number of iterations is ", avg_nb_it_basic, "\n")
print("Restart algo - HarmonicRitz number of iterations is ", avg_nb_it_restart, "\n")

print("No restart - HarmonicRitz smallest positive eigenvalue is ", avg_eigvals_basic, "\n")
print("Restart - HarmonicRitz smallest positive eigenvalue is ", avg_eigvals_restart, "\n")
println("Julia smallest positive eigenvalue is ", julia_min_eigval,"\n")

print("Basic algo - HarmonicRitz relative difference between eigvals are ", rel_diff_eigval_basic, "\n")
print("Restart algo - HarmonicRitz relative difference between eigvals are  ", rel_diff_eigval_restart, "\n")

println("Julia eigenvalues ", trueEigSys.values,"\n")


""" 1. Number of iterations vs bicgstab tolerance graphs """
# Graph of the number of iterations to converge for the basic Harmonic Ritz 
# program.
plot(tol_bicgstab_vals,avg_nb_it_basic,title="Number of iterations vs bicgstab tolerance \n basic Harm Ritz")
xlabel!("Bicgstab tolerance values")
ylabel!("Number of iterations")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing bicgstab tolerance/tol_bicgstab_basic_vs_nb_it.png")

# Graph of the number of iterations to converge for the restart Harmonic Ritz 
# program.
plot(tol_bicgstab_vals,avg_nb_it_restart,title="Number of iterations vs bicgstab tolerance \n restart Harm Ritz")
xlabel!("Bicgstab tolerance values")
ylabel!("Number of iterations")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing bicgstab tolerance/tol_bicgstab_restart_vs_nb_it.png")

""" 2. Relative difference vs bicgstab tolerance graphs """
# Graph of the relative different between the eigvals vs bicgstab tolerance 
# for the basic Harmonic Ritz program.
plot(tol_bicgstab_vals,rel_diff_eigval_basic,title="Eigvals relative different vs bicgstab tolerance \n basic Harm Ritz")
xlabel!("Bicgstab tolerance values")
ylabel!("Eigvals relative difference with Julia solve")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing bicgstab tolerance/tol_bicgstab_basic_vs_eigvals_relative_diff.png")

# Graph of the relative different between the eigvals vs bicgstab tolerance 
# for the restart Harmonic Ritz program.
plot(tol_bicgstab_vals,rel_diff_eigval_restart,title="Eigvals relative different vs bicgstab tolerance \n restart Harm Ritz")
xlabel!("Bicgstab tolerance values")
ylabel!("Number of iterations")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing bicgstab tolerance/tol_bicgstab_restart_vs_eigvals_relative_diff.png")



"""
2. For different matrix/operator sizes
"""
# Variable setup
max_nb_it = 1000
innerLoopDim = 25
restartDim = 20
operatorDim_vals = [30,35,40,45,50]
basis_solver = "bicgstab"
tol_MGS = 1.0e-12
tol_conv = 1.0e-6
tol_eigval = 1.0e-9
tol_bicgstab = 1.0e-9

# Memory allocation
nb_it_vals_basic = Vector{Int32}(undef, length(operatorDim_vals))
nb_it_vals_restart = Vector{Int32}(undef, length(operatorDim_vals))
eigval_julia_vals = Vector{Float64}(undef, length(operatorDim_vals))
eigval_basic_vals =  Vector{Float64}(undef, length(operatorDim_vals))
eigval_restart_vals = Vector{Float64}(undef, length(operatorDim_vals))
eigval_sums_basic = Vector{Float64}(undef, length(operatorDim_vals))
eigval_sums_restart = Vector{Float64}(undef, length(operatorDim_vals))
nb_it_sums_basic = Vector{Int32}(undef, length(operatorDim_vals))
nb_it_sums_restart = Vector{Int32}(undef, length(operatorDim_vals))
all_julia_eigvals = Vector{Vector}(undef, length(operatorDim_vals))

for loops = 1:10
	@threads for index = 1:length(operatorDim_vals)
		opt = Array{ComplexF64}(undef,operatorDim_vals[index],operatorDim_vals[index])
		rand!(opt)

		opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
		trueEigSys = eigen(opt)
		minEigPos = argmin(abs.(trueEigSys.values))
		eigval_julia_vals[index] = trueEigSys.values[minEigPos]

		all_julia_eigvals[index] = trueEigSys.values

		dims = size(opt)

		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])
		print("Start of basic program \n")
		eigval_basic_vals[index],nb_it_vals_basic[index] = 
			jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
				dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_bicgstab,max_nb_it,
                    basis_solver)

		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])
		print("Start of restart program \n")
		dims = size(opt)
		eigval_restart_vals[index], nb_it_vals_restart[index] = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("eigval_restart_vals[index] ", eigval_restart_vals[index], "\n")
		print("nb_it_vals_restart[index] ", nb_it_vals_restart[index], "\n")
	end 
	@threads for index = 1:length(operatorDim_vals)
		eigval_sums_basic[index] += eigval_basic_vals[index]
		eigval_sums_restart[index] += eigval_restart_vals[index]
		nb_it_sums_basic[index] += nb_it_vals_basic[index]
		nb_it_sums_restart[index] += nb_it_vals_restart[index]
	end 
end

avg_eigvals_basic = Vector{Float64}(undef, length(operatorDim_vals))
avg_eigvals_restart = Vector{Float64}(undef, length(operatorDim_vals))
avg_nb_it_basic = Vector{Float64}(undef, length(operatorDim_vals))
avg_nb_it_restart = Vector{Float64}(undef, length(operatorDim_vals))
@threads for index = 1:length(operatorDim_vals)
	avg_eigvals_basic[index] += eigval_basic_vals[index]/length(eigval_basic_vals)
	avg_eigvals_restart[index] += eigval_restart_vals[index]/length(eigval_restart_vals)
	avg_nb_it_basic[index] += nb_it_vals_basic[index]/length(nb_it_vals_basic)
	avg_nb_it_restart[index] += nb_it_vals_restart[index]/length(nb_it_vals_restart)
end 

rel_diff_eigval_basic = Vector{Float64}(undef, length(operatorDim_vals))
rel_diff_eigval_restart = Vector{Float64}(undef, length(operatorDim_vals))
@threads for index = 1:length(operatorDim_vals)
	rel_diff_eigval_basic[index] += abs((avg_eigvals_basic[index]-eigval_julia_vals[index])/eigval_julia_vals[index])*100
	rel_diff_eigval_restart[index] += abs((avg_eigvals_restart[index]-eigval_julia_vals[index])/eigval_julia_vals[index])*100
end 

print("operatorDim_vals", operatorDim_vals, "\n")

print("Basic algo - HarmonicRitz number of iterations is ", avg_nb_it_basic, "\n")
print("Restart algo - HarmonicRitz number of iterations is ", avg_nb_it_restart, "\n")

print("No restart - HarmonicRitz smallest positive eigenvalues are ", avg_eigvals_basic, "\n")
print("Restart - HarmonicRitz smallest average positive eigenvalues are ", avg_eigvals_restart, "\n")
println("Julia smallest positive eigenvalues are ", eigval_julia_vals,"\n")

print("Basic algo - HarmonicRitz relative difference between eigvals are ", rel_diff_eigval_basic, "\n")
print("Restart algo - HarmonicRitz relative difference between eigvals are  ", rel_diff_eigval_restart, "\n")

println("Julia eigenvalues for the different operator sizes ", all_julia_eigvals,"\n")


""" 1. Number of iterations vs operator size graphs """
# Graph of the number of iterations to converge for the basic Harmonic Ritz 
# program.
plot(operatorDim_vals,avg_nb_it_basic,title="Number of iterations vs bicgstab tolerance \n basic Harm Ritz")
xlabel!("Operator size")
ylabel!("Number of iterations")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing operator size/operator_size_basic_vs_nb_it.png")

# Graph of the number of iterations to converge for the restart Harmonic Ritz 
# program.
plot(operatorDim_vals,avg_nb_it_restart,title="Number of iterations vs bicgstab tolerance \n restart Harm Ritz")
xlabel!("Operator size")
ylabel!("Number of iterations")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing operator size/operator_size_restart_vs_nb_it.png")

""" 2. Relative difference vs bicgstab tolerance graphs """
# Graph of the relative different between the eigvals vs bicgstab tolerance 
# for the basic Harmonic Ritz program.
plot(operatorDim_vals,rel_diff_eigval_basic,title="Eigvals relative different vs bicgstab tolerance \n basic Harm Ritz")
xlabel!("Operator size")
ylabel!("Eigvals relative difference with Julia solve")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing operator size/operator_size_basic_vs_eigvals_relative_diff.png")

# Graph of the relative different between the eigvals vs bicgstab tolerance 
# for the restart Harmonic Ritz program.
plot(operatorDim_vals,rel_diff_eigval_restart,title="Eigvals relative different vs bicgstab tolerance \n restart Harm Ritz")
xlabel!("Operator size")
ylabel!("Eigvals relative difference with Julia solve")
savefig("/home/nic-molesky_lab/Github-Research-2023/Davidson iteration  Code and Articles/Graphs/Changing operator size/operator_size_restart_vs_eigvals_relative_diff.png")




"""
3. For different restart sizes
This is only possible for the restart version of the harmonic Ritz 
code. The different variables for the basic program are therefore 
commented out.
"""
# Variable setup
sz = 30
opt = Array{ComplexF64}(undef,sz,sz)
rand!(opt)
opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
trueEigSys = eigen(opt)
minEigPos = argmin(abs.(trueEigSys.values))
julia_min_eigval = trueEigSys.values[minEigPos]
dims = size(opt)
max_nb_it = 1000
innerLoopDim = 25
restartDim_vals = [5,10,15,20]
tol_MGS = 1.0e-12
tol_conv = 1.0e-6
tol_eigval = 1.0e-9
tol_bicgstab = 1.0e-9
basis_solver = "bicgstab"

# Memory allocation
nb_it_basic = 0
nb_it_vals_restart = Vector{Int32}(undef, length(restartDim_vals))
eigval_julia_vals = Vector{Float64}(undef, length(restartDim_vals))
eigval_basic = 0.0
eigval_restart_vals = Vector{Float64}(undef, length(restartDim_vals))
# eigval_sums_basic = Vector{Float64}(undef, length(restartDim_vals))
eigval_sums_restart = Vector{Float64}(undef, length(restartDim_vals))
# nb_it_sums_basic = Vector{Int32}(undef, length(restartDim_vals))
nb_it_sums_restart = Vector{Int32}(undef, length(restartDim_vals))

print("Start of basic program \n")
bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
kMat = zeros(ComplexF64, dims[2], dims[2])
eigval_basic,nb_it_vals_basic = 
	jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],
		dims[2] , innerLoopDim, tol_MGS, tol_conv,tol_bicgstab,max_nb_it)

for loops = 1:10
	@threads for index = 1:length(restartDim_vals)
		print("index ", index, "\n")
		bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
		bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
		trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
		kMat = zeros(ComplexF64, dims[2], dims[2])
		print("Start of restart program \n")
		eigval_restart_vals[index], nb_it_vals_restart[index] = 
			jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
				dims[2],innerLoopDim,restartDim_vals[index],tol_MGS,tol_conv,tol_eigval,
					tol_bicgstab,max_nb_it,basis_solver)
		print("eigval_restart_vals[index] ", eigval_restart_vals[index], "\n")
		print("nb_it_vals_restart[index] ", nb_it_vals_restart[index], "\n")
	end 
	@threads for index = 1:length(restartDim_vals)
		# eigval_sums_basic[index] += eigval_basic_vals[index]
		eigval_sums_restart[index] += eigval_restart_vals[index]
		# nb_it_sums_basic[index] += nb_it_vals_basic[index]
		nb_it_sums_restart[index] += nb_it_vals_restart[index]
	end 
end

# avg_eigvals_basic = Vector{Float64}(undef, length(restartDim_vals))
avg_eigvals_restart = Vector{Float64}(undef, length(restartDim_vals))
# avg_nb_it_basic = Vector{Float64}(undef, length(restartDim_vals))
avg_nb_it_restart = Vector{Float64}(undef, length(restartDim_vals))
@threads for index = 1:length(restartDim_vals)
	# avg_eigvals_basic[index] += eigval_basic_vals[index]/length(eigval_basic_vals)
	avg_eigvals_restart[index] += eigval_restart_vals[index]/length(eigval_restart_vals)
	# avg_nb_it_basic[index] += nb_it_vals_basic[index]/length(nb_it_vals_basic)
	avg_nb_it_restart[index] += nb_it_vals_restart[index]/length(nb_it_vals_restart)
end 

rel_diff_eigval_basic = abs((eigval_basic-julia_min_eigval)/julia_min_eigval)*100
rel_diff_eigval_restart = Vector{Float64}(undef, length(restartDim_vals))
@threads for index = 1:length(restartDim_vals)
	# rel_diff_eigval_basic[index] += abs((avg_eigvals_basic[index]-julia_min_eigval)/julia_min_eigval)*100
	rel_diff_eigval_restart[index] += abs((avg_eigvals_restart[index]-julia_min_eigval)/julia_min_eigval)*100
end 

print("restartDim_vals", restartDim_vals, "\n")

print("Basic algo - HarmonicRitz number of iterations is ", nb_it_basic, "\n")
print("Restart algo - HarmonicRitz number of iterations is ", avg_nb_it_restart, "\n")

print("No restart - HarmonicRitz smallest positive eigenvalue is ", eigval_basic, "\n")
print("Restart - HarmonicRitz smallest average positive eigenvalue is ", avg_eigvals_restart, "\n")
println("Julia smallest positive eigenvalue is ", julia_min_eigval,"\n")

print("Basic algo - HarmonicRitz relative difference between eigvals are ", rel_diff_eigval_basic, "\n")
print("Restart algo - HarmonicRitz relative difference between eigvals are  ", rel_diff_eigval_restart, "\n")

println("Julia eigenvalues ", trueEigSys.values,"\n")


""" 1. Number of iterations vs restart size graphs """
# Graph of the number of iterations to converge for the restart Harmonic Ritz 
# program.
plot(restartDim_vals,avg_nb_it_restart,title="Number of iterations vs restart size \n restart Harm Ritz")
xlabel!("Restart size")
ylabel!("Number of iterations")
# Need to change path to save graph on your computer
savefig("Davidson iteration  Code and Articles\Graphs\Changing restart size/restart_size_vs_nb_it.png")

""" 2. Relative difference vs restart size graphs """
# Graph of the number of iterations to converge for the restart Harmonic Ritz 
# program.
plot(restartDim_vals,rel_diff_eigval_restart ,title="Relative difference vs restart size \n restart Harm Ritz")
xlabel!("Restart size")
ylabel!("Relative difference")
# Need to change path to save graph on your computer
savefig("Davidson iteration  Code and Articles\Graphs\Changing restart size/restart_size_vs_relative_diff.png")

