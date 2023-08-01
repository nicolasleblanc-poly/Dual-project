"""
The MaxGOpr module provides support functions for purely CPU implementation of 
the embedded circulant form Green function calculated by MaxGCirc. Presently, 
this module exists solely for testing purposes and no documentation is provided.
The code is distributed under GNU LGPL.

Author: Sean Molesky 
"""
module MaxGOpr
using Base.Threads, AbstractFFTs, MaxGStructs, MaxGCirc
export grnOpr!, MaxGOprGenExt, MaxGOprGenSlf, blockGreenItr
"""

	function MaxGOprGenExt(trgDom::MaxGDom, srcDom::MaxGDom)::MaxGOprMem

Prepare memory for Green function operator between a pair of distinct domains. 
"""
function MaxGOprGenExt(adjMode::Int64, assemblyInfo::MaxGAssemblyOpts, 
	trgDom::MaxGDom, srcDom::MaxGDom)::MaxGOprMem
	# Create MaxG volumes
	srcVol = genMaxGVol(srcDom)
	trgVol = genMaxGVol(trgDom)
	# Pre-allocate memory for circulant green function vector
	greenCirc = Array{ComplexF64}(undef, 3, 3, srcVol.cells[1] + trgVol.cells[1],
		srcVol.cells[2] + trgVol.cells[2], srcVol.cells[3] + trgVol.cells[3])
	# Generate circulant Green function
	genGreenExt!(greenCirc, trgVol, srcVol, assemblyInfo)
	# Prepare memory
	return MaxGOprPrep(adjMode, greenCirc, trgVol, srcVol)
end
"""

	function MaxGOprGenSlf(trgDom::MaxGDom, srcDom::MaxGDom)::MaxGOprMem

Prepare memory for Green function operator for a self domain. 
"""
function MaxGOprGenSlf(adjMode::Int64, assemblyInfo::MaxGAssemblyOpts, 
	slfDom::MaxGDom)::MaxGOprMem
	# Create MaxG volume
	slfVol = genMaxGVol(slfDom)
	# Pre-allocate memory for circulant green function vector
	greenCirc = Array{ComplexF64}(undef, 3, 3, 2 * slfVol.cells[1], 2 * 
		slfVol.cells[2], 2 * slfVol.cells[3])
	# Generate circulant Green function
	genGreenSlf!(greenCirc, slfVol, assemblyInfo)
	# Prepare memory
	return MaxGOprPrep(adjMode, greenCirc, slfVol, slfVol)
end
# Lower level memory preparation function.
function MaxGOprPrep(adjMode::Int64, greenCirc::Array{ComplexF64}, 
	trgVol::MaxGVol, srcVol::MaxGVol)::MaxGOprMem
	## Pre-allocate operator memory 
	# Fourier transform of the Green function, making use of real space symmetry 
	# under transposition. Entries are xx, yy, zz, xy, xz, yz
	greenFour = Array{ComplexF64}(undef, srcVol.cells[1] + trgVol.cells[1], 
	srcVol.cells[2] + trgVol.cells[2], srcVol.cells[3] + trgVol.cells[3], 6)
	# Target vector memory
	trgVec = zeros(ComplexF64, trgVol.cells[1], trgVol.cells[2], 
		trgVol.cells[3], 3)
	# Source vector memory
	srcVec = zeros(ComplexF64, srcVol.cells[1], srcVol.cells[2], 
		srcVol.cells[3], 3)
	# Storage for directional components
	vecSumEmbd = Array{ComplexF64}(undef, (srcVol.cells[1] + trgVol.cells[1]) * 
	(srcVol.cells[2] + trgVol.cells[2]) * (srcVol.cells[3] + trgVol.cells[3]), 9)
	# Work area
	vecWrkEmbd = Array{ComplexF64}(undef, (srcVol.cells[1] + trgVol.cells[1]) * 
	(srcVol.cells[2] + trgVol.cells[2]) * (srcVol.cells[3] + trgVol.cells[3]), 3)
	## Plan Fourier transforms
	# External plan
	fftPlanFwdOut = plan_fft(greenCirc[1,1,:,:,:],(1,2,3))
	# In-place plans
	fftPlanInv = plan_ifft!(greenCirc[1,1,:,:,:],(1,2,3))
	fftPlanFwd = plan_fft!(greenCirc[1,1,:,:,:],(1,2,3))
	## Preform Fast-Fourier transforms of circulant Green functions
	grnItr = 0
	blcItr = 0

	for colItr in 1 : 3, rowItr in 1 : colItr

		blcItr = 3 * (colItr - 1) + rowItr
		grnItr = blockGreenItr(blcItr)

		greenFour[:,:,:,grnItr] =  fftPlanFwdOut * 
		greenCirc[rowItr,colItr,:,:,:]
	end
	# What is the point of greenCirc if it is not called in the return 
	# statement below?
	# What variables store the output of the Green function?
	return MaxGOprMem(adjMode, trgVol, srcVol, trgVec, srcVec, greenFour, 
		fftPlanInv, fftPlanFwd, vecSumEmbd, vecWrkEmbd)
end
"""
	
	grnOpr!(oMem::MaxGOprMem)::Nothing

Green function operator linked to oMem. 
"""
function grnOpr!(oMem::MaxGOprMem)::Nothing
	
	crcSize = (oMem.trgVol.cells[1] + oMem.srcVol.cells[1], 
		oMem.trgVol.cells[2] + oMem.srcVol.cells[2],
		oMem.trgVol.cells[3] + oMem.srcVol.cells[3])

	srcSize = (oMem.srcVol.cells[1], oMem.srcVol.cells[2], oMem.srcVol.cells[3])
	trgSize = (oMem.trgVol.cells[1], oMem.trgVol.cells[2], oMem.trgVol.cells[3])

	# Prepare embedded vectors, performing forward FFTs. 
	embdSow!(oMem.fftPlanFwd, crcSize, srcSize, oMem.vecWrkEmbd, oMem.srcVec)
	# Green function multiplications
	blockItr = 0
	greenItr = 0

	for colItr in 1 : 3, rowItr in 1 : 3

		blockItr = rowItr + (colItr - 1) * 3
		greenItr = blockGreenItr(blockItr)
		# Since the vacuum Green function is symmetric under transposition  
		# in real space, the adjoint---adjMode = 1---is simply implemented 
		# by the conjugate of the Fourier coefficients.  
		if oMem.adjMode == 1

			thrdMult!(1, prod(crcSize), view(oMem.vecSumEmbd, :, blockItr), 
				view(oMem.greenFour, :, :, :, greenItr), 
				view(oMem.vecWrkEmbd, :, colItr))
		else

			thrdMult!(0, prod(crcSize), view(oMem.vecSumEmbd, :, blockItr), 
				view(oMem.greenFour, :, :, :, greenItr), 
				view(oMem.vecWrkEmbd, :, colItr))
		end
	end
	# Collect results, performing inverse FFTs.
	embdReap!(oMem.fftPlanInv, trgSize, crcSize, oMem.trgVec, oMem.vecWrkEmbd, 
		oMem.vecSumEmbd)
	
	return nothing
end
# Collect circulant output and calculate the resulting projection. 
function embdReap!(fftPlanInv::AbstractFFTs.Plan{ComplexF64}, 
	trgSize::NTuple{3,Int64}, crcSize::NTuple{3,Int64}, 
	trgMem::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	vecTrgEmbd::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	vecWrkEmbd::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	# Sum components in preparation for inverse Fourier transform
	for dirItr in 1 : 3

		@threads for posItr in 1 : prod(crcSize) 

			vecTrgEmbd[posItr, dirItr] =  vecWrkEmbd[posItr, dirItr] + 
			vecWrkEmbd[posItr, dirItr + 3] + vecWrkEmbd[posItr, dirItr + 6]
		end
	end
	# Preform inverse Fourier transforms
	for dirItr in 1 : 3

		vecTrgEmbd[:, dirItr] = reshape((fftPlanInv * 
			reshape(vecTrgEmbd[:, dirItr], crcSize)), prod(crcSize))
	end
	# Project out of circulant form
	for dirItr in 1 : 3
		
		projVec!(trgSize, crcSize, view(trgMem, :, :, :, dirItr), 
			view(vecTrgEmbd, :, dirItr))
	end
end
# Create circulant embedded input vectors in preparation for Green operation. 
function embdSow!(fftPlanFwd::AbstractFFTs.Plan{ComplexF64},
	crcSize::NTuple{3,Int64}, srcSize::NTuple{3,Int64}, 
	vecSrcEmbd::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	vecSrc::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	# Size of circulant source vector
	for dirItr in 1 : 3

		embdVec!(crcSize, srcSize, view(vecSrcEmbd, :, dirItr), 
			view(vecSrc, :, :, :, dirItr))
	end
	
	for dirItr in 1 : 3

		vecSrcEmbd[:,dirItr] = reshape(fftPlanFwd * 
			reshape(vecSrcEmbd[:,dirItr], crcSize), prod(crcSize))
	end

end
# Circulant projection for a single Cartesian direction. 
function projVec!(trgSize::NTuple{3,Int64}, crcSize::NTuple{3,Int64}, 
	trgMem::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	embMem::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	# Get number of active threads to assign memory. 
	numThreads = nthreads()
	# Local cell counters
	cellX = Array{Int64}(undef, numThreads)	
	cellY = Array{Int64}(undef, numThreads)	
	cellZ = Array{Int64}(undef, numThreads)	
	# Project vector
	@threads for itr in 0 : (prod(crcSize) - 1)
		# Linear index to Cartesian index
		cellX[threadid()] = mod(itr, crcSize[1])
		cellY[threadid()] = div(mod(itr - cellX[threadid()], 
			crcSize[1] * crcSize[2]), crcSize[1])
		cellZ[threadid()] = div(itr - cellX[threadid()] - 
			(cellY[threadid()] * crcSize[1]), crcSize[1] * crcSize[2])

		if ((cellX[threadid()] < trgSize[1]) && 
			(cellY[threadid()] < trgSize[2]) && 
			(cellZ[threadid()] < trgSize[3]))

			trgMem[cellX[threadid()] + 1, cellY[threadid()] + 1, 
			cellZ[threadid()] + 1] = embMem[itr + 1]
		end
	end
end
# Threaded vector multiplication 
@inline function thrdMult!(mode::Int64, size::Int64, 
	trgMem::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	srcAMem::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	srcBMem::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	
	if mode == 0
		
		@threads for itr in 1 : size 

			trgMem[itr] = srcAMem[itr] * srcBMem[itr]
		end
	
	elseif mode == 1

		@threads for itr in 1 : size 

			trgMem[itr] = conj(srcAMem[itr]) * srcBMem[itr]
		end
	
	elseif mode == 2

		@threads for itr in 1 : size 

			trgMem[itr] = conj(srcAMem[itr]) * conj(srcBMem[itr])
		end

	else

		error("Improper use case.")
	end
end
# Circulant embedding for a single Cartesian direction. 
function embdVec!(crcSize::NTuple{3,Int64}, srcSize::NTuple{3,Int64}, 
	embMem::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	srcMem::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	# Get number of active threads to assign memory. 
	numThreads = nthreads()
	# Local cell counters
	cellX = Array{Int64}(undef, numThreads)	
	cellY = Array{Int64}(undef, numThreads)	
	cellZ = Array{Int64}(undef, numThreads)	

	@threads for itr in 0 : (prod(crcSize) - 1)
		# Linear index to Cartesian index
		cellX[threadid()] = mod(itr, crcSize[1])
		cellY[threadid()] = div(mod(itr - cellX[threadid()], 
			crcSize[1] * crcSize[2]), crcSize[1])
		cellZ[threadid()] = div(itr - cellX[threadid()] - 
			(cellY[threadid()] * crcSize[1]), crcSize[1] * crcSize[2])

		if ((cellX[threadid()] < srcSize[1]) && 
			(cellY[threadid()] < srcSize[2]) && 
			(cellZ[threadid()] < srcSize[3]))

			embMem[itr + 1] = srcMem[cellX[threadid()] + 1, 
			cellY[threadid()] + 1, cellZ[threadid()] + 1] 
		
		else

			embMem[itr + 1] = 0.0 + 0.0im
		end
	end
end
# Get Green block index for a given Cartesian index.
@inline function blockGreenItr(cartInd::Int64)::Int64

	if cartInd == 1

		return 1

	elseif cartInd == 2 || cartInd == 4

		return 4

	elseif cartInd == 5 

		return 2
	
	elseif cartInd == 7 || cartInd == 3

		return 5

	elseif cartInd == 8 || cartInd == 6

		return 6

	elseif cartInd == 9 

		return 3

	else

		error("Improper use case.")
		return 0
	end
end
end