const noiseFlr = 1.0e-15

function fourierMod!(greenCF::Array{ComplexF64}, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

	cfSize = size(greenCF) 
	numThreads = nthreads()
	phz = assemblyInfo.freqPhase 
	celNum = prod(cfSize[1:(end - 1)])
	gLinCF = reshape(greenCF, (celNum, cfSize[end]))
	thrdMemA = Array{ComplexF64}(undef, numThreads)	
	thrdMemB = Array{Float64}(undef, numThreads)	
	thrdMemC = Array{Float64}(undef, numThreads)	
	# Account of possibility of complex phase
	for blcItr in 1 : 6
		
		scal!(phz^2, view(gLinCF, :, blcItr))

	end
	# Initial correction of diagonal main diagonal blocks
	for blcItr in 1 : 3

		@threads for cItr in 1 : celNum 

			thrdMemA[threadid()] = gLinCF[cItr,blcItr]

			if abs(real(thrdMemA[threadid()])) < noiseFlr && 
				imag(thrdMemA[threadid()]) < noiseFlr

				gLinCF[cItr,blcItr] = 0.0 + 0.0im 

			elseif abs(real(thrdMemA[threadid()])) < noiseFlr

				gLinCF[cItr,blcItr] = 0.0 + imag(thrdMemA[threadid()]) * 1.0im 

			elseif imag(thrdMemA[threadid()]) < noiseFlr

				gLinCF[cItr,blcItr] = real(thrdMemA[threadid()]) + 0.0im 

			else

			end
		end
	end 
	# Correction of off-diagonal blocks to insure positive semi-definiteness
	# Indices for diagonal blocks ``meeting'' a given off-diagonal block 
	dBlcA = 1
	dBlcB = 1

	for blcItr in 4 : 6

		# Set diagonal blocks for a given off-diagonal block
		if blcItr == 4

			dBlcA = 1
			dBlcB = 2

		elseif blcItr == 5

			dBlcA = 1
			dBlcB = 3

		else

			dBlcA = 2
			dBlcB = 3
		end

		@threads for cItr in 1 : celNum 

			thrdMemA[threadid()] = gLinCF[cItr,blcItr]
			thrdMemB[threadid()] = abs(imag(thrdMemA[threadid()])) 
			thrdMemC[threadid()] = sqrt(abs(gLinCF[cItr,dBlcA] * 
				gLinCF[cItr,dBlcB])) 

			# if  thrdMemB[threadid()] > thrdMemC[threadid()]
				
			# 	gLinCF[cItr,blcItr] = 

			# 	real(thrdMemA[threadid()]) + 
			# 	imag(thrdMemA[threadid()]) * 1.0im * 
			# 	(thrdMemC[threadid()] / thrdMemB[threadid()])

			# else

			# end

			gLinCF[cItr,blcItr] = real(thrdMemA[threadid()])
		end
	end
	# Reintroduce complex phase
	for blcItr in 1 : 6
		
		scal!(1 / (phz^2), view(gLinCF, :, blcItr))	
	end

	return nothing
end

"""

weakE2(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
assemblyInfo::MaxGAssemblyOpts)::Array{ComplexF64,1}	

Head function for integration over edge adjacent square panels. See weakS for 
input parameter descriptions. 
"""
function weakE2(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Array{ComplexF64,1}

	# Memory initialization. 
	fp = 1
	intVal = [0.0, 0.0]
	faces = cubeFaces(scale)
	surfInts = zeros(ComplexF64, 9)
	srfScales = surfScale(scale, scale)
	fPairs = facePairs()
	# Return list is 
	# xxY; xxZ; yyX; yyZ; zzX; zzY; xy; xz; yz;
	# Lower case letters reference the normal directions of the rectangles.
	# Upper case letter reference increasing axis direction when necessary. 
	pairList = [1, 1, 15, 15, 29, 29, 3, 5, 11]
	gridList = [
	[0.0, scale[2], 0.0] ;; [0.0, 0.0, scale[3]] ;; [scale[1], 0.0, 0.0];; 
	[0.0, 0.0, scale[3]] ;; [scale[1], 0.0, 0.0] ;; [0.0, scale[2], 0.0];;
	[0.0, 0.0, 0.0] ;; [0.0, 0.0, 0.0] ;; [0.0, 0.0, 0.0]] 
	# Calculate edge integrals
	@inbounds for ind in 1 : 9

		fp = pairList[ind]
		# Define kernel of integration 
		intKer = (ordVec::Array{Float64,1}, vals::Array{Float64,1}) -> 
		surfKerN(ordVec, vals, gridList[1,ind], gridList[2,ind], 
			gridList[3,ind], fp, faces, faces, fPairs, assemblyInfo)
		# Perform surface integration
		intVal[:] = hcubature(2, intKer, [0.0, 0.0, 0.0, 0.0], 
			[1.0, 1.0, 1.0, 1.0], reltol = cubRelTol, abstol = cubAbsTol, 
			maxevals = 0, error_norm = Cubature.INDIVIDUAL)[1];
		surfInts[ind] = intVal[1] + intVal[2] * im
		# Scaling correction
		surfInts[ind] *= srfScales[fp]
	end

	return surfInts
end