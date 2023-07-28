"""
transformBasisIntegrals evaluates the integrands called by the weakS, weakE, 
and weakV head functions using a series of variable transformations and 
analytic integral evaluations---reducing the four dimensional surface integrals 
performed for ``standard'' cells to one dimensional integrals. No comments are 
included in this low level code, which is simply a julia translation of 
DIRECTFN_E by Athanasios Polimeridis. For a complete description of the steps 
being performed see the article cited above and references included therein. 
"""
# Self panels.
function weakSInt(rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	threads = nthreads()
	glQuadOrder = size(glQuad1)[1]
	intVals = zeros(ComplexF64, 2, threads)
	sumVal = 0.0 + 0.0im
	ψA = 0.0 
	ψB = 0.0
	θ = zeros(Float64, threads)
	ηA = zeros(Float64, threads)
	ηB = zeros(Float64, threads)

	for kk in 1 : 3
		
		for n1 in 1 : 8

			(ψA, ψB) = ψlimS(n1)
			
			for trd in 1 : threads
			
				intVals[1,trd] = 0.0 + 0.0im
			end

			@threads for n2 in 1 : glQuadOrder
				
				id = threadid()
				θ[id] = θf(ψA, ψB, glQuad1[n2,1])
				(ηA[id], ηB[id]) = ηlimS(n1, θ[id])
				intVals[2,id] = 0.0 + 0.0im
				
				for n3 in 1 : glQuadOrder
 
					intVals[2,id] += glQuad1[n3,2] * 
					nS(kk, n1, θ[id], θf(ηA[id], ηB[id], glQuad1[n3,1]), 
						rPoints, glQuad1, assemblyOpts)
				end
				intVals[1,id] += (glQuad1[n2,2] * (ηB[id] - ηA[id]) * 
					sin(θ[id]) * intVals[2,id] / 2.0)
			end
			sumVal += (ψB - ψA) * sum(intVals[1,:]) / 2.0
		end
	end	

	return equiJacobianS(rPoints) * sumVal
end
# Edge panels.
function weakEInt(rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	threads = nthreads()
	glQuadOrder = size(glQuad1)[1]
	intVals = zeros(ComplexF64, 2, threads)
	sumVal = 0.0 + 0.0im
	ψA = 0.0 
	ψB = 0.0
	ηA = zeros(Float64, threads)
	ηB = zeros(Float64, threads)
	θA = zeros(Float64, threads)
	θB = zeros(Float64, threads)
	
	for n1 in 1 : 6

		(ψA, ψB) = ψlimE(n1)
		
		for trd in 1 : threads
		
			intVals[1,trd] = 0.0 + 0.0im
		end

		@threads for n2 in 1 : glQuadOrder

			id = threadid()
			θB[id] = θf(ψA, ψB, glQuad1[n2, 1])
			(ηA[id], ηB[id]) = ηlimE(n1, θB[id])
			
			intVals[2,id] = 0.0 + 0.0im

			for n3 in 1 : glQuadOrder

				θA[id] = θf(ηA[id], ηB[id], glQuad1[n3, 1])
				intVals[2,id] += glQuad1[n3, 2] * cos(θA[id]) *  
				(nE(n1, 1, θA[id], θB[id], rPoints, glQuad1, assemblyOpts) + 
				nE(n1, -1, θA[id], θB[id], rPoints, glQuad1, assemblyOpts)) 
			end
			intVals[1,id] += glQuad1[n2, 2] * (ηB[id] - ηA[id]) * 
			intVals[2,id] / 2.0
		end
		sumVal += (ψB - ψA) * sum(intVals[1,:]) / 2.0
	end
	
	return equiJacobianEV(rPoints) * sumVal
end
# Vertex panels excluding singularity.
function weakVInt(sngMode::Int64, rPoints::Array{Float64,2}, 
	glQuad1::Array{Float64,2}, assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	threads = nthreads()
	sumVal = zeros(ComplexF64, threads) 
	glQuadOrder = size(glQuad1)[1]
	intVals = zeros(ComplexF64, 4, threads)
	xPoints = Array{Float64,3}(undef, 3, 2, threads)
	θ1 = zeros(Float64, threads) 
	θ2 = zeros(Float64, threads) 
	θ3 = zeros(Float64, threads)
	θ4 = zeros(Float64, threads)
	θ5 = zeros(Float64, threads) 
	θ6 = zeros(Float64, threads) 
	L1 = zeros(Float64, threads)
	L2 = zeros(Float64, threads)
	
	if sngMode == 1
 	
 		kerEVL =  (id::Int64, rPoints::Array{Float64,2}, 
 			xPoints::Array{Float64,3}, freqPhase::ComplexF64) -> 
 			kernelEVN(rPoints, view(xPoints, :, :, id), freqPhase)
	else

		kerEVL =  (id::Int64, rPoints::Array{Float64,2}, 
 			xPoints::Array{Float64,3}, freqPhase::ComplexF64) -> 
			kernelEV(rPoints, view(xPoints, :, :, id), freqPhase)
	end

	@threads for n1 in 1 : glQuadOrder

		id = threadid()
		θ1[id] = θf(0.0, π / 3.0, glQuad1[n1,1])
		L1[id] = 2.0 * sqrt(3.0) / (sin(θ1[id]) + sqrt(3.0) * cos(θ1[id])) 
		intVals[1,id] = 0.0 + 0.0im

		for n2 in 1 : glQuadOrder

			θ2[id] = θf(0.0, π / 3.0, glQuad1[n2,1])
			L2[id] = 2.0 * sqrt(3.0) / (sin(θ2[id]) + sqrt(3.0) * cos(θ2[id]))
			intVals[2,id] = 0.0 + 0.0im
			
			for n3 in 1 : glQuadOrder

				θ3[id] = θf(0.0, atan(L2[id] / L1[id]), glQuad1[n3,1])
				intVals[3,id] = 0.0 + 0.0im

				for n4 in 1 : glQuadOrder

					θ4[id] = θf(0.0, L1[id] / cos(θ3[id]), glQuad1[n4,1])
					simplexV!(id, xPoints, θ4[id], θ3[id], θ2[id], θ1[id])
					intVals[3,id] += glQuad1[n4,2] * (θ4[id]^3) * 
					kerEVL(id, rPoints, xPoints, assemblyOpts.freqPhase)
				end

				intVals[3,id] *= L1[id] * sin(θ3[id]) * cos(θ3[id]) / 
				(2.0 * cos(θ3[id]))
				θ5[id] = θf(atan(L2[id] / L1[id]), π / 2.0, glQuad1[n3,1])
				intVals[4,id] = 0.0 + 0.0im

				for n5 in 1 : glQuadOrder

					θ6[id] = θf(0.0, L2[id] / sin(θ5[id]), glQuad1[n5,1])
					simplexV!(id, xPoints, θ6[id], θ5[id], θ2[id], θ1[id])
					intVals[4,id] += glQuad1[n5,2] * (θ6[id]^3) *  
					kerEVL(id, rPoints, xPoints, assemblyOpts.freqPhase)
				end
				intVals[4,id] *= L2[id] * sin(θ5[id]) * cos(θ5[id]) / 
				(2.0 * sin(θ5[id]))
				intVals[2,id] += 0.5 * glQuad1[n3, 2] * (atan(L2[id] / L1[id]) * 
					(intVals[3,id] - intVals[4,id]) + π * intVals[4,id] / 2.0)
			end
			intVals[1,id] += glQuad1[n2,2] * intVals[2,id]
		end
		intVals[1,id] *= π / 6.0
		sumVal[id] += glQuad1[n1,2] * intVals[1,id]
	end

	return equiJacobianEV(rPoints) * π * sum(sumVal) / 6.0
end

function ψlimS(case::Int64)::Tuple{Float64,Float64}
	
	if case == 1 || case == 5 || case == 6
		
		return (0.0, π / 3.0)

	elseif case == 2 || case == 7
		
		return (π / 3.0, 2.0 * π / 3.0)

	elseif case == 3 || case == 4 || case == 8
		
		return (2.0 * π / 3.0, π)

	else

		error("Unrecognized case.")
	end
end

function ψlimE(case::Int64)::Tuple{Float64, Float64}
	
	if case == 1
		
		return (0.0, π / 3.0)

	elseif case == 2 || case == 3
		
		return (π / 3.0, π / 2.0)

	elseif case == 4 || case == 6
		
		return (π / 2.0, π)

	elseif case == 5
		
		return (0.0, π / 2.0)

	else
		
		error("Unrecognized case.")
	end
end

function ηlimS(case::Int64, θ::Float64)::Tuple{Float64,Float64}
	
	if case == 1 || case == 2
		
		return (0.0, 1.0)

	elseif case == 3
		
		return ((1 - tan(π - θ) / sqrt(3.0)) / (1 + tan(π - θ) / sqrt(3.0)), 
			1.0)

	elseif case == 4
		
		return (0.0, 
			(1 - tan(π - θ) / sqrt(3.0)) / (1 + tan(π - θ) / sqrt(3.0)))

	elseif case == 5
		
		return ((tan(θ) / sqrt(3.0) - 1.0) / (1 + tan(θ) / sqrt(3.0)), 0.0)

	elseif case == 6
		
		return (-1.0, (tan(θ) / sqrt(3.0) - 1.0) / (1.0 + tan(θ) / sqrt(3.0)))

	elseif case == 7 || case == 8
		
		return (-1.0, 0.0)

	else

		error("Unrecognized case.")
	end
end

function ηlimE(case::Int64, θ::Float64)::Tuple{Float64, Float64}
	
	if case == 1
		
		return (0.0, atan(sin(θ) + sqrt(3.0) * cos(θ)))

	elseif case == 2
		
		return (atan(sin(θ) - sqrt(3.0) * cos(θ)), 
			atan(sin(θ) + sqrt(3.0) * cos(θ)))

	elseif case == 3 || case == 4
		
		return (0.0, atan(sin(θ) - sqrt(3.0) * cos(θ)))

	elseif case == 5
		
		return (atan(sin(θ) + sqrt(3.0) * cos(θ)), π / 2.0)

	elseif case == 6
		
		return (atan(sin(θ) - sqrt(3.0) * cos(θ)), π / 2.0)

	else

		error("Unrecognized case.")
	end
end	

function nS(dir::Int64, case::Int64, θ1::Float64, θ2::Float64, 
	rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	IS = 0.0 + 0.0im
	glQuadOrder = size(glQuad1)[1]

	if case == 1 || case == 5
		
		for n in 1 : glQuadOrder

			IS += glQuad1[n,2] * aS(rPoints, θ1, θ2, 
				θf(0.0, (1.0 - θ2) / cos(θ1), glQuad1[n,1]), dir, glQuad1, 
				assemblyOpts)
		end
		return (1.0 - θ2) / (2.0 * cos(θ1)) * IS

	elseif case == 2 || case == 3

		for n in 1 : glQuadOrder

			IS += glQuad1[n,2] * aS(rPoints, θ1, θ2, 
				θf(0.0, sqrt(3.0) * (1.0 - θ2) / sin(θ1), glQuad1[n,1]), dir, 
				glQuad1, assemblyOpts)
		end
		return sqrt(3.0) * (1.0 - θ2) / (2.0 * sin(θ1)) * IS

	elseif case == 6 || case == 7
		
		for n in 1 : glQuadOrder

			IS += glQuad1[n,2] * aS(rPoints, θ1, θ2, 
				θf(0.0, sqrt(3.0) * (1.0 + θ2) / sin(θ1), glQuad1[n,1]), dir, 
				glQuad1, assemblyOpts)
		end
		return sqrt(3.0) * (1.0 + θ2) / (2.0 * sin(θ1)) * IS

	elseif case == 4 || case == 8

		for n in 1 : glQuadOrder

			IS += glQuad1[n,2] * aS(rPoints, θ1, θ2, 
				θf(0.0, -(1.0 + θ2) / cos(θ1), glQuad1[n,1]), dir, glQuad1, 
				assemblyOpts)
		end
		return -(1.0 + θ2) / (2.0 * cos(θ1)) * IS

	else
		
		error("Unrecognized case.")
	end
end

function nE(case1::Int64, case2::Int64, θ2::Float64, θ1::Float64, 
	rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	γ = 0.0 
	intVal1 = 0.0 + 0.0im 
	intVal2 = 0.0 + 0.0im
	glQuadOrder = size(glQuad1)[1]

	if case1 == 1 || case1 == 2 
		
		γ = (sin(θ1) + sqrt(3.0) * cos(θ1) - tan(θ2)) / 
		(sin(θ1) + sqrt(3.0) * cos(θ1) + tan(θ2))

		for n in 1 : glQuadOrder
			
			intVal1 += glQuad1[n, 2] * intNE(n, 1, γ, θ2, θ1, rPoints, glQuad1, 
				case2, assemblyOpts)
			intVal2 += glQuad1[n, 2] * intNE(n, 2, γ, θ2, θ1, rPoints, glQuad1, 
				case2, assemblyOpts)
		end
		return intVal2 / 2.0 + γ * (intVal1-intVal2) / 2.0
	
	elseif case1 == 3
		
		γ = sqrt(3.0) / tan(θ1)

		for n in 1 : glQuadOrder

			intVal1 += glQuad1[n, 2] * intNE(n, 1, γ, θ2, θ1, rPoints, glQuad1, 
				case2, assemblyOpts)
			intVal2 += glQuad1[n, 2] * intNE(n, 3, γ, θ2, θ1, rPoints, glQuad1, 
				case2, assemblyOpts)
		end
		return intVal2 / 2.0 + γ * (intVal1 - intVal2) / 2.0

	elseif case1 == 4

		for n in 1 : glQuadOrder
			
			intVal1 += glQuad1[n, 2] * intNE(n, 4, 1.0, θ2, θ1, rPoints, 
				glQuad1, case2, assemblyOpts)
		end
		
		return intVal1 / 2.0
	
	elseif case1 == 5 || case1 == 6
		
		for n in 1 : glQuadOrder
			
			intVal1 += glQuad1[n, 2] * intNE(n, 5, 1.0, θ2, θ1, rPoints, 
				glQuad1, case2, assemblyOpts)
		end
		return intVal1 / 2.0

	else

		error("Unrecognized case.")
	end
end

function intNE(n::Int64, case1::Int64, γ::Float64, θ2::Float64, θ1::Float64, 
	rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, case2::Int64,
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64
	
	if case1 == 1
		
		η = θf(0.0, γ, glQuad1[n,1])
		λ = sqrt(3.0) * (1 + η)  /  (cos(θ2) * (sin(θ1) + sqrt(3.0) * cos(θ1)))
	
	elseif case1 == 2
	
		η = θf(γ, 1.0, glQuad1[n,1])
		λ = sqrt(3.0) * (1.0 - abs(η)) / sin(θ2)
	
	elseif case1 == 3
	
		η = θf(γ, 1.0, glQuad1[n,1])
		λ = sqrt(3.0) * (1.0 - abs(η)) / 
		(cos(θ2) * (sin(θ1) - sqrt(3.0) * cos(θ1)))
	
	elseif case1 == 4
	
		η = θf(0.0, 1.0, glQuad1[n,1])
		λ = sqrt(3.0) * (1.0 - abs(η)) / 
		(cos(θ2) * (sin(θ1) - sqrt(3.0) * cos(θ1)))
	
	elseif case1 == 5
	
		η = θf(0.0, 1.0, glQuad1[n,1])
		λ = sqrt(3.0) * (1.0 - η) / sin(θ2)
	else
		error("Unrecognized case.")
	end

	return aE(rPoints, λ, η, θ2, θ1, glQuad1, case2, assemblyOpts)
end

function aS(rPoints::Array{Float64,2}, θ1::Float64, θ2::Float64, θ::Float64, 
	dir::Int64, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	xPoints = Array{Float64,2}(undef, 3, 2)
	glQuadOrder = size(glQuad1)[1]
	aInt = 0.0 + 0.0im
	η1 = 0.0 
	η2 = 0.0 
	ξ1 = 0.0

	for n in 1 : glQuadOrder
	
		(η1, ξ1) = subTriangles(θ2, θ * sin(θ1), dir)
		(η2, ξ2) = subTriangles(θf(0.0, θ, glQuad1[n,1]) * cos(θ1) + θ2, 
			(θ - θf(0.0, θ, glQuad1[n,1])) * sin(θ1), dir)
		simplex!(xPoints, η1, η2, ξ1, ξ2)
		aInt += glQuad1[n,2] * θf(0.0, θ, glQuad1[n,1]) * 
		kernelSN(rPoints, xPoints, assemblyOpts.freqPhase)
	end
	return θ * aInt / 2.0
end

function aE(rPoints::Array{Float64,2}, λ::Float64, η::Float64, θ2::Float64, 
	θ1::Float64, glQuad1::Array{Float64,2}, case::Int64, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	xPoints = Array{Float64,2}(undef, 3, 2)
	glQuadOrder = size(glQuad1)[1]
	intVal = 0.0 + 0.0im
	ζ = 0.0
	
	for n in 1 : glQuadOrder
	
		ζ = θf(0.0, λ, glQuad1[n,1])
		simplexE!(xPoints, ζ, η, θ2, θ1, case)
		intVal += glQuad1[n,2] * ζ * ζ * kernelEVN(rPoints, xPoints, 
			assemblyOpts.freqPhase)
	end

	return λ * intVal / 2.0
end

function subTriangles(λ1::Float64, λ2::Float64, 
	dir::Int64)::Tuple{Float64,Float64}

	if dir == 1
		
		return (λ1, λ2)
	
	elseif dir == 2
		
		return ((1.0 - λ1 - λ2 * sqrt(3)) / 2.0, 
			(sqrt(3.0) + λ1 * sqrt(3.0) - λ2) / 2.0)
	
	elseif dir == 3
		
		return (( - 1.0 - λ1 + λ2 * sqrt(3)) / 2.0, 
			(sqrt(3.0) - λ1 * sqrt(3.0) - λ2) / 2.0)

	else
		
		error("Unrecognized case.")
	end
end

function equiJacobianEV(rPoints::Array{Float64,2})::Float64

	
	return sqrt(dot(cross(rPoints[:,2] - rPoints[:,1], 
		rPoints[:,3] - rPoints[:,1]), cross(rPoints[:,2] - rPoints[:,1], 
		rPoints[:,3] - rPoints[:,1]))) * sqrt(dot(cross(rPoints[:,5] - 
		rPoints[:,4], rPoints[:,6] - rPoints[:,4]), cross(rPoints[:,5] - 
		rPoints[:,4], rPoints[:,6] - rPoints[:,4]))) / 12.0
end

function equiJacobianS(rPoints::Array{Float64,2})::Float64

	return dot(cross(rPoints[:,1] - rPoints[:,2], rPoints[:,3] - rPoints[:,1]), 
		cross(rPoints[:,1] - rPoints[:,2], rPoints[:,3] - rPoints[:,1])) / 12.0
end

@inline function θf(θa::Float64, θb::Float64, pos::Float64)::Float64	
	
	return ((θb - θa) * pos + θa + θb) / 2.0   
end

function simplexV!(xPoints::Array{Float64,2}, θ4::Float64, θ3::Float64, 
	θ2::Float64, θ1::Float64)

	simplex!(xPoints, θ4 * cos(θ3) * cos(θ1) - 1.0, θ4 * sin(θ3) * cos(θ2) - 
		1.0, θ4 * cos(θ3) * sin(θ1), θ4 * sin(θ3) * sin(θ2))
	
	return nothing
end

function simplexV!(id::Int64, xPoints::Array{Float64,3}, θ4::Float64, 
	θ3::Float64, θ2::Float64, θ1::Float64)

	simplex!(id, xPoints, θ4 * cos(θ3) * cos(θ1) - 1.0, θ4 * sin(θ3) * 
		cos(θ2) - 1.0, θ4 * cos(θ3) * sin(θ1), θ4 * sin(θ3) * sin(θ2))
	
	return nothing
end

function simplexE!(xPoints::Array{Float64,2}, λ::Float64, η::Float64, 
	θ2::Float64, θ1::Float64, case::Int64)

	if case == 1
	
		simplex!(xPoints, η, λ * cos(θ2) * cos(θ1) - η , λ * sin(θ2), 
			λ * cos(θ2) * sin(θ1))
	
	elseif case ==  - 1
	
		simplex!(xPoints,  -η,  -(λ * cos(θ2) * cos(θ1) - η) , λ * sin(θ2), 
			λ * cos(θ2) * sin(θ1))

	else

		error("Unrecognized case.")
	end
	return nothing
end

function simplex!(xPoints::Array{Float64,2}, η1::Float64, η2::Float64, 
	ξ1::Float64, ξ2::Float64)

	xPoints[1,1] = (sqrt(3.0) * (1 - η1) - ξ1) / (2 * sqrt(3))
	xPoints[2,1] = (sqrt(3.0) * (1 + η1) - ξ1) / (2 * sqrt(3))
	xPoints[3,1] = ξ1 / sqrt(3.0)
	xPoints[1,2] = (sqrt(3.0) * (1 - η2) - ξ2) / (2 * sqrt(3))
	xPoints[2,2] = (sqrt(3.0) * (1 + η2) - ξ2) / (2 * sqrt(3))
	xPoints[3,2] = ξ2 / sqrt(3.0)
	
	return nothing
end

function simplex!(id::Int64, xPoints::Array{Float64,3}, η1::Float64, 
	η2::Float64, ξ1::Float64, ξ2::Float64)

	xPoints[1,1,id] = (sqrt(3.0) * (1 - η1) - ξ1) / (2 * sqrt(3))
	xPoints[2,1,id] = (sqrt(3.0) * (1 + η1) - ξ1) / (2 * sqrt(3))
	xPoints[3,1,id] = ξ1 / sqrt(3.0)
	xPoints[1,2,id] = (sqrt(3.0) * (1 - η2) - ξ2) / (2 * sqrt(3))
	xPoints[2,2,id] = (sqrt(3.0) * (1 + η2) - ξ2) / (2 * sqrt(3))
	xPoints[3,2,id] = ξ2 / sqrt(3.0)
	
	return nothing
end

function kernelEV(rPoints::Array{Float64,2}, 
	xPoints::Union{Array{Float64,2},SubArray{Float64,2}}, 
	freqPhase::ComplexF64)::ComplexF64

	return	sclGreen(distMag(xPoints[1,1] * rPoints[1,1] + xPoints[2,1] * 
		rPoints[1,2] + xPoints[3,1] * rPoints[1,3] - (xPoints[1,2] * 
			rPoints[1,4] + xPoints[2,2] * rPoints[1,5] + xPoints[3,2] * 
			rPoints[1,6]), xPoints[1,1] * rPoints[2,1] + xPoints[2,1] * 
		rPoints[2,2] + xPoints[3,1] * rPoints[2,3] - (xPoints[1,2] * 
			rPoints[2,4] + xPoints[2,2] * rPoints[2,5] + xPoints[3,2] * 
			rPoints[2,6]), xPoints[1,1] * rPoints[3,1] + xPoints[2,1] * 
		rPoints[3,2] + xPoints[3,1] * rPoints[3,3] - (xPoints[1,2] * 
			rPoints[3,4] + xPoints[2,2] * rPoints[3,5] + xPoints[3,2] * 
			rPoints[3,6])), freqPhase)
end

function kernelEVN(rPoints::Array{Float64,2}, 
	xPoints::Union{Array{Float64,2},SubArray{Float64,2}}, 
	freqPhase::ComplexF64)::ComplexF64

	return	sclGreenN(distMag(xPoints[1,1] * rPoints[1,1] + xPoints[2,1] * 
		rPoints[1,2] + xPoints[3,1] * rPoints[1,3] - (xPoints[1,2] * 
			rPoints[1,4] + xPoints[2,2] * rPoints[1,5] + xPoints[3,2] * 
			rPoints[1,6]), xPoints[1,1] * rPoints[2,1] + xPoints[2,1] * 
		rPoints[2,2] + xPoints[3,1] * rPoints[2,3] - (xPoints[1,2] * 
			rPoints[2,4] + xPoints[2,2] * rPoints[2,5] + xPoints[3,2] * 
			rPoints[2,6]), xPoints[1,1] * rPoints[3,1] + xPoints[2,1] * 
		rPoints[3,2] + xPoints[3,1] * rPoints[3,3] - (xPoints[1,2] * 
			rPoints[3,4] + xPoints[2,2] * rPoints[3,5] + xPoints[3,2] * 
			rPoints[3,6])), freqPhase)
end

function kernelSN(rPoints::Array{Float64,2}, 
	xPoints::Array{Float64,2}, freqPhase::ComplexF64)::ComplexF64
	
	return	sclGreenN(distMag(xPoints[1,1] * rPoints[1,1] + xPoints[2,1] * 
		rPoints[1,2] + xPoints[3,1] * rPoints[1,3] - (xPoints[1,2] * 
			rPoints[1,1] + xPoints[2,2] * rPoints[1,2] + xPoints[3,2] * 
			rPoints[1,3]), xPoints[1,1] * rPoints[2,1] + xPoints[2,1] * 
		rPoints[2,2] + xPoints[3,1] * rPoints[2,3] - (xPoints[1,2] * 
			rPoints[2,1] + xPoints[2,2] * rPoints[2,2] + xPoints[3,2] * 
			rPoints[2,3]), xPoints[1,1] * rPoints[3,1] + xPoints[2,1] * 
		rPoints[3,2] + xPoints[3,1] * rPoints[3,3] - (xPoints[1,2] * 
			rPoints[3,1] + xPoints[2,2] * rPoints[3,2] + xPoints[3,2] * 
			rPoints[3,3])), freqPhase)
end