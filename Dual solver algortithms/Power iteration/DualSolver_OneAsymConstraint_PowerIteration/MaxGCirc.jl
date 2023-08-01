"""
The MaxGCirc module furnishes the unique elements of the electromagnetic 
Green functions, embedded in a circulant form. The code is distributed under 
GNU LGPL.

Author: Sean Molesky 

Reference: Sean Molesky, MaxG documentation sections II and IV.
"""
module MaxGCirc
using Cubature, MaxGParallelUtilities, MaxGStructs, MaxGBasisIntegrals,
LinearAlgebra
export genGreenExt!, genGreenSlf!, separationGrid
# Settings for cubature integral evaluation, relative tolerance and absolute 
# tolerance
const cubRelTol = 1e-7; # changed the value from 1e-8
const cubAbsTol = 1e-12; 
"""

	genGreenExt!(greenCirc::Array{ComplexF64}, srcVol::MaxGVol, 
	trgVol::MaxGVol, assemblyInfo::MaxGAssemblyOpts)::Nothing

Calculate circulant form for the discrete Green function between a target 
volume, trgVol, and a source domain, srcVol.
"""
function genGreenExt!(greenCirc::Array{ComplexF64}, trgVol::MaxGVol, 
	srcVol::MaxGVol, assemblyInfo::MaxGAssemblyOpts)::Nothing

	fPairs = facePairs()
	trgFaces = cubeFaces(trgVol.scale)
	srcFaces = cubeFaces(srcVol.scale)
	sGridT = separationGrid(trgVol, srcVol, 0)
	sGridS = separationGrid(trgVol, srcVol, 1)
	# Calculate Green function
	assembleGreenCircExt!(greenCirc, trgVol, srcVol, sGridT, sGridS, trgFaces, 
		srcFaces, fPairs, assemblyInfo)
	
	return nothing
end
"""
	
	genGreenSlf!(greenCirc::Array{ComplexF64}, slfVol::MaxGVol, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

Calculate circulant form for the discrete Green function on a single domain.
"""
function genGreenSlf!(greenCirc::Array{ComplexF64}, slfVol::MaxGVol, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

	fPairs = facePairs()
	trgFaces = cubeFaces(slfVol.scale)
	srcFaces = cubeFaces(slfVol.scale)
	sGrid = separationGrid(slfVol, slfVol, 0)
	# Calculate Green function
	assembleGreenCircSlf!(greenCirc, slfVol, sGrid, trgFaces, srcFaces, 
		fPairs, assemblyInfo)
	
	return nothing
end
"""
Generate the circulant form of the external Green function between a pair of 
distinct domains. 
"""
function assembleGreenCircExt!(greenCirc::Array{ComplexF64}, trgVol::MaxGVol, 
	srcVol::MaxGVol, sGridT::Array{<:StepRangeLen,1}, 
	sGridS::Array{<:StepRangeLen,1}, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, fPairs::Array{Int64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

	indSplit1 = trgVol.cells[1]
	indSplit2 = trgVol.cells[2]
	indSplit3 = trgVol.cells[3]

	greenExt! = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> 
	greenExtFunc!(greenMat, ind1, ind2, ind3, indSplit1, indSplit2, indSplit3, 
		sGridT, sGridS, trgVol.scale, srcVol.scale, trgFaces, srcFaces, fPairs, 
		assemblyInfo)
	threadArrayW!(greenCirc, 3, size(greenCirc), greenExt!)
	
	return nothing
end
"""
Generate Green function self interaction circulant vector.
"""
function assembleGreenCircSlf!(greenCirc::Array{ComplexF64}, slfVol::MaxGVol, 
	sGrid::Array{<:StepRangeLen,1}, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, fPairs::Array{Int64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

	# Allocate array to store intermediate Toeplitz interaction form.
	greenToe = Array{ComplexF64}(undef, 3, 3, slfVol.cells[1], slfVol.cells[2], 
		slfVol.cells[3])
	# Write Green function, ignoring singular integrals. 
	greenFunc! = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> greenFuncInn!(greenMat, ind1, ind2, ind3, sGrid, 
		slfVol.scale, trgFaces, srcFaces, fPairs, assemblyInfo)
	threadArrayW!(greenToe, 3, size(greenToe), greenFunc!)
	# 1D quadrature points for singular integrals.
	quadOrder = assemblyInfo.ordGLIntNear
	# Gauss-Legendre quadrature.
	glSngQuad = gaussQuad1(quadOrder) 
	# Correction values for singular integrals.
	# Return order of normal faces xx yy zz
	wS = (^(prod(slfVol.scale), -1) .* 
		weakS(slfVol.scale, glSngQuad, assemblyInfo))
	# Return order of normal faces xxY xxZ yyX yyZ zzX zzY xy xz yz
	wE = (^(prod(slfVol.scale), -1) .* 
		weakE(slfVol.scale, glSngQuad, assemblyInfo))
	# Return order of normal faces xx yy zz xy xz yz 
	wV = (^(prod(slfVol.scale), -1) .* 
		weakV(slfVol.scale, glSngQuad, assemblyInfo))
	# Correct singular integrals for coincident and adjacent cells.
	greenFunc! = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> greenFuncSng!(greenMat, ind1, ind2, ind3, wS, wE, wV, 
		sGrid, slfVol.scale, trgFaces, srcFaces, fPairs, assemblyInfo)
	threadArrayW!(greenToe, 3, (3, 3, min(slfVol.cells[1], 2), 
		min(slfVol.cells[2], 2), min(slfVol.cells[3], 2)), greenFunc!)
	# Add in identity operator
	greenToe[1,1,1,1,1] -= 1 / (assemblyInfo.freqPhase^2)
	greenToe[2,2,1,1,1] -= 1 / (assemblyInfo.freqPhase^2)
	greenToe[3,3,1,1,1] -= 1 / (assemblyInfo.freqPhase^2)
	# Embed self Green function into a circulant form
	indSplit1 = div(size(greenCirc)[3], 2)
	indSplit2 = div(size(greenCirc)[4], 2)
	indSplit3 = div(size(greenCirc)[5], 2)
	embedFunc = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> greenToeToCirc!(greenMat, greenToe, ind1, ind2, ind3, 
		indSplit1, indSplit2, indSplit3)
	threadArrayW!(greenCirc, 3, size(greenCirc), embedFunc)
	
	return nothing
end
"""
Generate circulant self Green function from Toeplitz self Green function. The 
implemented mask takes into account the relative flip in the assumed dipole 
direction under a coordinate reflection. 
"""
function greenToeToCirc!(greenCirc::SubArray{ComplexF64,2}, 
	greenToe::Array{ComplexF64}, ind1::Int64, ind2::Int64, ind3::Int64, 
	indSplit1::Int64, indSplit2::Int64, indSplit3::Int64)::Nothing
	
	if ind1 == (indSplit1 + 1) || ind2 == (indSplit2 + 1) || 
		ind3 == (indSplit3 + 1)

		greenCirc[:,:] = zeros(ComplexF64, 3, 3)
	
	else		

		fi = indFlip(ind1, indSplit1)
		fj = indFlip(ind2, indSplit2)
		fk = indFlip(ind3, indSplit3)

		greenCirc[:,:] = view(greenToe, :, :, indSelect(ind1, indSplit1), 
		indSelect(ind2, indSplit2), indSelect(ind3, indSplit3)) .* 
		[1.0 (fi * fj)  (fi * fk); (fj * fi) 1.0 (fj * fk); 
		(fk * fi) (fk * fj) 1.0]
	end

	return nothing
end
"""
Write Green function element for a pair of cubes in distinct domains. Recall 
that grids span the separations between a pair of volumes. 
"""
@inline function greenExtFunc!(greenMat::SubArray{ComplexF64}, ind1::Int64, 
	ind2::Int64, ind3::Int64, indSplit1::Int64, indSplit2::Int64, 
	indSplit3::Int64, sGridT::Array{<:StepRangeLen,1}, 
	sGridS::Array{<:StepRangeLen,1}, scaleT::NTuple{3,Float64}, 
	scaleS::NTuple{3,Float64}, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, fPairs::Array{Int64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing
	
	greenFuncOut!(greenMat, gridSelect(ind1, indSplit1, 1, sGridT, sGridS), 
		gridSelect(ind2, indSplit2, 2, sGridT, sGridS), 
		gridSelect(ind3, indSplit3, 3, sGridT, sGridS), 
		scaleT, scaleS, trgFaces, srcFaces, fPairs, assemblyInfo)

	if ind1 == (indSplit1 + 1) || ind2 == (indSplit2 + 1) || 
		ind3 == (indSplit3 + 1)

		greenMat[:,:] = zeros(ComplexF64, 3, 3)
	
	else		

		fi = sign(gridSelect(ind1, indSplit1, 1, sGridT, sGridS))
		fj = sign(gridSelect(ind1, indSplit1, 1, sGridT, sGridS))
		fk = sign(gridSelect(ind1, indSplit1, 1, sGridT, sGridS))

		greenMat[:,:] = greenMat[:,:] .* 
		[1.0 (fi * fj)  (fi * fk); (fj * fi) 1.0 (fj * fk); 
		(fk * fi) (fk * fj) 1.0]
	end

	return nothing
end
"""
Write a general external Green function element to a shared memory array. 
"""
function greenFuncOut!(greenMat::SubArray{ComplexF64,2}, gridX::Float64, 
	gridY::Float64, gridZ::Float64, scaleT::NTuple{3,Float64}, 
	scaleS::NTuple{3,Float64}, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, fPairs::Array{Int64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

	surfMat = zeros(ComplexF64, 36)
	# Green function between all cube faces.
	greenSurfsAdp!(gridX, gridY, gridZ, surfMat, trgFaces, srcFaces, 1:36, 
		fPairs, surfScale(scaleS, scaleT), assemblyInfo)
	# Add cube face contributions depending on source and target current 
	# orientation. 
	surfSums!(greenMat::SubArray{ComplexF64,2}, surfMat::Array{ComplexF64,1})
	
	return nothing
end
"""
Write a general self Green function element to a shared memory array. 
"""
function greenFuncInn!(greenMat::SubArray{ComplexF64}, ind1::Int64, 
	ind2::Int64, ind3::Int64, sGrid::Array{<:StepRangeLen,1}, 
	scale::NTuple{3,Float64}, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, fPairs::Array{Int64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

	surfMat = zeros(ComplexF64, 36)
	# Green function between all cube faces.
	if (ind1 > 2) || (ind2 > 2) || (ind3 > 2)
		
		greenSurfsAdp!(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], surfMat, 
		trgFaces, srcFaces, 1:36, fPairs, surfScale(scale, scale), assemblyInfo)
		# Add cube face contributions depending on source and target current 
		# orientation. 
		surfSums!(greenMat::SubArray{ComplexF64,2}, surfMat::Array{ComplexF64,1})
	end
	return nothing
end
"""
Generate Green elements for adjacent cubes, assumed to be in the same domain. 
wS, wE, and wV refer to self-intersecting, edge intersecting, and vertex 
intersecting cube face integrals respectively. In the wE and wV cases, the 
first value returned is for in-plane faces, and the second value is for 
``corned'' faces. 

Note that the convention by which facePairs are generated begins by looping 
over the source faces. Due to this choice, the transpose of the mask  
follows the standard source to target matrix convention used elsewhere. 
"""
function greenFuncSng!(greenMat::SubArray{ComplexF64,2}, ind1::Int64, 
	ind2::Int64, ind3::Int64, wS::Array{ComplexF64,1}, wE::Array{ComplexF64,1},
	wV::Array{ComplexF64,1}, sGrid::Array{<:StepRangeLen,1}, 
	scale::NTuple{3,Float64}, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, fPairs::Array{Int64,2}, 
	assemblyInfo::MaxGAssemblyOpts)

	surfMat = zeros(ComplexF64, 36)
	# linear index conversion
	linearCon = LinearIndices((1:6, 1:6))
	
	# Face convention yzL yzU (x-faces) xzL xzU (y-faces) xyL xyU (z-faces)
	# Index based corrections.
	if (ind1, ind2, ind3) == (1, 1, 1) 
		
		correctionVal = [
		wS[1]  0.0    wE[7]  wE[7]  wE[8]  wE[8]
		0.0    wS[1]  wE[7]  wE[7]  wE[8]  wE[8]
		wE[7]  wE[7]  wS[2]   0.0   wE[9]  wE[9]
		wE[7]  wE[7]  0.0    wS[2]  wE[9]  wE[9]
		wE[8]  wE[8]  wE[9]  wE[9]  wS[3]  0.0
		wE[8]  wE[8]  wE[9]  wE[9]  0.0    wS[3]]
		
		mask =[
		1 0 1 1 1 1
		0 1 1 1 1 1
		1 1 1 0 1 1
		1 1 0 1 1 1
		1 1 1 1 1 0
		1 1 1 1 0 1]

		pairListUn = linearCon[findall(iszero, transpose(mask))]
		# Uncorrected surface integrals.
		greenSurfsAdp!(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], surfMat,
			trgFaces, srcFaces, pairListUn, fPairs, surfScale(scale, scale), 
			assemblyInfo)
	
	elseif (ind1, ind2, ind3) == (2, 1, 1) 
		
		correctionVal = [
		0.0  wS[1]  wE[7]  wE[7]  wE[8]  wE[8]
		0.0  0.0    0.0    0.0    0.0    0.0
		0.0  wE[7]  wE[3]  0.0    wV[6]  wV[6]
		0.0  wE[7]  0.0    wE[3]  wV[6]  wV[6]
		0.0  wE[8]  wV[6]  wV[6]  wE[5]  0.0
		0.0  wE[8]  wV[6]  wV[6]  0.0    wE[5]]
		
		mask = [
		0 1 1 1 1 1
		0 0 0 0 0 0
		0 1 1 0 1 1
		0 1 0 1 1 1
		0 1 1 1 1 0
		0 1 1 1 0 1]

		pairListUn = linearCon[findall(iszero, transpose(mask))]
		# Uncorrected surface integrals.
		greenSurfsAdp!(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], surfMat,
			trgFaces, srcFaces, pairListUn, fPairs, surfScale(scale, scale), 
			assemblyInfo)

	elseif (ind1, ind2, ind3) == (2, 1, 2)
		
		correctionVal = [
		0.0  wE[2]  wV[4]  wV[4]  0.0  wE[8]
		0.0  0.0    0.0    0.0    0.0  0.0
		0.0  wV[4]  wV[2]  0.0    0.0  wV[6]
		0.0  wV[4]  0.0    wV[2]  0.0  wV[6]
		0.0  wE[8]  wV[6]  wV[6]  0.0  wE[5]
		0.0  0.0    0.0    0.0    0.0  0.0]
		
		mask = [
		0 1 1 1 0 1
		0 0 0 0 0 0
		0 1 1 0 0 1
		0 1 0 1 0 1
		0 1 1 1 0 1
		0 0 0 0 0 0] 

		pairListUn = linearCon[findall(iszero, transpose(mask))]
		# Uncorrected surface integrals.
		greenSurfsAdp!(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], surfMat,
			trgFaces, srcFaces, pairListUn, fPairs, surfScale(scale, scale), 
			assemblyInfo)

	elseif (ind1, ind2, ind3) == (1, 1, 2) 
		
		correctionVal = [
		wE[2]  0.0    wV[4]  wV[4]  0.0  wE[8]
		0.0    wE[2]  wV[4]  wV[4]  0.0  wE[8]
		wV[4]  wV[4]  wE[4]  0.0    0.0  wE[9]
		wV[4]  wV[4]  0.0    wE[4]  0.0  wE[9]
		wE[8]  wE[8]  wE[9]  wE[9]  0.0  wS[3]
		0.0    0.0    0.0    0.0    0.0  0.0]
		
		mask = [
		1 0 1 1 0 1
		0 1 1 1 0 1
		1 1 1 0 0 1
		1 1 0 1 0 1
		1 1 1 1 0 1
		0 0 0 0 0 0]

		pairListUn = linearCon[findall(iszero, transpose(mask))]
		# Uncorrected surface integrals.
		greenSurfsAdp!(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], surfMat,
			trgFaces, srcFaces, pairListUn, fPairs, surfScale(scale, scale), 
			assemblyInfo)

	elseif (ind1, ind2, ind3) == (1, 2, 1) 
		
		correctionVal = [
		wE[1]  0.0    0.0  wE[7]  wV[5]  wV[5]
		0.0    wE[1]  0.0  wE[7]  wV[5]  wV[5]
		wE[7]  wE[7]  0.0  wS[2]  wE[9]  wE[9]
		0.0    0.0    0.0  0.0    0.0    0.0
		wV[5]  wV[5]  0.0  wE[9]  wE[6]  0.0
		wV[5]  wV[5]  0.0  wE[9]  0.0    wE[6]]
		
		mask = [
		1 0 0 1 1 1
		0 1 0 1 1 1
		1 1 0 1 1 1
		0 0 0 0 0 0
		1 1 0 1 1 0
		1 1 0 1 0 1]

		pairListUn = linearCon[findall(iszero, transpose(mask))]
		# Uncorrected surface integrals.
		greenSurfsAdp!(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], surfMat,
			trgFaces, srcFaces, pairListUn, fPairs, surfScale(scale, scale), 
			assemblyInfo)

	elseif (ind1, ind2, ind3) == (2, 2, 1) 

		correctionVal = [
		0.0  wE[1]  0.0  wE[7]  wV[5]  wV[5]
		0.0  0.0    0.0  0.0    0.0    0.0
		0.0  wE[7]  0.0  wE[3]  wV[6]  wV[6]
		0.0  0.0    0.0  0.0    0.0    0.0
		0.0  wV[5]  0.0  wV[6]  wV[3]  0.0
		0.0  wV[5]  0.0  wV[6]  0.0    wV[3]]

		mask = [
		0 1 0 1 1 1
		0 0 0 0 0 0
		0 1 0 1 1 1
		0 0 0 0 0 0
		0 1 0 1 1 0
		0 1 0 1 0 1]  

		pairListUn = linearCon[findall(iszero, transpose(mask))]
		# Uncorrected surface integrals.
		greenSurfsAdp!(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], surfMat,
			trgFaces, srcFaces, pairListUn, fPairs, surfScale(scale, scale), 
			assemblyInfo)

	elseif (ind1, ind2, ind3) == (1, 2, 2) 
		
		correctionVal = [
		wV[1]  0.0    0.0  wV[4]  0.0  wV[5]
		0.0    wV[1]  0.0  wV[4]  0.0  wV[5]
		wV[4]  wV[4]  0.0  wE[4]  0.0  wE[9]
		0.0    0.0    0.0  0.0    0.0  0.0
		wV[5]  wV[5]  0.0  wE[9]  0.0  wE[6]
		0.0    0.0    0.0  0.0    0.0  0.0]
		
		mask = [
		1 0 0 1 0 1
		0 1 0 1 0 1
		1 1 0 1 0 1
		0 0 0 0 0 0
		1 1 0 1 0 1
		0 0 0 0 0 0]  

		pairListUn = linearCon[findall(iszero, transpose(mask))]
		# Uncorrected surface integrals.
		greenSurfsAdp!(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], surfMat,
			trgFaces, srcFaces, pairListUn, fPairs, surfScale(scale, scale), 
			assemblyInfo)

	elseif (ind1, ind2, ind3) == (2, 2, 2) 
		
		correctionVal = [
		0.0  wV[1]  0.0  wV[4]  0.0  wV[5]
		0.0  0.0    0.0  0.0    0.0  0.0
		0.0  wV[4]  0.0  wV[2]  0.0  wV[6]
		0.0  0.0    0.0  0.0    0.0  0.0
		0.0  wV[5]  0.0  wV[6]  0.0  wV[3]
		0.0  0.0    0.0  0.0    0.0  0.0]
		
		mask = [
		0 1 0 1 0 1
		0 0 0 0 0 0
		0 1 0 1 0 1
		0 0 0 0 0 0
		0 1 0 1 0 1
		0 0 0 0 0 0]

		pairListUn = linearCon[findall(iszero, transpose(mask))]
		# Uncorrected surface integrals.
		greenSurfsAdp!(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], surfMat,
			trgFaces, srcFaces, pairListUn, fPairs, surfScale(scale, scale), 
			assemblyInfo)

	else
		
		println(ind1, ind2, ind3)
		error("Attempted to access improper case.")
	end
	# Correct values of surfMat where needed
	for fp in 1 : 36
		
		if mask[fPairs[fp,1], fPairs[fp,2]] == 1
		
			surfMat[fp] = correctionVal[fPairs[fp,1], fPairs[fp,2]]
		end
	end
	# Overwrite problematic elements of Green function matrix.
	surfSums!(greenMat::SubArray{ComplexF64,2}, surfMat::Array{ComplexF64,1})
	
	return nothing
end
"""
Mutate greenMat to hold Green function interactions.

The storage format of greenMat, see documentation for explanation, is 
[[ii, ji, ki]^{T}; [ij, jj, kj]^{T}; [ik, jk, kk]^{T}].
"""
function surfSums!(greenMat::SubArray{ComplexF64,2}, 
	surfMat::Array{ComplexF64,1})::Nothing

	# ii
	greenMat[1,1] = surfMat[15] - surfMat[16] - surfMat[21] + surfMat[22] + 
	surfMat[29] - surfMat[30] - surfMat[35] + surfMat[36]
	# ji
	greenMat[2,1] = - surfMat[13] + surfMat[14] + surfMat[19] - surfMat[20] 
	# ki
	greenMat[3,1] = - surfMat[25] + surfMat[26] + surfMat[31] - surfMat[32] 
	# ij
	greenMat[1,2] = - surfMat[3] + surfMat[4] + surfMat[9] - surfMat[10]
	# jj
	greenMat[2,2] = surfMat[1] - surfMat[2] - surfMat[7] + surfMat[8] + 
	surfMat[29] - surfMat[30] - surfMat[35] + surfMat[36]
	# kj
	greenMat[3,2] = - surfMat[27] + surfMat[28] + surfMat[33] - surfMat[34]
	# ik
	greenMat[1,3] = - surfMat[5] + surfMat[6] + surfMat[11] - surfMat[12]
	# jk
	greenMat[2,3] = - surfMat[17] + surfMat[18] + surfMat[23] - surfMat[24]
	# kk
	greenMat[3,3] = surfMat[1] - surfMat[2] - surfMat[7] + surfMat[8] + 
	surfMat[15] - surfMat[16] - surfMat[21] + surfMat[22]

	return nothing
end
"""
Kernel function for Green function surface integrals
"""
function surfKer(ordVec::Array{Float64,1}, vals::Array{Float64,1}, 
	gridX::Float64, gridY::Float64, gridZ::Float64, fp::Int64, 
	trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)::Nothing

	z = sclGreen(distMag(
		cubeVecAltAdp(1, ordVec, fp, trgFaces, srcFaces, fPairs) + gridX, 
		cubeVecAltAdp(2, ordVec, fp, trgFaces, srcFaces, fPairs) + gridY, 
		cubeVecAltAdp(3, ordVec, fp, trgFaces, srcFaces, fPairs) + gridZ), 
		assemblyInfo.freqPhase)

	vals[:] = [real(z), imag(z)]

	return nothing
end
"""
Adaptive calculation of the Green function over face pair interactions. 
"""
function greenSurfsAdp!(gridX::Float64, gridY::Float64, gridZ::Float64, 
	surfMat::Array{ComplexF64,1}, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, pairList::Union{UnitRange{Int64},Array{Int64,1}}, 
	fPairs::Array{Int64,2}, srfScales::Array{Float64,1}, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

	# Container for intermediate integral evaluation. 
	intVal = [0.0, 0.0]

	@inbounds for fp in pairList

		surfMat[fp] = 0.0 + 0.0im
		# Define kernel of integration 
		intKer = (ordVec::Array{Float64,1}, vals::Array{Float64,1}) -> 
		surfKer(ordVec, vals, gridX, gridY, gridZ, fp, trgFaces, srcFaces, 
			fPairs, assemblyInfo)
		# Perform surface integration
		intVal[:] = hcubature(2, intKer, [0.0, 0.0, 0.0, 0.0], 
			[1.0, 1.0, 1.0, 1.0], reltol = cubRelTol, abstol = cubAbsTol, 
			maxevals = 0, error_norm = Cubature.INDIVIDUAL)[1];
		surfMat[fp] = intVal[1] + intVal[2] * im
		# Scaling correction
		surfMat[fp] *= srfScales[fp]
	end

return nothing
end
"""
Generates grid of spanning separations for a pair of volumes. The flipped 
separation grid is used in the generation of the circulant form.  
"""
function separationGrid(trgVol::MaxGVol, srcVol::MaxGVol, 
	flip::Int64)::Array{<:StepRangeLen,1}
	
	start = zeros(3)
	stop = zeros(3)
	gridS = srcVol.grid
	gridT = trgVol.grid

	if flip == 1

		sep = round.(srcVol.scale, digits = 8)

		for i in 1 : 3
		
			start[i] = round(gridT[i][1] - gridS[i][end], digits = 8)
			stop[i] = round(gridT[i][1] - gridS[i][1], digits = 8)

			if stop[i] < start[i]
		
				sep[i] *= -1.0; 
			end
		end
		
		return [start[1] : sep[1] : stop[1], start[2] : sep[2] : stop[2], 
		start[3] : sep[3] : stop[3]]
	else
		
		sep = round.(trgVol.scale, digits = 8)
		
		for i in 1 : 3
		
			start[i] = round(gridT[i][1] - gridS[i][1], digits = 8)
			stop[i] =  round(gridT[i][end] - gridS[i][1], digits = 8)
			
			if stop[i] < start[i]
		
				sep[i] *= -1.0; 
			end
		end
		
		return [start[1] : sep[1] : stop[1], start[2] : sep[2] : stop[2], 
		start[3] : sep[3] : stop[3]]
	end 
end
"""
Return the separation between two elements from circulant embedding indices and 
domain grids. 
"""
@inline function gridSelect(ind::Int64, indSplit::Int64, dir::Int64, 
	gridT::Array{<:StepRangeLen,1}, gridS::Array{<:StepRangeLen,1})::Float64
	
	if ind <= indSplit
	
		return (gridT[dir][ind])

	else

		if ind > (1 + indSplit)
		
			ind -= 1
		end		

		return (gridS[dir][ind - indSplit])
	end
end
"""
Return a reference index relative to the embedding index of the Green function. 
"""
@inline function indSelect(ind::Int64, indSplit::Int64)::Int64

	if ind <= indSplit
		
		return ind

	else
		
		if ind == (1 + indSplit)
			
			ind -= 1

		else

			ind -= 2
		end
		
		return 2 * indSplit - ind
	end
end
"""
Flip effective dipole direction based on index values. 
"""
@inline function indFlip(ind::Int64, indSplit::Int64)::Float64

	if ind <= indSplit
		
		return 1.0

	else
		
		return -1.0
	end
end
end