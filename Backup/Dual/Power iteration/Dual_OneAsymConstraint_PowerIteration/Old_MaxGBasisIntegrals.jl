#=
Conventions for the values returned by the weak functions. Small letters 
correspond to normal face directions; capital letters correspond to grid 
increment directions. 

Self 	xx  yy  zz
     	1   2   3


Edge 	xxY xxZ yyX yyZ zzX zzY xy xz yz
     	1   2   3   4   5   6   7  8  9     


Vertex 	xx  yy  zz  xy  xz  yz 
       	1   2   3   4   5   6

MaxGBasisIntegrals contains all necessary support functions for  numerical 
integration of the electromagnetic Green function. This code is translated from 
DIRECTFN_E by Athanasios Polimeridis, and is distributed under the GNU LGPL.

Author: Sean Molesky

Reference: Polimeridis AG, Vipiana F, Mosig JR, Wilton DR. 
DIRECTFN: Fully numerical algorithms for high precision computation of singular 
integrals in Galerkin SIE methods. 
IEEE Transactions on Antennas and Propagation. 2013; 61(6):3112-22.

In what follows the word weak is used in reference to the fact that the form of 
the scalar Green function is weakly singular: the integrand exhibits a 
singularity proportional to the inverse of the separation distance. The letters 
S, E and V refer, respectively, to integration over self-adjacent triangles, 
edge-adjacent triangles, and vertex-adjacent triangles. 

The article cited above contains useful error comparison plots for the number 
evaluation points considered. 
=#
module MaxGBasisIntegrals
using Base.Threads, LinearAlgebra, FastGaussQuadrature, Cubature, MaxGStructs
export  sclGreen, sclGreenN, distMag, weakS, weakE, weakE2, weakV, facePairs, 
surfScale, cubeFaces, cubeVecAlt, cubeVecAltAdp, gaussQuad2, gaussQuad1,
rSurfEdgCrn, rSurfEdgFlt

const π = 3.1415926535897932384626433832795028841971693993751058209749445923
"""

sclGreen(distMag::Float64, freqPhase::ComplexF64)::ComplexF64

Returns the scalar (Helmholtz) Green function. The separation distance mag is 
assumed to be scaled (divided) by the wavelength. 
"""
@inline function sclGreen(distMag::Float64, freqPhase::ComplexF64)::ComplexF64

	return exp(2im * π * distMag * freqPhase) / (4 * π * distMag * freqPhase^2)
end
"""

sclGreenN(distMag::Float64, freqPhase::ComplexF64)::ComplexF64

Returns the scalar (Helmholtz) Green function with the singularity removed. 
The separation distance mag is assumed to be scaled (divided) by the wavelength. 
"""
@inline function sclGreenN(distMag::Float64, freqPhase::ComplexF64)::ComplexF64
    
    if distMag > 1e-7

        return (exp(2im * π * distMag * freqPhase) - 1) / 
        (4 * π * distMag * freqPhase^2)

    else

        return ((im / freqPhase) - π * distMag) / 2

    end
end
"""

distMag(v1::Float64, v2::Float64, v3::Float64)::Float64

Returns the Euclidean norm for a three dimensional vector. 
"""
@inline function distMag(v1::Float64, v2::Float64, v3::Float64)::Float64
	
	return sqrt(v1^2 + v2^2 + v3^2)
end
"""

weakS(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
assemblyInfo::MaxGAssemblyOpts)::Array{ComplexF64,1}

Head function for integration over coincident square panels. The scale 
vector contains the characteristic lengths of a cuboid voxel relative to the 
wavelength. glQuad1 is an array of Gauss-Legendre quadrature weights and 
positions. The assemblyOps parameter determines the level of precision used for 
integral calculations. Namely, assemblyInfo.ordGLIntNear is used internally in 
all weakly singular integral computations. 
"""
function weakS(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Array{ComplexF64,1}

	grdPts = Array{Float64}(undef, 3, 18)
	# Weak self integrals for the three characteristic faces of a cuboid. 
	# dir = 1 -> xy face (z-nrm)   dir = 2 -> xz face (y-nrm) 
	# dir = 3 -> yz face (x-nrm)
	return [weakSDir(3, scale, grdPts, glQuad1, assemblyInfo) +
	rSurfSlf(scale[2], scale[3], assemblyInfo);
	weakSDir(2, scale, grdPts, glQuad1, assemblyInfo) +
	rSurfSlf(scale[1], scale[3], assemblyInfo); 
	weakSDir(1, scale, grdPts, glQuad1, assemblyInfo) + 
	rSurfSlf(scale[1], scale[2], assemblyInfo)]
end
# Weak self-integral of a particular face.
function weakSDir(dir::Int, scale::NTuple{3,Float64}, grdPts::Array{Float64,2},
	glQuad1::Array{Float64,2}, assemblyInfo::MaxGAssemblyOpts)::ComplexF64

	weakGridPts!(dir, scale, grdPts)

	return (((
	weakSInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5]), glQuad1, 
		assemblyInfo) +
	weakSInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4]), glQuad1, 
		assemblyInfo) +
	weakEInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	 	grdPts[:,1], grdPts[:,5], grdPts[:,4]), glQuad1, 
	assemblyInfo) +
	weakEInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
		grdPts[:,1], grdPts[:,2], grdPts[:,5]), glQuad1, 
	assemblyInfo)) + (
	weakSInt(hcat(grdPts[:,4], grdPts[:,1], grdPts[:,2]), glQuad1, 
		assemblyInfo) +
	weakSInt(hcat(grdPts[:,4], grdPts[:,2], grdPts[:,5]), glQuad1, 
		assemblyInfo) +
	weakEInt(hcat(grdPts[:,4], grdPts[:,1], grdPts[:,2], 
	 	grdPts[:,4], grdPts[:,2], grdPts[:,5]), glQuad1, 
	assemblyInfo) +
	weakEInt(hcat(grdPts[:,4], grdPts[:,2], grdPts[:,5], 
		grdPts[:,4], grdPts[:,1], grdPts[:,2]), glQuad1, 
	assemblyInfo))) / 2.0)

end
"""

weakE(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
assemblyInfo::MaxGAssemblyOpts)::Array{ComplexF64,1}	

Head function for integration over edge adjacent square panels. See weakS for 
input parameter descriptions. 
"""
function weakE(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Array{ComplexF64,1}
	
	grdPts = Array{Float64}(undef, 3, 18)
	# Labels are panelDir-panelDir-gridIncrement
	vals = weakEDir(3, scale, grdPts, glQuad1, assemblyInfo)
	# Lower case letters reference the normal directions of the rectangles.
	# Upper case letter reference the increasing axis direction when necessary. 
	xxY = vals[1] + rSurfEdgFlt(scale[1], scale[2], assemblyInfo)
	xxZ = vals[3] + rSurfEdgFlt(scale[1], scale[3], assemblyInfo)
	xyA = vals[2] + rSurfEdgCrn(scale[3], scale[2], scale[1], assemblyInfo)
	xzA = vals[4] + rSurfEdgCrn(scale[2], scale[3], scale[1], assemblyInfo)

	vals = weakEDir(2, scale, grdPts, glQuad1, assemblyInfo)

	yyZ = vals[1] + rSurfEdgFlt(scale[2], scale[3], assemblyInfo)
	yyX = vals[3] + rSurfEdgFlt(scale[2], scale[1], assemblyInfo)
	yzA = vals[2] + rSurfEdgCrn(scale[1], scale[3], scale[2], assemblyInfo)
	xyB = vals[4] + rSurfEdgCrn(scale[3], scale[2], scale[1], assemblyInfo)

	vals = weakEDir(1, scale, grdPts, glQuad1, assemblyInfo)

	zzX = vals[1] + rSurfEdgFlt(scale[3], scale[1], assemblyInfo)
	zzY = vals[3] + rSurfEdgFlt(scale[3], scale[2], assemblyInfo)
	xzB = vals[2] + rSurfEdgCrn(scale[2], scale[3], scale[1], assemblyInfo)
	yzB = vals[4] + rSurfEdgCrn(scale[1], scale[3], scale[2], assemblyInfo)

	return [xxY; xxZ; yyX; yyZ; zzX; zzY; (xyA + xyB) / 2.0; (xzA + xzB) / 2.0;
	(yzA + yzB) / 2.0]
end
#= Weak edge integrals for a given face as specified by dir.

	dir = 1 -> z face -> [y-edge (++ gridX): zz(x), xz(x);
						  x-edge (++ gridY) zz(y) yz(y)]

	dir = 2 -> y face -> [x-edge (++ gridZ): yy(z), yz(z); 
						  z-edge (++ gridX) yy(x) xy(x)]

	dir = 3 -> x face -> [z-edge (++ gridY): xx(y), xy(y); 
						  y-edge (++ gridZ) xx(z) xz(z)]
=#
function weakEDir(dir::Int, scale::NTuple{3,Float64}, grdPts::Array{Float64,2},
	glQuad1::Array{Float64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Array{ComplexF64,1}

	grdPts = Array{Float64}(undef, 3, 18)
	weakGridPts!(dir, scale, grdPts) 

	return [weakEInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5],
	grdPts[:, 2], grdPts[:, 3], grdPts[:, 5]), glQuad1, assemblyInfo) +
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 3], grdPts[:, 6], grdPts[:, 5]), glQuad1, assemblyInfo) +
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4],
	grdPts[:, 2], grdPts[:, 3], grdPts[:, 5]), glQuad1, assemblyInfo) +
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 3], grdPts[:, 6], grdPts[:, 5]), glQuad1, assemblyInfo);
	weakEInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5],
	grdPts[:, 2], grdPts[:, 11], grdPts[:, 5]), glQuad1, assemblyInfo) +
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 11], grdPts[:, 14], grdPts[:, 5]), glQuad1, assemblyInfo) +
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4],
	grdPts[:, 2], grdPts[:, 11], grdPts[:, 5]), glQuad1, assemblyInfo) +
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 11], grdPts[:, 14], grdPts[:, 5]), glQuad1, assemblyInfo);
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 4], grdPts[:, 5], grdPts[:, 7]), glQuad1, assemblyInfo) +
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 8], grdPts[:, 7]), glQuad1, assemblyInfo) +
	weakEInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 4], grdPts[:, 5], grdPts[:, 7]), glQuad1, assemblyInfo) +
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 8], grdPts[:, 7]), glQuad1, assemblyInfo);
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 4], grdPts[:, 5], grdPts[:, 13]), glQuad1, assemblyInfo) +
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 14], grdPts[:, 13]), glQuad1, assemblyInfo) +
	weakEInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 4], grdPts[:, 5], grdPts[:, 13]), glQuad1, assemblyInfo) +
	weakVInt(1, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 14], grdPts[:, 13]), glQuad1, assemblyInfo)]
end
"""

weakV(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
assemblyInfo::MaxGAssemblyOpts)::Tuple{ComplexF64,ComplexF64}	

Head function returning integral values for the Green function over vertex 
adjacent square panels. See weakS for input parameter descriptions. 
"""
function weakV(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Array{ComplexF64,1}

	grdPts = Array{Float64}(undef, 3, 18)
	# Vertex integrals for x-normal face.
	vals = weakVDir(3, scale, grdPts, glQuad1, assemblyInfo)
	xxO = vals[1]
	xyA = vals[2]
	xzA = vals[3]
	# Vertex integrals for y-normal face.
	vals = weakVDir(2, scale, grdPts, glQuad1, assemblyInfo)
	yyO = vals[1]
	yzA = vals[2]
	xyB = vals[3]
	# Vertex integrals for z-normal face.
	vals = weakVDir(1, scale, grdPts, glQuad1, assemblyInfo)
	zzO = vals[1]
	xzB = vals[2]
	yzB = vals[3]
	return[xxO; yyO; zzO; (xyA + xyB) / 2.0; (xzA + xzB) / 2.0; 
	(yzA + yzB) / 2.0]
	
end
#= Weak edge integrals for a given face as specified by dir.
	dir = 1 -> z face -> [zz zx zy]
	dir = 2 -> y face -> [yy yz yx]
	dir = 3 -> x face -> [xx xy xz]
=#
function weakVDir(dir::Int, scale::NTuple{3,Float64}, grdPts::Array{Float64,2},
	glQuad1::Array{Float64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Array{ComplexF64,1}

	weakGridPts!(dir, scale, grdPts) 

	return [weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 6], grdPts[:, 9]), glQuad1, assemblyInfo) +
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 9], grdPts[:, 8]), glQuad1, assemblyInfo) +
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 6], grdPts[:, 9]), glQuad1, assemblyInfo) +
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 9], grdPts[:, 8]), glQuad1, assemblyInfo);
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 17], grdPts[:, 14]), glQuad1, assemblyInfo) +
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 8], grdPts[:, 17]), glQuad1, assemblyInfo) +
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 17], grdPts[:, 14]), glQuad1, assemblyInfo) +
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 8], grdPts[:, 17]), glQuad1, assemblyInfo);
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 15], grdPts[:, 14]), glQuad1, assemblyInfo) +
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 6], grdPts[:, 15]), glQuad1, assemblyInfo) +
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 15], grdPts[:, 14]), glQuad1, assemblyInfo) +
	weakVInt(0, hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 6], grdPts[:, 15]), glQuad1, assemblyInfo)]

end
"""
Generate all unique pairs of cube faces. 
"""
function facePairs()::Array{Int64,2}

	fPairs = Array{Int64,2}(undef, 36, 2)
	
	for i in 1 : 6, j in 1 : 6

		k = (i - 1) * 6 + j
		fPairs[k,1] = i
		fPairs[k,2] = j	
	end
	
	return fPairs
end
"""
Determine appropriate scaling for surface integrals
"""
function surfScale(scaleS::NTuple{3,Float64}, 
	scaleT::NTuple{3,Float64})::Array{Float64,1}

	srcScaling = 1.0
	trgScaling = 1.0
	srfScales = Array{Float64,1}(undef, 36)

	for srcFId in 1 : 6

		if srcFId == 1 || srcFId == 2

			srcScaling = scaleS[2] * scaleS[3]

		elseif srcFId == 3 || srcFId == 4

			srcScaling = scaleS[1] * scaleS[3]

		else

			srcScaling = scaleS[1] * scaleS[2]
		end

		for trgFId in 1 : 6
			
			if trgFId == 1 || trgFId == 2

				trgScaling = scaleT[1]

			elseif trgFId == 3 || trgFId == 4

				trgScaling = scaleT[2]

			else

				trgScaling = scaleT[3]
			end			

			srfScales[(srcFId - 1) * 6 + trgFId] = srcScaling / trgScaling
		end
	end

	return srfScales
end
"""
Generate array of cuboid faces based from a characteristic size, l[]. 
L and U reference relative positions on the corresponding normal axis.
Points are number in a counter-clockwise convention when viewing the 
face from the exterior of the cube. 
"""
function cubeFaces(size::NTuple{3,Float64})::Array{Float64,3}
	
	yzL = hcat([-size[1], -size[2], -size[3]], [-size[1], size[2], -size[3]], 
		[-size[1], size[2], size[3]], [-size[1], -size[2], size[3]]) ./ 2

	yzU = hcat([size[1], -size[2], -size[3]], [size[1], -size[2], size[3]], 
		[size[1], size[2], size[3]], [size[1], size[2], -size[3]]) ./ 2

	xzL = hcat([-size[1], -size[2], -size[3]], [-size[1], -size[2], size[3]], 
		[size[1], -size[2], size[3]], [size[1], -size[2], -size[3]]) ./ 2

	xzU = hcat([-size[1], size[2], -size[3]], [size[1], size[2], -size[3]], 
		[size[1], size[2], size[3]], [-size[1], size[2], size[3]]) ./ 2
	
	xyL = hcat([-size[1], -size[2], -size[3]], [size[1], -size[2], -size[3]], 
		[size[1], size[2], -size[3]], [-size[1], size[2], -size[3]]) ./ 2
	
	xyU = hcat([-size[1], -size[2], size[3]], [-size[1], size[2], size[3]], 
		[size[1], size[2], size[3]], [size[1], -size[2], size[3]]) ./ 2
	
	return cat(yzL, yzU, xzL, xzU, xyL, xyU, dims = 3)
end
"""
Determine a directional component, set by dir, of the separation vector for a 
pair points, set by iP1 and iP2 through glQuad2, for a given pair of source and
target faces.

The relative positions used here are supplied by Gauss-Legendre quadrature, 
glQuad2, with respect to the edge vectors of the cube faces. For more 
information on these quantities, see the gaussQuad2 and cubeFaces functions.  
"""
@inline function cubeVecAlt(dir::Int64, iP1::Int64, iP2::Int64, fp::Int64, 
	glQuad2::Array{Float64,2}, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, fPairs::Array{Int64,2})::Float64
	
	return (trgFaces[dir, 1, fPairs[fp, 1]] +
		glQuad2[iP1, 1] * (trgFaces[dir, 2, fPairs[fp, 1]] - 
			trgFaces[dir, 1, fPairs[fp, 1]]) +
		glQuad2[iP1, 2] * (trgFaces[dir, 4, fPairs[fp, 1]] - 
			trgFaces[dir, 1, fPairs[fp, 1]])) -
	(srcFaces[dir, 1, fPairs[fp,2]] +
		glQuad2[iP2, 1] * (srcFaces[dir, 2, fPairs[fp, 2]] - 
			srcFaces[dir, 1, fPairs[fp, 2]]) +
		glQuad2[iP2, 2] * (srcFaces[dir, 4, fPairs[fp, 2]] - 
			srcFaces[dir, 1, fPairs[fp, 2]]))
end
"""
``Generalized'' version of cubeVecAlt for adaptive implementation. ord 
variables take on values between zero and one. The first pair of entries are 
coordinates in the source surface, the second pair of entries are coordinates 
in the target surface. 
"""
@inline function cubeVecAltAdp(dir::Int64, ordVec::Array{Float64,1}, fp::Int64, 
	trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2})::Float64
	
	return (trgFaces[dir, 1, fPairs[fp, 1]] +
		ordVec[3] * (trgFaces[dir, 2, fPairs[fp, 1]] - 
			trgFaces[dir, 1, fPairs[fp, 1]]) +
		ordVec[4] * (trgFaces[dir, 4, fPairs[fp, 1]] - 
			trgFaces[dir, 1, fPairs[fp, 1]])) -
	(srcFaces[dir, 1, fPairs[fp, 2]] +
		ordVec[1] * (srcFaces[dir, 2, fPairs[fp, 2]] - 
			srcFaces[dir, 1, fPairs[fp, 2]]) +
		ordVec[2] * (srcFaces[dir, 4, fPairs[fp, 2]] - 
			srcFaces[dir, 1, fPairs[fp, 2]]))
end
"""
Create grid point system for calculation for calculation of weakly singular 
integrals. 
"""
function weakGridPts!(dir::Int, scale::NTuple{3,Float64}, 
	grdPts::Array{Float64,2})::Nothing

	if dir == 1

		gridX = scale[1] 
		gridY = scale[2]
		gridZ = scale[3]
	
	elseif dir == 2

		gridX = scale[3] 
		gridY = scale[1]
		gridZ = scale[2]

	elseif dir == 3

		gridX = scale[2] 
		gridY = scale[3]
		gridZ = scale[1]
	else

		error("Invalid direction selection.")

	end

	grdPts[:, 1] = [0.0; 	   	 0.0; 0.0]
	grdPts[:, 2] = [gridX; 	   	 0.0; 0.0]
	grdPts[:, 3] = [2.0 * gridX; 0.0; 0.0]

	grdPts[:, 4] = [0.0;         gridY; 0.0]
	grdPts[:, 5] = [gridX; 		 gridY; 0.0]
	grdPts[:, 6] = [2.0 * gridX; gridY; 0.0]

	grdPts[:, 7] = [0.0; 		 2.0 * gridY; 0.0]
	grdPts[:, 8] = [gridX; 		 2.0 * gridY; 0.0]
	grdPts[:, 9] = [2.0 * gridX; 2.0 * gridY; 0.0]

	grdPts[:, 10] = [0.0; 	 	  0.0; gridZ]
	grdPts[:, 11] = [gridX; 	  0.0; gridZ]
	grdPts[:, 12] = [2.0 * gridX; 0.0; gridZ]

	grdPts[:, 13] = [0.0; 		  gridY; gridZ]
	grdPts[:, 14] = [gridX; 	  gridY; gridZ]
	grdPts[:, 15] = [2.0 * gridX; gridY; gridZ]

	grdPts[:, 16] = [0.0; 		  2.0 * gridY; gridZ]
	grdPts[:, 17] = [gridX;       2.0 * gridY; gridZ]
	grdPts[:, 18] = [2.0 * gridX; 2.0 * gridY; gridZ]

	return nothing
end
#=
The code contained in transformBasisIntegrals evaluates the integrands called 
by the weakS, weakE, and weakV head functions using a series of variable 
transformations and analytic integral evaluations---reducing the four 
dimensional surface integrals performed for ``standard'' cells to chains of one 
dimensional integrals. No comments are included in this low level code, which 
is simply a julia translation of DIRECTFN_E by Athanasios Polimeridis. For a 
complete description of the steps being performed see the article cited above 
and references included therein. 
=#
include("transformBasisIntegrals.jl")
"""
gaussQuad2(ord::Int64)::Array{Float64,2}

Returns locations and weights for 2D Gauss-Legendre quadrature. Order must be 
an integer between 1 and 32, or equal to 64. The first column of the returned 
array is the ``x-position'', on the interval [0,1]. The second column is the 
``y-positions'', also on the interval [0,1]. The third is column is the 
evaluation weights. 
"""
function gaussQuad2(ord::Int64)::Array{Float64,2}
	
	glQuad2 = Array{Float64,2}(undef, ord * ord, 3)
	glQuad1 = gaussQuad1(ord)

	for j in 1:ord, i in 1:ord

		glQuad2[i + (j - 1) * ord, 1] = (glQuad1[i, 1] + 1.0) / 2.0
		glQuad2[i + (j - 1) * ord, 2] = (glQuad1[j, 1] + 1.0) / 2.0
		glQuad2[i + (j - 1) * ord, 3] = glQuad1[i, 2] * glQuad1[j, 2] / 4.0
	end
	
	return glQuad2
end
#=
gaussQuad1(ord::Int64)::Array{Float64,2}

Returns locations and weights for 1D Gauss-Legendre quadrature. Order must be  
an integer between 1 and 64. The first column of the returned 
array is positions, on the interval [-1,1], the second column contains the 
associated weights.
=#
# include("glQuads.jl")
# gausschebyshev(), gausslegendre(), gaussjacobi(), gaussradau(), 
# gausslobatto(), gausslaguerre(), gausshermite()
function gaussQuad1(ord::Int64)::Array{Float64,2}

	pos, val = gausslegendre(ord)

	return [pos ;; val]
end
#= 
Direct evaluation of 1 / (4 * π * distMag) integral for a square panel with 
itself. la and lb are the edge lengths. 
=#
@inline function rSurfSlf(la::Float64, lb::Float64, 
	assemblyInfo::MaxGAssemblyOpts)::ComplexF64

    return (1 / (48 * π * assemblyInfo.freqPhase^2)) * (8 * la^3 + 8 * lb^3 
    - 8 * la^2 * sqrt(la^2 + lb^2) - 8 * lb^2 * sqrt(la^2 + lb^2) - 
    3 * la^2 * lb * (2 * log(la) + 2 * log(la + lb - sqrt(la^2 + lb^2)) + 
    log(sqrt(la^2 + lb^2) - lb) - 5 * log(lb + sqrt(la^2 + lb^2)) - 
    2 * log(lb - la + sqrt(la^2 + lb^2)) - 
    2 * log(la + 2 * lb - sqrt(la^2 + 4 * lb^2)) + 
    log(sqrt(la^2 + 4 * lb^2) - 2 * lb) + 
    2 * log(la - 2 * lb + sqrt(la^2 + 4 * lb^2)) + 
    log(2 * lb + sqrt(la^2 + 4 * lb^2)) + 
    2 * log(2 * lb - la + sqrt(la^2 + 4 * lb^2)) - 
    2 * log(la + 2 * lb + sqrt(la^2 + 4 * lb^2))) + 6 * la * lb^2 * 
    (log(64) + 4 * log(lb) + 2 * log(sqrt(la^2 + lb^2) - la) + 
    3 * log(la + sqrt(la^2 + lb^2)) - 3 * log(sqrt(la^2 + 4 * lb^2) - la) - 
    3 * log(sqrt(la^4 + 5 * la^2 * lb^2 + 4 * lb^4) + 
    la * (sqrt(la^2 + lb^2) - la - sqrt(la^2 + 4 * lb^2)))))
end
#= 
Direct evaluation of 1 / (4 * π * distMag) integral for a pair of cornered edge 
panels. la, lb, and lc are the edge lengths, and la is assumed to be common to 
both panels. 
=#
@inline function rSurfEdgCrn(la::Float64, lb::Float64, lc::Float64, 
	assemblyInfo::MaxGAssemblyOpts)::ComplexF64

   	return (1 / (48 * π * assemblyInfo.freqPhase^2)) * (8 * lb * lc * 
   	sqrt(lb^2 + lc^2) - 8 * lb * lc * sqrt(la^2 + lb^2 + lc^2) - 12 * la^3 * 
    acot(la * lc / (la^2 + lb^2 - lb * sqrt(la^2 + lb^2 + lc^2))) + 
    12 * la^3 * atan(la / lc) - 
    12 * la * lc^2 * atan(la * lb / (lc * sqrt(la^2 + lb^2 + lc^2))) - 
    12 * la * lb^2 * atan(la * lc / (lb * sqrt(la^2 + lb^2 + lc^2))) - 
    16 * la^3 * atan(lb * lc / (la * sqrt(la^2 + lb^2 + lc^2))) + 
    6 * lc^3 * atanh(lb / sqrt(lb^2 + lc^2)) - 
    6 * lc * (la^2 + lc^2) * atanh(lb / sqrt(la^2 + lb^2 + lc^2)) - 
    15 * la^2 * lc * log(la^2 + lc^2) - lc^3 * log(la^2 + lc^2) + 
    2 * lc^3 * log(lc / (lb + sqrt(lb^2 + lc^2))) + 
    6 * la^2 * lc * log(sqrt(la^2 + lb^2 + lc^2) - lb) + 
    24 * la^2 * lc * log(sqrt(la^2 + lb^2 + lc^2) + lb) + 
    2 * lc^3 * log(sqrt(la^2 + lb^2 + lc^2) + lb) + 
    6 * la * lb * (-2 * la * log(la^2 + lb^2) - 
    lc * log((lb^2 + lc^2) * (sqrt(la^2 + lb^2 + lc^2) - la)) + 
    3 * lc * log(la + sqrt(la^2 + lb^2 + lc^2)) + 
    la * log(sqrt(la^2 + lb^2 + lc^2) - lc) + 
    3 * la * log(sqrt(la^2 + lb^2 + lc^2) + lc)) + 
    2 * lb^3 * (
    log((sqrt(la^2 + lb^2 + lc^2) - lc) / (lc + sqrt(la^2 + lb^2 + lc^2))) + 
    log(1 + (2 * lc * (lc + sqrt(lb^2 + lc^2))) / lb^2)))
end
#= 
Direct evaluation of 1 / (4 * π * distMag) integral for a pair of flat edge 
panels. la and lb are the edge lengths, and lb is assumed to be ``doubled''. 
=#
@inline function rSurfEdgFlt(la::Float64, lb::Float64, 
	assemblyInfo::MaxGAssemblyOpts)::ComplexF64
       
    return (1 / (12 * π * assemblyInfo.freqPhase^2)) * (-la^3 + 2 * lb^2 * 
    (3 * lb + sqrt(la^2 + lb^2) - 2 * sqrt(la^2 + 4 * lb^2)) + la^2 * 
    (2 * sqrt(la^2 + lb^2) - sqrt(la^2 + 4 * lb^2))) + 
    (1 / (64 * π)) * la * lb * (lb * (-62 * log(2) - 
    5 * log(-la + sqrt(la^2 + lb^2)) + 
    4 * log(8 * lb^2 * (-la + sqrt(la^2 + lb^2))) - 
    33 * log(la + sqrt(la^2 + lb^2)) + 17 * log(-la + sqrt(la^2 + 4 * lb^2)) - 
    24 * log(lb * (-la + sqrt(la^2 + 4 * lb^2))) + 
    57 * log(la + sqrt(la^2 + 4 * lb^2))) + 
    4 * la * (-8 * asinh(lb / la) + 6 * asinh(2 * lb / la) + 
    6 * atanh(lb / sqrt(la^2 + lb^2)) + 12 * log(la) - 
    13 * log(-lb + sqrt(la^2 + lb^2)) + log((-lb + sqrt(la^2 + lb^2)) / la) + 
    log(la / (lb + sqrt(la^2 + lb^2))) - 7 * log(lb + sqrt(la^2 + lb^2)) - 
    2 * log((lb + sqrt(la^2 + lb^2))/la) - 
    3 * log(-(((lb + sqrt(la^2 + lb^2)) * 
    (2 * lb - sqrt(la^2 + 4 * lb^2))) / (la^2))) - 
    3 * log((-lb + sqrt(la^2 + lb^2)) / (-2 * lb + sqrt(la^2 + 4 * lb^2))) + 
    11 * log(-2 * lb + sqrt(la^2 + 4 * lb^2)) - 
    3 * log((lb + sqrt(la^2 + lb^2)) / (2 * lb + sqrt(la^2 + 4 * lb^2))) + 
    log(2 * lb + sqrt(la^2 + 4 * lb^2)) + 
    9 * log((2 * lb + sqrt(la^2 + 4 * lb^2)) / (lb + sqrt(la^2 + lb^2))) - 
    2 * log(la^2 + 2 * lb * (lb - sqrt(la^2 + lb^2)))))
end
end