module product 
# using Random, MaxGOpr, FFTW, Distributed, MaxGParallelUtilities, MaxGStructs, 
# MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, MaxGCUDA,  MaxGOpr


using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random

# using MaxGOpr
# using Distributed
# @everywhere using Printf, Base.Threads, LinearAlgebra.BLAS, MaxGParallelUtilities, MaxGStructs, 
# MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, MaxGCUDA,  MaxGOpr, Random, FFTW

export Gv_AA, GAdjv_AA, Gv_AB, A, asym, sym, asym_vect, sym_vect# , test

# Green function test function 
# function test(gMemSlfN, vect)
# 	copyto!(gMemSlfN.srcVec, vect)
# 	print("gMemSlfN.srcVec ",gMemSlfN.srcVec, "\n")
# 	grnOpr!(gMemSlfN) # Same thing as greenActAA!()
# 	print("gMemSlfN.trgVec ",gMemSlfN.trgVec, "\n")
# end


# Function of the (G_{AA})v product 
function Gv_AA(gMemSlfN, cellsA, vec) # offset
	# Reshape the input vector 
	reshaped_vec = reshape(vec, (cellsA[1],cellsA[2],cellsA[3],3))
	copyto!(gMemSlfN.srcVec, reshaped_vec)
	grnOpr!(gMemSlfN) # Same thing as greenActAA!()
	mode = 1
    return reshape(gMemSlfN.trgVec + G_offset(reshaped_vec, mode), (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # reshape the output vector  
end 


# Function of the (G_{AB}^A)v product 
function GAdjv_AA(gMemSlfA, cellsA, vec) # , offset
	# Reshape the input vector 
	reshaped_vec = reshape(vec, (cellsA[1],cellsA[2],cellsA[3],3))
	copyto!(gMemSlfA.srcVec, reshaped_vec)
	grnOpr!(gMemSlfA) # Same thing as greenAdjActAA!()
	mode = 2 
    return reshape(gMemSlfA.trgVec + G_offset(reshaped_vec, mode), (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # reshape the output vector 
end

# Function of the (G_{AB})v product 
function Gv_AB(gMemExtN, cellsA, cellsB, currSrcAB) # offset
	# No need to reshape the input vector currScrAB here because 
	# of how it was defined in main. 
	## Size settings for external Green function
	srcSizeAB = (cellsB[1], cellsB[2], cellsB[3])
	# crcSizeAB = (cellsA[1] + cellsB[1], cellsA[2] + cellsB[2], cellsA[3] + cellsB[3])
	# trgSizeAB = (cellsA[1], cellsA[2], cellsA[3])
	# Source B
	for dirItr in 1 : 3
		for itrZ in 1 : srcSizeAB[3], itrY in 1 : srcSizeAB[2], 
			itrX in 1 : srcSizeAB[1]
			currSrcAB[itrX,itrY,itrZ,dirItr] = 0.0 + 0.0im
		end
	end
	# elecIn = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
	currSrcAB[1,1,1,3] = 10.0 + 0.0im
	copyto!(gMemExtN.srcVec, currSrcAB)
	# Apply the Green operator 
	grnOpr!(gMemExtN) # Same thing as greenActAB!
	# copy!(elecIn,currTrgAB);
	# This will do
	mode = 1
    return reshape(gMemExtN.trgVec, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) #currTrgAB #+ G_offset(currSrcAB, mode) # I assume I have to add the offset like for the AA G func 
end 

function G_offset(v, mode) # offset, 
	offset = 1e-4im
    if mode == offset # Double check this part of the code with Sean because I feel like the condition
		# mode == offset is never met seeing that mode is either 1 or 2.  
		return (offset).*v
	else 
		return conj(offset).*v
	end
end

function asym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,vec)
	# print("size(vect)", size(vect), "\n")
	chi_inv_coeff_dag = conj(chi_inv_coeff)
	term_1 = chi_inv_coeff_dag*vec # *P 
	term_2 = GAdjv_AA(gMemSlfA, cellsA, vec) # P*
	term_3 = chi_inv_coeff*vec # *P # supposed to be *conj.(transpose(P)) but gives error, so let's use *P for now
	term_4 = Gv_AA(gMemSlfN, cellsA, vec) # P*
	# print("size(term_1)", size(term_1), "\n")
	# print("size(term_2)", size(term_2), "\n")
	# print("term_1 " , term_1, "\n")
	# print("term_2 " , term_2, "\n")
	# print("term_3 " , term_3, "\n")
	# print("term_4 " , term_4, "\n")
	return (term_1.-term_2.+ term_4.-term_3)./2im # 
end 

function sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,vec)
	# print("size(vect)", size(vect), "\n")
	chi_inv_coeff_dag = conj(chi_inv_coeff)
	term_1 = chi_inv_coeff_dag*vec # *P 
	term_2 = GAdjv_AA(gMemSlfA, cellsA, vec) # P*
	term_3 = chi_inv_coeff*vec # *P # supposed to be *conj.(transpose(P)) but gives error, so let's use *P for now
	term_4 = Gv_AA(gMemSlfN, cellsA, vec) # P*
	# print("size(term_1)", size(term_1), "\n")
	# print("size(term_2)", size(term_2), "\n")
	# print("term_3 " , term_3, "\n")
	return (term_1.-term_2.- term_4.+term_3)./2  
end


end

# Old code and test code below 

# Old code 
# Start of old code 
# function A(greenCircAA, cellsA, chi_inv_coeff, l, P, vect)
# 	# For when we will also consider the symmetric part + l[2]sym(greenCircAA, cellsA, chi_inv_coeff, P))
# 	return l[1]*(asym(greenCircAA, cellsA, chi_inv_coeff, P))
# end

# function asym(greenCircAA, cellsA, chi_inv_coeff, P)
# 	chi_inv_coeff_dag = conj(chi_inv_coeff)
# 	# print("chi_inv_coeff_dag ",chi_inv_coeff_dag, "\n")
# 	term_1 = chi_inv_coeff_dag*P 
# 	term_2 = GAdjv_AA(greenCircAA, cellsA, P)
# 	term_3 = chi_inv_coeff*conj.(transpose(P))
# 	term_4 = Gv_AA(greenCircAA, cellsA, P)
# 	return (term_1-term_2-term_3+term_4)/2im
# end

# function sym(greenCircAA, cellsA, chi_inv_coeff, P)
# 	chi_inv_coeff_dag = conj(chi_inv_coeff)
# 	term_1 = chi_inv_coeff_dag*P 
# 	term_2 = GAdjv_AA(greenCircAA, cellsA, P)
# 	term_3 = chi_inv_coeff*conj.(transpose(P))
# 	term_4 = Gv_AA(greenCircAA, cellsA, P)
# 	return (term_1-term_2-term_3+term_4)/2
# end
# End of old code 

# Test code 
# Start of test code 
# # Define test volume, all lengths are defined relative to the wavelength. 
# # Number of cells in the volume. 
# cellsA = [16, 16, 16]
# cellsB = [1, 1, 1]
# currSrc = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)

# # Edge lengths of a cell relative to the wavelength. 
# scaleA = (0.02, 0.02, 0.02)
# scaleB = (0.02, 0.02, 0.02)
# # Center position of the volume. 
# coordA = (0.0, 0.0, 0.0)
# coordB = (-0.3, 0.3, 0.0)
# # Create MaxG volumes.
# volA = genMaxGVol(MaxGDom(cellsA, scaleA, coordA))
# volB = genMaxGVol(MaxGDom(cellsB, scaleB, coordB))
# # Information for Green function construction. 
# # Complex frequency ratio. 
# freqPhase = 1.0 + im * 0.0
# # Gauss-Legendre approximation orders. 
# ordGLIntFar = 2
# ordGLIntMed = 8
# ordGLIntNear = 16
# # Cross over points for Gauss-Legendre approximation.
# crossMedFar = 16
# crossNearMed = 8
# assemblyInfo = MaxGAssemblyOpts(freqPhase, ordGLIntFar, ordGLIntMed, 
# 	ordGLIntNear, crossMedFar, crossNearMed)
# # Pre-allocate memory for circulant green function vector. 
# # Let's say we are only considering the AA case for simplicity
# greenCircAA = Array{ComplexF64}(undef, 3, 3, 2 * cellsA[1], 2 * cellsA[2], 
# 	2 * cellsA[3])
# greenCircBA = Array{ComplexF64}(undef, 3, 3, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# Nic Test:
# CPU computation of Green function
# genGreenSlf!(greenCircAA, volA, assemblyInfo)

# rand!(currSrc)

# print(greenAct!())

# print(Gv(greenCircAA, cellsA, currSrc))
# End of test code 