"""
The product module contains the functions to do the product between the 
different Green functions and a vector. 

Author: Nicolas Leblanc
"""

module product_sym_and_asym
# using Random, MaxGOpr, FFTW, Distributed, MaxGParallelUtilities, MaxGStructs, 
# MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, MaxGCUDA,  MaxGOpr


using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random

export Gv_AA,GAdjv_AA,Gv_AB,A,asym,sym,asym_vect,sym_vect,sym_and_asym_sum


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
	# Source B
	for dirItr in 1 : 3
		for itrZ in 1 : srcSizeAB[3], itrY in 1 : srcSizeAB[2], 
			itrX in 1 : srcSizeAB[1]
			currSrcAB[itrX,itrY,itrZ,dirItr] = 0.0 + 0.0im
		end
	end
	currSrcAB[1,1,1,3] = 10.0 + 0.0im
	copyto!(gMemExtN.srcVec, currSrcAB)
	# Apply the Green operator 
	grnOpr!(gMemExtN) # Same thing as greenActAB!
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
	chi_inv_coeff_dag = conj(chi_inv_coeff)
	term_1 = chi_inv_coeff_dag*vec # chi_inv_coeff_dag*P*vec
	term_2 = GAdjv_AA(gMemSlfA, cellsA, vec) # P*GAdjv_AA(gMemSlfA, cellsA, vec)
	term_3 = chi_inv_coeff*vec # chi_inv_coeff*vec*adjoint(P) # supposed to be *conj.(transpose(P)) but gives error, so let's use *P for now
	term_4 = Gv_AA(gMemSlfN, cellsA, vec) # Gv_AA(gMemSlfN, cellsA, adjoint(P)*vec)
	return (term_1.-term_2.+ term_4.-term_3)./2im # 
end 

function sym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,vec)
	chi_inv_coeff_dag = conj(chi_inv_coeff)
	term_1 = chi_inv_coeff_dag*vec # chi_inv_coeff_dag*P*vec
	term_2 = GAdjv_AA(gMemSlfA, cellsA, vec) # P*GAdjv_AA(gMemSlfA, cellsA, vec)
	term_3 = chi_inv_coeff*vec # chi_inv_coeff*vec*adjoint(P) # supposed to be *conj.(transpose(P)) but gives error, so let's use *P for now
	term_4 = Gv_AA(gMemSlfN, cellsA, vec) # Gv_AA(gMemSlfN, cellsA, adjoint(P)*vec)
	return (term_1.-term_2.- term_4.+term_3)./2  
end

# New sym and asym sum function: 
function sym_and_asym_sum(xi,l,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, vec)	
	
	# For 3 of the 4 terms in sym and in asym, there is a P|v> product. 
	# We can therefore just compute this product once for each P. This 
	# avoids calculating twice (one for sym and once for asym). It 
	# doesn't seem like it would make much of a different but 
	# doing something twice a lot of times kinda makes a big/huge difference.

	# P_sum_asym = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3,cellsA[1]*cellsA[2]*cellsA[3]*3)
    # # P sum asym
    # if length(l) > 0
    #     for j in eachindex(l)
	# 		if length(P) > 1
	# 			P_sum_asym += (l[j])*P[1][j]
	# 		else
	# 			P_sum_asym += (l[j])*P[j]
	# 		end 
            
    #     end 
    # end 

	P_asym = xi*P[1]
	# P sum sym
	P_sum_sym = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3,cellsA[1]*cellsA[2]*cellsA[3]*3)
	if length(l) > 0
		P_sum_sym += (l[j])*P[1+j]
		# We no longer need the code below since we only have the one 
		# Asym constraint 
		# l used to be for the Asym constraints and l2 for the Sym constraints 
        # for j in eachindex(l)
		# 	if length(P) > 1
		# 		P_sum_sym += (l2[j])*P[1][Int(length(l))+j]
		# 	else
		# 		P_sum_sym += (l2[j])*P[Int(length(l))+j]
		# 	end 
        # end 
    end 
	P_sum_sym_adjoint = adjoint(P_sum_sym)
	P_asym_vec_product = xi*P_asym*vec
	P_asym_adjoint_vec_product = xi*adjoint(P_asym)*vec
	P_sum_sym_vec_product = P_sum_sym*vec
	P_sum_sym_adjoint_vec_product = P_sum_sym_adjoint*vec

	chi_inv_coeff_dag = conj(chi_inv_coeff)
	term1 = ((chi_inv_coeff_dag*P_asym_vec_product-chi_inv_coeff*P_asym_adjoint_vec_product)/2im) + ((chi_inv_coeff_dag*P_sum_sym_vec_product+chi_inv_coeff*P_sum_sym_adjoint_vec_product)/2)
	term2 = -(P_asym/2im+P_sum_sym/2)*GAdjv_AA(gMemSlfN, cellsA, vec)
	term3 = Gv_AA(gMemSlfA, cellsA, P_asym_adjoint_vec_product/2im - P_sum_sym_adjoint_vec_product/2)
 
	return term1 + term2 + term3

end 
end
