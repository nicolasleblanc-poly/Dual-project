module Av_product_Davidson 

using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, bfgs_power_iteration_asym_only, dual_asym_only, gmres,
phys_setup, opt_setup

export A_v_product 

function A_v_product(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,alpha,vec)
    chi_inv_coeff_dag = conj(chi_inv_coeff)
    alpha_coeff = (alpha/(2im))[1]
    G_v_product = Gv_AA(gMemSlfN, cellsA, vec)
    term_1 = (P/2)*G_v_product
    term_2 = (1/2)*GAdjv_AA(gMemSlfA, cellsA, conj.(transpose(P))*vec)
    term_3 = alpha_coeff.*G_v_product
    # print("size(alpha_coeff) ", size(alpha_coeff), "\n")
    # print("size(chi_inv_coeff) ", size(chi_inv_coeff), "\n")
    # print("size(vec) ", size(vec), "\n")
    # print("alpha_coeff ", alpha_coeff, "\n")
    term_4 = alpha_coeff.*chi_inv_coeff*vec
    term_5 = alpha_coeff.*GAdjv_AA(gMemSlfA, cellsA, vec)
    term_6 = alpha_coeff.*chi_inv_coeff_dag*vec
    return term_1+term_2+term_3+term_4+term_5+term_6
end 
end 