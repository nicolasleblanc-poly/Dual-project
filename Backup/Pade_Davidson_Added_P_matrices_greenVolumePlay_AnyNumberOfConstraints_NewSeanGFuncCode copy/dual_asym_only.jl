module dual_asym_only
export dual, c1
using b_asym_only, gmres, product
# Code to get the value of the objective and of the dual.
# The code also calculates the constraints
# and can return the gradient.
function c1(l,l2,P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff,GAdj_T,i) # asymmetric part 
    # OG code 
    # Left term
    # print("In c1 \n")
    chi_inv_coeff_dag = conj(chi_inv_coeff)
    if length(P) > 1
        PT = P[1][i]*T 
    else
        PT = P[i]*T 
    end
    # PT = P[1][i]*T 
    # first term
    ei_tr = conj.(transpose(ei))
    EPT=ei_tr*PT
    I_EPT = imag(EPT) 
    print("I_EPT ", I_EPT, "\n")
    # second term 
    term2 = conj.(transpose(T))*((chi_inv_coeff_dag-chi_inv_coeff)/2im)*PT
    # third term 
    # term3 = 0
    if length(P) > 1
        term3 = conj.(transpose(T))*((P[1][i]/2im)*GAdj_T)
    else
        term3 = conj.(transpose(T))*((P[i]/2im)*GAdj_T)
    end
    # term3 = conj.(transpose(T))*((P[1][i]/2im)*GAdj_T)
    # fourth term 
    term4 = (conj.(transpose(T)))*Gv_AA(gMemSlfA, cellsA, PT/2im)
    TasymT = I_EPT-(term2-term3+term4)
    print("TasymT ", TasymT, "\n")
    
    return real(I_EPT - TasymT)[1] 


     # New code that might actually not work for what we need with the gradient...
    # P_sum_asym = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3,1)
    # # P sum
    # if length(l) > 0
    #     for j in eachindex(l)
    #         P_sum_asym += (l[j])*P[j]
    #     end 
    # end 
    # P_sum_asym_T_product = P_sum_asym.*T

    # # Left term 
    # ei_tr = conj.(transpose(ei))
    # EPT=ei_tr*P_sum_asym_T_product
    # I_EPT = imag(EPT) 
    # print("I_EPT ", I_EPT, "\n")

    # # Right term => Sym*T
    # chi_inv_coeff_dag = conj(chi_inv_coeff)
    # first_term = ((chi_inv_coeff_dag+chi_inv_coeff)/2im)*P_sum_asym_T_product
    # second_term = (P_sum_asym/2).*GAdjv_AA(gMemSlfN, cellsA, T)
    # third_term = Gv_AA(gMemSlfA, cellsA, P_sum_asym_T_product/2)
    # # second_term = GAdjv_AA(gMemSlfA, cellsA, P_sum_asym_T_product/2im)
    # # third_term = (P_sum_asym/2im).*Gv_AA(gMemSlfN, cellsA, T)
    
    # total_sum = I_EPT - conj.(transpose(T))*(first_term-second_term+third_term)


    # # G|v> type calculation
    # # symT = sym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, T)
    # # TsymT = conj.(transpose(T))*symT
    # # print("T_sym_T ", TsymT, "\n")
    # # return real(I_EPT - TsymT)[1] 
    # return real(total_sum) # [1]

    # PT = T  # P*
    # ei_tr = transpose(ei) # we have <ei^*| instead of <ei|
    # print("size(ei) ", size(ei), "\n")
    # print("size(T) ", size(T), "\n")
    

    # print("l ", l, "\n")

    # print("T ", T, "\n")

    # print("T-T inner product ", conj.(transpose(T))*T, "\n")


    
    # Right term => Asym*T
    # G|v> type calculation

    # print("l ", l, "\n")

    # asymT = asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, T)
    # # print("asymT ", asymT, "\n")
    # # output(l,T,fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)/l[1]
    # # print("asym_T", asym_T, "\n")
    # TasymT = conj.(transpose(T))*asymT
    # print("T_asym_T ", TasymT, "\n")
    # print("I_EPT ", I_EPT,"\n")
    # print("T_asym_T ", T_asym_T,"\n")
    # print("I_EPT - T_chiGA_T",I_EPT - T_chiGA_T,"\n")
    # return real(I_EPT - T_asym_T[1]) # for the <ei^*| case
    # return real(I_EPT - TasymT)[1] 
end

function c2(l,l2,P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff,GAdj_T,j) # symmetric part 
    # OG code 
    # Left term
    # print("In c1 \n")
    chi_inv_coeff_dag = conj(chi_inv_coeff)
    if length(P) > 1
        PT = P[1][j]*T 
    else
        PT = P[j]*T 
    end
    # PT = P[1][j]*T 
    # first term
    ei_tr = conj.(transpose(ei))
    EPT=ei_tr*PT
    I_EPT = real(EPT) 
    print("I_EPT ", I_EPT, "\n")
    # second term 
    term2 = conj.(transpose(T))*((chi_inv_coeff_dag-chi_inv_coeff)/2)*PT
    # third term 
    if length(P) > 1
        term3 = conj.(transpose(T))*((P[1][j]/2)*GAdj_T)
    else
        term3 = conj.(transpose(T))*((P[j]/2)*GAdj_T)
    end
    # term3 = conj.(transpose(T))*((P[1][j]/2)*GAdj_T)
    # fourth term 
    term4 = (conj.(transpose(T)))*Gv_AA(gMemSlfA, cellsA, PT/2)
    TsymT = I_EPT-(term2-term3-term4)
    print("TsymT ", TsymT, "\n")
    
    return real(I_EPT - TsymT)[1] 

    #     # New code that might actually not work for what we need with the gradient...
# #     P_sum_sym = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3,1)
# #     # P sum
# #     if length(l2) > 0
# #         for j in eachindex(l2)
# #             P_sum_sym += (l2[j])*P[Int(length(l))+j]
# #         end 
# #     end 
# #     P_sum_sym_T_product = P_sum_sym.*T

# #     # Left term 
# #     ei_tr = conj.(transpose(ei))
# #     EPT=ei_tr*P_sum_sym_T_product
# #     I_EPT = real(EPT) 
# #     print("I_EPT ", I_EPT, "\n")

# #     # Right term => Sym*T
# #     chi_inv_coeff_dag = conj(chi_inv_coeff)
# #     first_term = ((chi_inv_coeff_dag+chi_inv_coeff)/2)*P_sum_sym_T_product
# #     second_term = (P_sum_sym/2).*GAdjv_AA(gMemSlfN, cellsA, T)
# #     third_term = Gv_AA(gMemSlfA, cellsA, P_sum_sym_T_product/2)
    
# #     total_sum = I_EPT - conj.(transpose(T))*(first_term-second_term-third_term)


# #     # G|v> type calculation
# #     # symT = sym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, T)
# #     # TsymT = conj.(transpose(T))*symT
# #     # print("T_sym_T ", TsymT, "\n")
# #     # return real(I_EPT - TsymT)[1] 
# #     return real(total_sum) # [1]

#     # OG code 
#     # Left term
#     # print("In c1 \n")
#     PT = T  # P*
#     # ei_tr = transpose(ei) # we have <ei^*| instead of <ei|\
#     # print("size(ei) ", size(ei), "\n")
#     # print("size(T) ", size(T), "\n")
#     ei_tr = conj.(transpose(ei))
#     # print("T ", T, "\n")

#     # print("T-T inner product ", conj.(transpose(T))*T, "\n")

#     EPT=ei_tr*PT
#     I_EPT = real(EPT) 
#     print("I_EPT ", I_EPT, "\n")
#     # Right term => asym*T
    
#     # G|v> type calculation

#     # print("l ", l, "\n")

#     symT = sym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, T)
#     # print("asymT ", asymT, "\n")
#     # output(l,T,fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)/l[1]
#     # print("asym_T", asym_T, "\n")
#     TsymT = conj.(transpose(T))*symT
#     print("T_sym_T ", TsymT, "\n")
#     # print("I_EPT ", I_EPT,"\n")
#     # print("T_asym_T ", T_asym_T,"\n")
#     # print("I_EPT - T_chiGA_T",I_EPT - T_chiGA_T,"\n")
#     # return real(I_EPT - T_asym_T[1]) # for the <ei^*| case
#     return real(I_EPT - TsymT)[1] 
end

# Code for the dual value calculation without using the gradient. The 
# gradient is still just calculated because it is needed for BFGS. If 
# another method is used, we wouldn't need the gradient. 
function dual(l,l2,g,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff, cellsA,fSlist,get_grad)
    b = bv_asym_only(ei, l, l2, P) 
    print("l ", l, "\n")
    print("l2 ", l2, "\n")
    print("b ", b, "\n")
    print("size(b) ", size(b), "\n")
    # l = [2] # initial Lagrange multipliers


    P_sum_asym = zeros(ComplexF64, cellsA[1]*cellsA[2]*cellsA[3]*3,cellsA[1]*cellsA[2]*cellsA[3]*3)
    # print("size(P_sum_asym) ", size(P_sum_asym), "\n")
    # P sum asym
    # print("P[1][1] ", P[1][1], "\n")
    # print("size(P[1][1]) ", size(P[1][1]), "\n")
    print("length(P) ", length(P), "\n")
    if length(l) > 0
        for j in eachindex(l)
            if length(P) > 1
				P_sum_asym += (l[j])*P[1][j]
			else
				P_sum_asym += (l[j])*P[j]
			end
            # P_sum_asym .+= (l[j])*P[1][j]
        end 
    end  
	# P sum sym
	P_sum_sym = zeros(ComplexF64, cellsA[1]*cellsA[2]*cellsA[3]*3,cellsA[1]*cellsA[2]*cellsA[3]*3)
	if length(l2) > 0
        for j in eachindex(l)
            if length(P) > 1
				P_sum_sym += (l2[j])*P[1][Int(length(l))+j]
			else
				P_sum_sym += (l2[j])*P[Int(length(l))+j]
			end
            # P_sum_sym += (l2[j])*P[1][Int(length(l))+j]
        end 
    end 
    P_sum = P_sum_asym + P_sum_sym

    # P_sum_total = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3,1)
    # # P su
    # for i in range(1,length(l)+length(l2))
    #     P_sum_total += P[i]
    # end 
    
    # When GMRES is used as the T solver
    T = GMRES_with_restart(l, l2, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

    # Psum_T_product = P_sum_total.*T

    # ei_tr = transpose(ei)
    ei_tr = conj.(transpose(ei)) 
    k0 = 2*pi
    Z = 1
    # I put the code below here since it is used no matter the lenght of fSlist
    # ei_T=ei_tr*T

    # To get the just the dual value if we don't need the gradients, 
    # we just need to calculate D = (1/4)*Re{<s_lambda|T>} 
    # where |s_lambda> = lambda_j P_j |e_i>

    s_lambda = P_sum*ei
    obj = (1/4)*real(conj.(transpose(s_lambda))*T)
    D = obj[1]
    print("obj ", obj,"\n")

    # ei_P_T=ei_tr*Psum_T_product
    # obj = imag(ei_P_T)[1]  # this is just the objective part of the dual 0.5*(k0/Z)*
    # print("obj ", obj, "\n")
    # D = obj 
    # Reminder: a gradient is just a constraint evaluated a |t>.

    GAdj_T = GAdjv_AA(gMemSlfN, cellsA, T)

    g = ones(Float64, length(l), 1)
    g2 = ones(Float64, length(l2), 1)
    # Calculation and storing of the gradients 
    if length(l)>0
        print("Asym constraints only \n")
        for i in eachindex(l)
            g[i] = c1(l,l2,P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff,GAdj_T,i)
        end
        print("g ", g, "\n")
    end 
    if length(l2)>0 
        print("Sym constraints only \n")
        for j in eachindex(l2)
            g2[j] = c2(l,l2,P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff,GAdj_T,j)
        end 
        print("g2 ", g2, "\n")
    end 

    # # Code to add gradients to the dual value 
    # if length(l)>0 
    #     for i in range(1,length(l), step=1) 
    #         D += l[i]*g[i]
    #     end 
    # end 
    # if length(l2)>0 
    #     for j in range(1,length(l2), step=1)
    #         D += l2[j]*g2[j]
    #     end 
    # end 


    # When conjugate gradient is used as the T solver 
    # T = cg(l, l2, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

    # When biconjugate gradient is used as the T solver 
    # T = bicg(l, l2, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

    # When stabilized biconjugate gradient is used as the T solver 
    # T = bicgstab(l, l2, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)
    
    # print("D after adding grad ", D, "\n")
    print(length(fSlist), "\n")
    if length(fSlist)>0
        print("In fSlist loop \n")
        fSval = 0
        for k in fSlist
            prod_k = sym_and_asym_sum(l,l2,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
            # Asym_k = asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
            # Sym_k = sym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
            k_tr = conj.(transpose(k)) 
            # kAsymk = l[1]*k_tr*Asym_k
            # kSymk = l[2]*k_tr*Sym_k
            k_prod_k = k_tr*prod_k
            fSval += real(k_prod_k[1])
            # fSval += real(kAsymk[1]+kSymk[1])

            # A_k = asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
            # k_tr = conj.(transpose(k)) 
            # kAk=k_tr*A_k
            # fSval += real(kAk[1])
        end
        D += fSval
    end
    gradient= vcat(g,g2) # Combine the sym and asym L mults into one list

    print("dual ", D,"\n")
    # print("Done dual \n")
    if get_grad == true
        return real(D[1]), gradient, real(obj) 
    elseif get_grad == false
        return real(D[1]), real(obj) 
    end
end
end

# This is the code for the dual value calculation when we just the gradients times
# the multiplier values. 
# function dual(l,l2,g,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff, cellsA,fSlist,get_grad)
#     b = bv_asym_only(ei, l, l2, P) 
#     print("l ", l, "\n")
#     print("l2 ", l2, "\n")
#     print("b ", b, "\n")
#     # l = [2] # initial Lagrange multipliers

#     P_sum_total = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3,1)
#     # P su
#     for i in range(1,length(l)+length(l2))
#         P_sum_total += P[i]
#     end 
    
#     # When GMRES is used as the T solver
#     T = GMRES_with_restart(l, l2, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

#     Psum_T_product = P_sum_total.*T

#     # ei_tr = transpose(ei)
#     ei_tr = conj.(transpose(ei)) 
#     k0 = 2*pi
#     Z = 1
#     # I put the code below here since it is used no matter the lenght of fSlist
#     # ei_T=ei_tr*T
#     ei_P_T=ei_tr*Psum_T_product
#     obj = imag(ei_P_T)[1]  # this is just the objective part of the dual 0.5*(k0/Z)*
#     print("obj ", obj, "\n")
#     D = obj 
#     # Reminder: a gradient is just a constraint evaluated a |t>.

#     GAdj_T = GAdjv_AA(gMemSlfN, cellsA, T)

#     g = ones(Float64, length(l), 1)
#     g2 = ones(Float64, length(l2), 1)
#     # Calculation and storing of the gradients 
#     if length(l)>0
#         print("Asym constraints only \n")
#         for i in eachindex(l)
#             g[i] = c1(l,l2,P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff,GAdj_T,i)
#         end
#         print("g ", g, "\n")
#     end 
#     if length(l2)>0 
#         print("Sym constraints only \n")
#         for j in eachindex(l2)
#             g2[j] = c2(l,l2,P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff,GAdj_T,j)
#         end 
#         print("g2 ", g2, "\n")
#     end 

#     # Code to add gradients to the dual value 
#     if length(l)>0 
#         for i in range(1,length(l), step=1) 
#             D += l[i]*g[i]
#         end 
#     end 
#     if length(l2)>0 
#         for j in range(1,length(l2), step=1)
#             D += l2[j]*g2[j]
#         end 
#     end 

#     # To get the just the dual value if we don't need the gradients, 
#     # we just need to calculate D = (1/4)*Re{<s_lambda|T>} 
#     # where |s_lambda> = lambda_i P_i |e_i>
    


#     # When conjugate gradient is used as the T solver 
#     # T = cg(l, l2, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

#     # When biconjugate gradient is used as the T solver 
#     # T = bicg(l, l2, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

#     # When stabilized biconjugate gradient is used as the T solver 
#     # T = bicgstab(l, l2, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)
    
#     print("D after adding grad ", D, "\n")
#     print(length(fSlist), "\n")
#     if length(fSlist)>0
#         print("In fSlist loop \n")
#         fSval = 0
#         for k in fSlist
#             prod_k = sym_and_asym_sum(l,l2,gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
#             # Asym_k = asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
#             # Sym_k = sym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
#             k_tr = conj.(transpose(k)) 
#             # kAsymk = l[1]*k_tr*Asym_k
#             # kSymk = l[2]*k_tr*Sym_k
#             k_prod_k = k_tr*prod_k
#             fSval += real(k_prod_k[1])
#             # fSval += real(kAsymk[1]+kSymk[1])

#             # A_k = asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
#             # k_tr = conj.(transpose(k)) 
#             # kAk=k_tr*A_k
#             # fSval += real(kAk[1])
#         end
#         D += fSval
#     end
#     gradient= vcat(g,g2) # Combine the sym and asym L mults into one list

#     print("dual", D,"\n")
#     # print("Done dual \n")
#     if get_grad == true
#         return real(D[1]), gradient, real(obj) 
#     elseif get_grad == false
#         return real(D[1]), real(obj) 
#     end
# end





# Start of new gradient for sym and asym cases code 
    # g = ones(Float64, length(l), 1)
    # g2 = ones(Float64, length(l2), 1)
    # g = c1(l,l2,P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff)
    # g2 = c2(l,l2,P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff)
    # grads = g+g2
    # End of new gradient for sym and asym cases code 
    # Never mind, we need to go back to the old code because we actually need 
    # a vector containing the values of the gradient and not just the values 

    # Start  of old gradient for sym and asym cases code 
    # print("C1(T)", C1(T)[1], "\n")
    # print("C2(T)", C2(T)[1], "\n")

    
    # End of old gradient for sym and asym cases code 

    # g[1] = c1(P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff)

    # print("ei ", ei, "\n")
  

    

    # This line would be for the new code where we sum the gradients since we sum the P's 
    # D += g + g2 # g: sum of asym constraints and g2: sum of sym constraints
