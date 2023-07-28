module dual_asym_only
export dual, c1
using b_asym_only, gmres, product, cg_asym_only, bicg_asym_only, bicgstab_asym_only
# Code to get the value of the objective and of the dual.
# The code also calculates the constraints
# and can return the gradient.
function c1(P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff) # asymmetric part 
    # Left term
    # print("In c1 \n")
    PT = T  # P*
    # ei_tr = transpose(ei) # we have <ei^*| instead of <ei|\
    # print("size(ei) ", size(ei), "\n")
    # print("size(T) ", size(T), "\n")
    ei_tr = conj.(transpose(ei))

    print("T ", T, "\n")

    print("T-T inner product ", conj.(transpose(T))*T, "\n")


    EPT=ei_tr*PT
    I_EPT = imag(EPT) 
    print("I_EPT ", I_EPT, "\n")
    # Right term => asym*T
    # G|v> type calculation

    # print("l ", l, "\n")

    asymT = asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, T)
    # print("asymT ", asymT, "\n")
    # output(l,T,fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)/l[1]
    # print("asym_T", asym_T, "\n")
    TasymT = conj.(transpose(T))*asymT
    print("T_asym_T ", TasymT, "\n")
    # print("I_EPT ", I_EPT,"\n")
    # print("T_asym_T ", T_asym_T,"\n")
    # print("I_EPT - T_chiGA_T",I_EPT - T_chiGA_T,"\n")
    # return real(I_EPT - T_asym_T[1]) # for the <ei^*| case
    return real(I_EPT - TasymT)[1] 
end
function dual(l,g,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff, cellsA,fSlist,get_grad)
    b = bv_asym_only(ei, l, P) 
    print("l ", l, "\n")
    print("b ", b, "\n")
    # l = [2] # initial Lagrange multipliers

    # When GMRES is used as the T solver 
    T = GMRES_with_restart(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

    # When conjugate gradient is used as the T solver 
    # T = cg(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

    # When biconjugate gradient is used as the T solver 
    # T = bicg(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

    # When stabilized biconjugate gradient is used as the T solver 
    # T = bicgstab(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

    g = ones(Float64, length(l), 1)
    # print("C1(T)", C1(T)[1], "\n")
    # print("C2(T)", C2(T)[1], "\n")
    g[1] = c1(P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff)
    print("g[1] ", g[1], "\n")
    # print("ei ", ei, "\n")
    # ei_tr = transpose(ei)
    ei_tr = conj.(transpose(ei)) 
    k0 = 2*pi
    Z = 1
    # I put the code below here since it is used no matter the lenght of fSlist
    ei_T=ei_tr*T
    obj = imag(ei_T)[1]  # this is just the objective part of the dual 0.5*(k0/Z)*
    print("obj ", obj, "\n")
    D = obj 
    for i in range(1,length(l), step=1)
        D += l[i]*g[i]
    end
    print("D after adding grad ", D, "\n")
    print(length(fSlist), "\n")
    if length(fSlist)>0
        print("In fSlist loop \n")
        fSval = 0
        for k in fSlist
            Asym_k = asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
            # Sym_k = sym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
            k_tr = conj.(transpose(k)) 
            kAsymk = l[1]*k_tr*Asym_k
            # kSymk = l[2]*k_tr*Sym_k
            fSval += real(kAsymk[1]) #+kSymk[1])
        end
        D += fSval
    end
    print("dual", D,"\n")
    # print("Done dual \n")
    if get_grad == true
        return real(D[1]), g, real(obj) 
    elseif get_grad == false
        return real(D[1]), real(obj) 
    end
end
end