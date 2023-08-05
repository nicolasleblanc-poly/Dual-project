"""
This module is a dual function calculator. It also calculates the 
asymmetric and symmetric constraint. 

Author: Nicolas Leblanc
"""
module dual_asym_only
export dual, c1
using b_sym_and_asym, gmres, product_sym_and_asym #, bicg_asym_only,bicgstab_asym_only,cg_asym_only
function c1(P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff) # Asymmetric part 
    # Left term
    PT = T  # P*
    ei_tr = conj.(transpose(ei))
    EPT=ei_tr*PT
    I_EPT = imag(EPT)     
    # Right term => asym*T
    # G|v> type calculation
    asymT = asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, T)
    TasymT = conj.(transpose(T))*asymT
    return real(I_EPT - TasymT)[1] 
end

function c2(P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff) # Symmetric part 
    # Left term
    PT = T  # P*
    ei_tr = conj.(transpose(ei))
    EPT=ei_tr*PT
    R_EPT = real(EPT) 
    # Right term => asym*T
    # G|v> type calculation
    symT = sym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, T)
    TsymT = conj.(transpose(T))*symT
    return real(R_EPT - TsymT)[1] 
end


function dual(xi,l,g,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff, cellsA,fSlist,get_grad)
    b = bv_sym_and_asym(ei, xi, l, P) 
    print("l ", l, "\n")
    print("b ", b, "\n")
        
    # When GMRES is used as the T solver
    T = GMRES_with_restart(xi,l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)
    
    # When conjugate gradient is used as the T solver 
    # T = cg(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

    # When biconjugate gradient is used as the T solver 
    # T = bicg(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)

    # When stabilized biconjugate gradient is used as the T solver 
    # T = bicgstab(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)
    

    g = ones(Float64, 1+length(l), 1)
    g[1] = c1(P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff) # Asym constraint
    g[2] = c2(P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff) # Sym constraint

    print("g[1] ", g[1], "\n")
    ei_tr = conj.(transpose(ei)) 
    k0 = 2*pi
    Z = 1
    # I put the code below here since it is used no matter the lenght of fSlist
    ei_T=ei_tr*T
    obj = imag(ei_T)[1]  # this is just the objective part of the dual 0.5*(k0/Z)*
    print("obj ", obj, "\n")
    D = obj 
    D += xi*g[1] # For the Asym constraint 
    for i in range(1,length(l), step=1) # For the Sym constraint 
        D += l[i]*g[i+1] # i+1 since i = 1 is for the Asym constraint 
    end
    print("D after adding grad ", D, "\n")
    print(length(fSlist), "\n")
    if length(fSlist)>0
        print("In fSlist loop \n")
        fSval = 0
        for k in fSlist
            Asym_k = asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
            Sym_k = sym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, k)
            k_tr = conj.(transpose(k)) 
            kAsymk = l[1]*k_tr*Asym_k
            kSymk = l[2]*k_tr*Sym_k
            fSval += real(kAsymk[1]+kSymk[1])
        end
        D += fSval
    end
    print("dual", D,"\n")
    if get_grad == true
        return real(D[1]), g, real(obj) 
    elseif get_grad == false
        return real(D[1]), real(obj) 
    end
end
end