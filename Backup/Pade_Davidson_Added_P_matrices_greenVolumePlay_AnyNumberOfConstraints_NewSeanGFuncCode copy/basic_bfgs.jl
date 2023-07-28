module basic_bfgs 
export bfgs
using LinearAlgebra, product

function bfgs(gMemSlfN,gMemSlfA,xeta,l,l2,dual,P,chi_inv_coeff,ei,
    cellsA)
    sigma = 1e-3 # Possible values for sigma range from 1e-1 to 1e-6

    # Solve for t using initial Lagrange multiplier values 
    dualval = 1
    grad = 1 
    obj = 
    t = 1 


    q_c = (2*real(conj.(transpose(t))*ei)-(conj.(transpose(t))*
    (xeta)*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, t))
    -sigma*conj.(transpose(t))*t)

    # Wtih q_c, we call Davidson and get an approximate solve for the smallest
    # positive eigenvalue. We then use this approximate result in the Padé 
    # approximate code.  



end

end