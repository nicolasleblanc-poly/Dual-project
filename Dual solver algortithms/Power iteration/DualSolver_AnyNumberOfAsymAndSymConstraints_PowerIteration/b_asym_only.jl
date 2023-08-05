"""
This module creates the b vector used to solve for |T>.

Author: Nicolas Leblanc
"""

module b_asym_only
export bv_asym_only

# creation of b 
function bv_asym_only(ei, l, l2, P)  

    # How the code is after adding the P's: 
    # One super important thing to understand the code below 
    # is that I assumed that the first set of P's were associated
    # with the asymmetric constraints and the rest of the P's 
    # were associated with the symmetric constraints. 
    val = (-1/(2im)).*ei 
    
    # Asym 
    if length(l) > 0 # l is associated to the asymmetric constraints 
        for i in eachindex(l)
            val -= (l[i]/(2im))*(P[1][i]*ei)
        end 
    end 
    # Sym 
    if length(l2) > 0 # l2 is associated to the symmetric constraints 
        for j in eachindex(l2)
            val += (l2[j]/2)*(P[1][j+Int(length(l))]*ei)
            print("val ", val, "\n")
        end 
    end 
    return val 
end
end
