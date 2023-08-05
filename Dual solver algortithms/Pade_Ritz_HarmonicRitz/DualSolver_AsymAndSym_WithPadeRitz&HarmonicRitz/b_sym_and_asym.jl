module b_sym_and_asym
export bv_sym_and_asym

"""
This module is to find a first approximate of the xi value such that all of the 
eigenvalues are positive.

Author: Nicolas Leblanc
"""

# creation of b 
function bv_sym_and_asym(ei, xi, l, P) 

    return -ei/(4im)+(l[1]/4)*(P[2]*ei)+(xi/4im)*(P[1]*ei)

    # Just like for A, the xi lambda is associated 
    # with the asym constraint and the second lambda is 
    # associated with the sym constraint.
end

end