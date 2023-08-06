"""
This module creates the b vector used to solve for |T>.

Author: Nicolas Leblanc
"""

module b_asym_only
export bv_asym_only
# creation of b 
function bv_asym_only(ei, l, P) 
    # As you can see, I didn't add the P projection matrix here.

    # If we have a negative in front of the second term 
    return -((1+l[1])/(2im)).*ei # l[1]*
    # You can either divide by l[1] in b right here or multiply by l[1] in A in gmres
end

end