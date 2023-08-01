module b_asym_only
export bv_asym_only


"""
This module creates the b vector used to solve for |T>.

Author: Nicolas Leblanc
"""

# creation of b 
function bv_asym_only(ei, l, l2, P)  

    # # How the code was before adding the P's: 
    # # Start 
    # val = -1/(2im)
    # if length(l) > 0 # l is associated to the asymmetric constraints 
    #     for i in eachindex(l)
    #         val -= l[i]/(2im)
    #     end 
    # end 
    # if length(l2) > 0 # l2 is associated to the symmetric constraints 
    #     for j in eachindex(l2)
    #         val += l2[j]/2
    #     end 
    # end 
    # return val.*ei
    # # End 

    # How the code is after adding the P's: 
    # Start 
    # One super important thing to understand the code below 
    # is that I assumed that the first set of P's were associated
    # with the asymmetric constraints and the rest of the P's 
    # were associated with the symmetric constraints. 
    val = (-1/(2im)).*ei 
    
    # Asym 
    if length(l) > 0 # l is associated to the asymmetric constraints 
        for i in eachindex(l)
            # print("size(ei) ",size(ei),"\n")
            # print("size(val) ",size(val),"\n")
            # print("size(P[i]) ",size(P[1][i]),"\n")
            # print("(l[i]/(2im))*(P[i].*ei) ", (l[i]/(2im))*(P[1][i]*ei), "\n")
            print("(l[i]/(2im)) ", (l[i]/(2im)), "\n")
            print("(P[1][i]*ei) ", (P[1][i]*ei), "\n")
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
    return val # val.*ei
    # End 
end
end
