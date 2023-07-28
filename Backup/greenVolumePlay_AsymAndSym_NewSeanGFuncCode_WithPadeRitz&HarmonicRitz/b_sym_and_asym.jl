module b_sym_and_asym
export bv_sym_and_asym
# creation of b 
function bv_sym_and_asym(ei, xi, l, P) 
    # print("-ei/(2im) ", -ei/(2im), "\n")
    # print("(l[1]/(2im))*P*ei  ", (l[1]/(2im))*P*ei , "\n")

    # If we have a negative in front of the second term 
    # return (-((1+xi)/(2im))+l/2).*ei # l[1]*

    return -ei/(4im)+(l[1]/4)*(P[2]*ei)+(xi/4im)*(P[1]*ei)

    # You can either divide by l[1] in b right here or multiply by l[1] in A in gmres


    # return .-ei/(2im) - (l[1]/(2im)).*ei # *P

    # At the end of the intership, we weren't sure if there was a negative
    # in front of the second term or not. 

    # If we don't have a negative in front of the second term 
    # return .-ei/(2im) + (l[1]/(2im)).*ei # *P

    # just like for A, the first lambda is associated 
    # with the asym part and the second lambda is 
    # associated with the sym part 
end

end