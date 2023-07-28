module davidson_TestFile_New
using LinearAlgebra, Random, Arpack
using KrylovKit
# function modified_gram_schmidt(V, t, i)
#     # vk = 0
#     print("V ", V, "\n")
#     for j = 1:i-1 # size(V)[2]
#         vj = V[1:end,j]
#         print("vj ", vj, "\n")
#         global qj = vj/norm(vj)
#         print("qj ", qj, "\n")
#         print("(conj.(transpose(qj))*vk) ", (conj.(transpose(qj))*vj)[1], "\n")
#         # for k = 1:j-1
#         for k = j+1:i
#         # for k = j+1:size(V)[2]
#             print("(conj.(transpose(qj))*vk)*qj ", (conj.(transpose(qj))*vj)[1]*qj, "\n")
#             vj = vj - ((conj.(transpose(qj))*vj)[1])*qj
#             # vk = vk - (conj.(transpose(vk))*qj)*qj
#         end
#         # print("dot product ", dot(vj,qj),"\n")
#     end
#     t = t-((conj.(transpose(qj))*t)[1])*qj
#     print("dot product ", dot(t,qj),"\n")

#     # print("product ", conj.transpose(vk), t,"\n")
#     return t = t-((conj.(transpose(qj))*t)[1])*qj
#     # return t = t-(conj.(transpose(t))*vk)*vk
# end 

function modified_gram_schmidt(A, t, index) # i
    # orthogonalises the columns of the input matrix
    print("size(A) ", size(A)[2], "\n")
    print("size(t) ", size(t), "\n")
    print("index ", index, "\n")
    print("A ", A, "\n")
    print("A val ", A[:,1], "\n")
    # print("size(t)[2] ", size(t)[2], "\n")
    print("t ", t, "\n")
    matrix = Array{Float64}(undef, size(A)[1],index+1) # 1rst old thing index+size(t)[2] , other old thing size(A)[1]
    print("matrix ", matrix, "\n")
    print("matrix[:,1:size(A)[1]] ", matrix[:,1:index], "\n")
    # i is the amount of cols (which are vectors) of A that we are considering here 
    matrix[:,1:index] = A #  matrix[:,1:size(A)[2]] = A
    matrix[:,index+1:end] = t

    print("size(matrix) ", size(matrix),"\n")
    print("matrix ", matrix, "\n")
    num_vectors = size(matrix)[2]
    orth_matrix = copy(matrix)
    for vec_idx = 1:num_vectors
        orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx]/norm(orth_matrix[:, vec_idx])
        for span_base_idx = (vec_idx+1):num_vectors
            # perform block step
            orth_matrix[:, span_base_idx] -= dot(orth_matrix[:, span_base_idx], orth_matrix[:, vec_idx])*orth_matrix[:, vec_idx]
        end
    end

    # Calculation of dot products to ensure the code works correctly


    print("dot1 ", dot(orth_matrix[:,1:1],orth_matrix[:,end:end]), "\n")
    print("dot2 ", dot(orth_matrix[:,2:2],orth_matrix[:,end:end]), "\n")
    # print("orth_matrix ", dot(orth_matrix[:,end:end],orth_matrix[:,end:end]), "\n")
    # print("orth_matrix ", orth_matrix, "\n")
    # print("orth_matrix ", orth_matrix, "\n")

    print("orth_matrix ", orth_matrix, "\n")
    print("orth_matrix[:,end:end] ", orth_matrix[:,end:end], "\n")
    return orth_matrix[:,end:end]
    # return orth_matrix
end

function davidson_it(A)
    # m = 20 # Amount of iterations of the inner loop

    tol = 1e-15 # Tolerance for which the program will converge 

    rows = size(A)[1]
    cols = size(A)[2]

    v = zeros(Float64, rows, 1)
    # v[1,1] = 1
    # v[2,1] = 2
    # v[3,1] = 3
    # v = Array{Float64}(undef, rows,1)
    rand!(v)
    print("v ", v, "\n")

    vk = v/norm(v) # Vector 
    wk = A*v # Vector 
    hk = (conj.(transpose(vk))*wk)[1] # Number
    print("hk ", hk, "\n")

    # Vk = Array{Float64}(undef, rows, cols)
    Vk = zeros(Float64, rows, cols)
    # The first index is to select a row
    # The second index is to select a column
    Vk[:,1] = vk
    print("Vk ", Vk, "\n")

    # Wk = Array{Float64}(undef, rows, cols)
    Wk = zeros(Float64, rows, cols)
    Wk[:,1] = wk
    print("Wk ", Wk, "\n")
    
    Hk = zeros(Float64, rows, cols)
    # Hk = Array{Float64}(undef, rows, cols)
    # print("hk ", hk[1], "\n")
    Hk[1,1] = hk
    print("Hk ", Hk, "\n")

    u = vk # Vector 
    theta = hk # Number 
    r = wk - real(theta)*u # Vector
    print("r ", r, "\n")

    # t = Array{Float64}(undef, rows, 1)
    t = zeros(Float64, rows, 1)

    # u_hat = Array{Float64}(undef, rows, 1)
    u_hat = zeros(Float64, rows, 1)

    # for index = 1:1
    for i = 2:cols # Old version 
    # for i = 1:cols # New version 
        # while det(((I-u*conj.(transpose(u)))*(A-theta[1]*I)*(I-u*conj.(transpose(u))))) == 0.0+0.0im
        #     rand!(v)
        # end
        diagonal_A = diag(A)
        A_diagonal_matrix = Diagonal(diagonal_A)
        print("A_diagonal_matrix ", A_diagonal_matrix, "\n")

        # print("I-u*conj.(transpose(u)) ", I-u*conj.(transpose(u)), "\n")
        # print("(A-theta[1]*I) ", (A-theta[1]*I), "\n")
        # print("I-u*conj.(transpose(u)) ", I-u*conj.(transpose(u)), "\n")

        u_mod = copy(u)
        print("u_mod[end:end,1] ", u_mod[end:end,1], "\n")
        u_mod[end] = 0.0
        print("u ", u, "\n")
        print("u_mod ", u_mod, "\n")

        # print("det diagional version of A ", det(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-theta[1]*I)*(I-u_mod*conj.(transpose(u_mod))))),"\n")
        # print("det of normal A ", det(((I-u_mod*conj.(transpose(u_mod)))*(A-theta[1]*I)*(I-u_mod*conj.(transpose(u_mod))))),"\n")
        # print("((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))) ", ((I-u_mod*conj.(transpose(u)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u*conj.(transpose(u)))), "\n")
        # print("size((I-u*conj.(transpose(u)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u*conj.(transpose(u)))", size((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))), "\n")
        
        print("I-u_mod*conj.(transpose(u_mod)) ", I-u_mod*conj.(transpose(u_mod)),"\n")
        print("A_diagonal_matrix-theta[1]*I ", A_diagonal_matrix-theta[1]*I,"\n")
        print("I-u_mod*conj.(transpose(u_mod)) ", I-u_mod*conj.(transpose(u_mod)),"\n")

        t = inv(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))))*(-r)
        
        print("size(r) ", size(r), "\n")
        print("size(t) ", size(t), "\n")

        # Test below: 
        # Instead of solving the whole thing, let's just solve the part of interest 
        # print("inv(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))))[1:2,1:2] ", inv(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))))[1:2,1:2], "\n")
        # print("-r[:,1:2] ", -r[1:2,:], "\n")
        # t_test = (inv(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))))[1:2,1:2])*(-r[1:2,:])
        # print("size(t_test) ", size(t_test), "\n")
        # print("t_test ", t_test, "\n")
        # t = zeros(Float64, rows, 1)
        # t[1:2,:] = t_test
    
        index = i-1
        print("Vk ", Vk, "\n")
        print("Vk[:,index] ", Vk[:,1:index], "\n")
        # New vk vector that will be added as a new column to Vk
        vk = modified_gram_schmidt(Vk[:,1:index], t, index) # modified_gram_schmidt(Vk, t, i)
        print("vk ", vk, "\n")
        # Expand V_k with this vector to V_{k+1}
        Vk[:,i] = vk
        print("new Vk ", Vk, "\n")
        # V_{k+1} = vk -> False! See above.

        # New wk that will be added as a new column to Wk
        wk = A*vk 
        print("wk ", wk, "\n")
        print("Wk ", Wk, "\n")
        # Expand W_k with this vector to W_{k+1}
        Wk[:,i] = wk 
        print("new Wk ", Wk, "\n")
        # [:,1:index]

        # print("conj.(transpose(A)) ", conj.(transpose(A)), "\n")
        # print("A ", A, "\n")
        # print("transpose(A)", transpose(A), "\n")

        print("Vk ", Vk, "\n")
        print("Vk[:,1:i] ", Vk[:,1:i], "\n")
        print("wk ", wk, "\n")
        # V*_{k+1} wk = V*_{k+1} A V*_{k+1} = V*_{k+1} A V*_{k+1}
        # print("(conj.(transpose(Vk))*wk)[1] ", (conj.(transpose(Vk))*wk)[1], "\n")

        # Hk[i,i] = (conj.(transpose(Vk))*wk)[1]
        print("OG Hk ", Hk, "\n")
        # hk is now a vector
        hk = (conj.(transpose(Vk[:,1:i]))*wk)
        Hk[1:i,i] = hk
        # Hk[1:i,i] = (conj.(transpose(Vk[:,1:i]))*wk)
        print("new Hk v1 ", Hk, "\n")

        # If we assume that A is hermitan, we don't have index+1to calculate vk*Wk
        # for the last row of Hk. We can just take the complex conjugate of 
        # Vk*wk, which we just caculated.
        Hk[i,1:i] = conj(transpose(hk))
        # Hk[i,1:i] = conj(transpose((conj.(transpose(Vk[:,1:i]))*wk)))
        print("new Hk v2 ", Hk, "\n")

        # We comment this part for debugging purposes
        # if A != conj((transpose(A)))
        #     print("(conj.(transpose(Vk))*wk)[1] ", (conj.(transpose(vk))*Wk)[1], "\n")
        #     Hk[end, 1:end] = (conj.(transpose(vk))*Wk) # [1]
        # end
        
        print("Julia eigvals ", eigsolve(Hk[1:i,1:i]), "\n")

        # Compute the largest eigenpar (theta,s) of H_{k+1} 
        # with the norm(s) = 1
        eigvals, eigenvectors = eigs(Hk[1:i,1:i], which=:LM)
        print("eigvals ", eigvals, "\n")
        print("eigenvectors ", eigenvectors, "\n")


        # We only want the eigenvalue that is positive
        position = 0

        # Problems with the code right now:
        # 1. I get a singular matrix when doing the direct solve. 
        # 2. Someting I get negative eigenvalues. 

        max_eigvals = 0.0 
        for i in eachindex(eigvals) # We need the eigenvalues to be positive, right (like before)?
            if eigvals[i] > 0 
                if eigvals[i] > max_eigvals
                    max_eigvals = eigvals[i]
                    theta = eigvals[i]
                    position = i 
                end 
                print("position ", position, "\n")
            end 
        end 
        eigenvector = eigenvectors[:,position]
        print("theta ", theta, "\n")
        print("eigenvector ", eigenvector, "\n")
        s = eigenvector/norm(eigenvector) # normalized_eigenvector
        print("s ", s, "\n")
        
        print("Vk ", Vk, "\n")
        u = Vk[:,1:i]*s # Compute the Ritz vector u  
        print("u ", u, "\n")
        u_hat = A*u # Should also have: A*u = W_k*s

        # u_hat test to see if A*u = W_k*s 
        u_hat_test = Wk[:,1:i]*s
        print("u_hat_test ", u_hat_test, "\n")

        print("u_hat ", u_hat, "\n")
        print("theta[1] ", real(theta[1]), "\n")
        print("theta[1]*u ", theta[1]*u, "\n")
        r = u_hat - theta[1]*u # Residual vector 
        print("r ", r, "\n")
        
    end
    print("norm of residual ", norm(r), "\n")
    # if norm(r) <= tol
    #     print("Exited the loop using break")
    #     break 
    # end 
    Vk = zeros(Float64, rows, cols)
    # Vk = Array{Float64}(undef, rows, cols)
    Vk[1:end,1] = vk

    Wk = zeros(Float64, rows, cols)
    # Wk = Array{Float64}(undef, rows, cols)
    Wk[1:end,1] = u_hat
    
    Hk = zeros(Float64, rows, cols)
    # Hk = Array{Float64}(undef, rows, cols)
    Hk[1,1] = real(theta[1])

    # for index = 1:1
    #     for i = 2:cols # Old version 
    #     # for i = 1:cols # New version 
    #         # while det(((I-u*conj.(transpose(u)))*(A-theta[1]*I)*(I-u*conj.(transpose(u))))) == 0.0+0.0im
    #         #     rand!(v)
    #         # end
    #         diagonal_A = diag(A)
    #         A_diagonal_matrix = Diagonal(diagonal_A)
    #         print("A_diagonal_matrix ", A_diagonal_matrix, "\n")

    #         # print("I-u*conj.(transpose(u)) ", I-u*conj.(transpose(u)), "\n")
    #         # print("(A-theta[1]*I) ", (A-theta[1]*I), "\n")
    #         # print("I-u*conj.(transpose(u)) ", I-u*conj.(transpose(u)), "\n")

    #         u_mod = copy(u)
    #         print("u_mod[end:end,1] ", u_mod[end:end,1], "\n")
    #         u_mod[end] = 0.0
    #         print("u ", u, "\n")
    #         print("u_mod ", u_mod, "\n")

    #         # print("det diagional version of A ", det(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-theta[1]*I)*(I-u_mod*conj.(transpose(u_mod))))),"\n")
    #         # print("det of normal A ", det(((I-u_mod*conj.(transpose(u_mod)))*(A-theta[1]*I)*(I-u_mod*conj.(transpose(u_mod))))),"\n")
    #         # print("((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))) ", ((I-u_mod*conj.(transpose(u)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u*conj.(transpose(u)))), "\n")
    #         # print("size((I-u*conj.(transpose(u)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u*conj.(transpose(u)))", size((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))), "\n")
            
    #         print("I-u_mod*conj.(transpose(u_mod)) ", I-u_mod*conj.(transpose(u_mod)),"\n")
    #         print("A_diagonal_matrix-theta[1]*I ", A_diagonal_matrix-theta[1]*I,"\n")
    #         print("I-u_mod*conj.(transpose(u_mod)) ", I-u_mod*conj.(transpose(u_mod)),"\n")

    #         t = inv(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))))*(-r)
            
    #         print("size(r) ", size(r), "\n")
    #         print("size(t) ", size(t), "\n")

    #         # Test below: 
    #         # Instead of solving the whole thing, let's just solve the part of interest 
    #         # print("inv(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))))[1:2,1:2] ", inv(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))))[1:2,1:2], "\n")
    #         # print("-r[:,1:2] ", -r[1:2,:], "\n")
    #         # t_test = (inv(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))))[1:2,1:2])*(-r[1:2,:])
    #         # print("size(t_test) ", size(t_test), "\n")
    #         # print("t_test ", t_test, "\n")
    #         # t = zeros(Float64, rows, 1)
    #         # t[1:2,:] = t_test
        
    #         index = i-1
    #         print("Vk ", Vk, "\n")
    #         print("Vk[:,index] ", Vk[:,1:index], "\n")
    #         # New vk vector that will be added as a new column to Vk
    #         vk = modified_gram_schmidt(Vk[:,1:index], t, index) # modified_gram_schmidt(Vk, t, i)
    #         print("vk ", vk, "\n")
    #         # Expand V_k with this vector to V_{k+1}
    #         Vk[:,i] = vk
    #         print("new Vk ", Vk, "\n")
    #         # V_{k+1} = vk -> False! See above.

    #         # New wk that will be added as a new column to Wk
    #         wk = A*vk 
    #         print("wk ", wk, "\n")
    #         print("Wk ", Wk, "\n")
    #         # Expand W_k with this vector to W_{k+1}
    #         Wk[:,i] = wk 
    #         print("new Wk ", Wk, "\n")
    #         # [:,1:index]

    #         # print("conj.(transpose(A)) ", conj.(transpose(A)), "\n")
    #         # print("A ", A, "\n")
    #         # print("transpose(A)", transpose(A), "\n")

    #         print("Vk ", Vk, "\n")
    #         print("Vk[:,1:i] ", Vk[:,1:i], "\n")
    #         print("wk ", wk, "\n")
    #         # V*_{k+1} wk = V*_{k+1} A V*_{k+1} = V*_{k+1} A V*_{k+1}
    #         # print("(conj.(transpose(Vk))*wk)[1] ", (conj.(transpose(Vk))*wk)[1], "\n")

    #         # Hk[i,i] = (conj.(transpose(Vk))*wk)[1]
    #         print("OG Hk ", Hk, "\n")
    #         # hk is now a vector
    #         hk = (conj.(transpose(Vk[:,1:i]))*wk)
    #         Hk[1:i,i] = hk
    #         # Hk[1:i,i] = (conj.(transpose(Vk[:,1:i]))*wk)
    #         print("new Hk v1 ", Hk, "\n")

    #         # If we assume that A is hermitan, we don't have index+1to calculate vk*Wk
    #         # for the last row of Hk. We can just take the complex conjugate of 
    #         # Vk*wk, which we just caculated.
    #         Hk[i,1:i] = conj(transpose(hk))
    #         # Hk[i,1:i] = conj(transpose((conj.(transpose(Vk[:,1:i]))*wk)))
    #         print("new Hk v2 ", Hk, "\n")

    #         # We comment this part for debugging purposes
    #         # if A != conj((transpose(A)))
    #         #     print("(conj.(transpose(Vk))*wk)[1] ", (conj.(transpose(vk))*Wk)[1], "\n")
    #         #     Hk[end, 1:end] = (conj.(transpose(vk))*Wk) # [1]
    #         # end
            
    #         print("Julia eigvals ", eigsolve(Hk[1:i,1:i]), "\n")

    #         # Compute the largest eigenpar (theta,s) of H_{k+1} 
    #         # with the norm(s) = 1
    #         eigvals, eigenvectors = eigs(Hk[1:i,1:i], which=:LM)
    #         print("eigvals ", eigvals, "\n")
    #         print("eigenvectors ", eigenvectors, "\n")


    #         # We only want the eigenvalue that is positive
    #         position = 0

    #         # Problems with the code right now:
    #         # 1. I get a singular matrix when doing the direct solve. 
    #         # 2. Someting I get negative eigenvalues. 

    #         max_eigvals = 0.0 
    #         for i in eachindex(eigvals) # We need the eigenvalues to be positive, right (like before)?
    #             if eigvals[i] > 0 
    #                 if eigvals[i] > max_eigvals
    #                     max_eigvals = eigvals[i]
    #                     theta = eigvals[i]
    #                     position = i 
    #                 end 
    #                 print("position ", position, "\n")
    #             end 
    #         end 
    #         eigenvector = eigenvectors[:,position]
    #         print("theta ", theta, "\n")
    #         print("eigenvector ", eigenvector, "\n")
    #         s = eigenvector/norm(eigenvector) # normalized_eigenvector
    #         print("s ", s, "\n")
            
    #         print("Vk ", Vk, "\n")
    #         u = Vk[:,1:i]*s # Compute the Ritz vector u  
    #         print("u ", u, "\n")
    #         u_hat = A*u # Should also have: A*u = W_k*s

    #         # u_hat test to see if A*u = W_k*s 
    #         u_hat_test = Wk[:,1:i]*s
    #         print("u_hat_test ", u_hat_test, "\n")

    #         print("u_hat ", u_hat, "\n")
    #         print("theta[1] ", real(theta[1]), "\n")
    #         print("theta[1]*u ", theta[1]*u, "\n")
    #         r = u_hat - theta[1]*u # Residual vector 
    #         print("r ", r, "\n")
            
    #     end
    #     print("norm of residual ", norm(r), "\n")
    #     if norm(r) <= tol
    #         print("Exited the loop using break")
    #         break 
    #     end 
    #     Vk = zeros(Float64, rows, cols)
    #     # Vk = Array{Float64}(undef, rows, cols)
    #     Vk[1:end,1] = vk
    
    #     Wk = zeros(Float64, rows, cols)
    #     # Wk = Array{Float64}(undef, rows, cols)
    #     Wk[1:end,1] = u_hat
        
    #     Hk = zeros(Float64, rows, cols)
    #     # Hk = Array{Float64}(undef, rows, cols)
    #     Hk[1,1] = real(theta[1])
    # end
    return real(theta)
end 

A = Array{Float64}(undef, 3, 3)
A[1,1] = 2
A[1,2] = -2
A[1,3] = 0
A[2,1] = -1
A[2,2] = 3
A[2,3] = -1 
A[3,1] = 0
A[3,2] = -1
A[3,3] = 4
# A = zeros(Int8, 3, 3)
# A[1,1] = 2
# A[1,2] = -1
# A[1,3] = 0
# A[2,1] = -1
# A[2,2] = 2
# A[2,3] = -1 
# A[3,1] = 0
# A[3,2] = -1
# A[3,3] = 2
print("A ", A, "\n")
# print(A[1,:])
# print("size(A) ", size(A)[2], "\n")

# print(eigen(A).values)

print(davidson_it(A), "\n")
end 