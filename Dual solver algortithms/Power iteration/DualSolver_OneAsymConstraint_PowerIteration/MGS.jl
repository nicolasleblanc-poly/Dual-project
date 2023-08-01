using LinearAlgebra, Random, Arpack


# # Test version of MGS function 
# function MGS(V, t) # Modified_gram_schmidt
#     # vk = 0
#     print("V ", V, "\n")
#     vj = V[:,1]
#     print("vj ", vj, "\n")
#     global qj = vj/norm(vj)
#     print("qj ", qj, "\n")
#     # print("(conj.(transpose(qj))*vk) ", (conj.(transpose(qj))*vj)[1], "\n")
#     # for k = 1:j-1
#     # for k = j+1:i
#     for k = 2:size(V)[2] # j+1:size(V)[2]
#         # print("(conj.(transpose(qj))*vk)*qj ", (conj.(transpose(qj))*vj)[1]*qj, "\n")
#         # vj -= ((conj.(transpose(qj))*vj)[1])*qj
#         print("qj ", qj, "\n")
#         print("Old vj ", vj, "\n")
#         print("((conj.(transpose(qj))*vj)[1])*qj ", dot(qj,vj)*qj , "\n")
#         vj = vj - dot(qj,vj)*qj # ((conj.(transpose(qj))*vj)[1])*qj
#         print("New vj ", vj, "\n")
#         qj = V[:,k]/norm(vj)
#         # vk = vk - (conj.(transpose(vk))*qj)*qj
#     end
#     print("qj ", qj, "\n")
#     print("vj ", vj, "\n")
#     print("dot product 1 ", dot(vj,qj),"\n")
#     t = t - dot(qj,t)*qj # ((conj.(transpose(qj))*t)[1])*qj
#     print("dot product 2 ", dot(t,qj),"\n")

#     # print("product ", conj.transpose(vk), t,"\n")
#     return t = t - dot(qj,t)*qj
#     # return t = t-(conj.(transpose(t))*vk)*vk
# end 


# Backup version of MGS function 
# function MGS(V, t) # Modified_gram_schmidt
#     # vk = 0
#     print("V ", V, "\n")
#     for j = 1:size(V)[2]
#         vj = V[:,j]
#         print("vj ", vj, "\n")
#         global qj = vj/norm(vj)
#         print("qj ", qj, "\n")
#         # print("(conj.(transpose(qj))*vk) ", (conj.(transpose(qj))*vj)[1], "\n")
#         # for k = 1:j-1
#         # for k = j+1:i
#         for k = j+1:size(V)[2]
#             # print("(conj.(transpose(qj))*vk)*qj ", (conj.(transpose(qj))*vj)[1]*qj, "\n")
#             # vj -= ((conj.(transpose(qj))*vj)[1])*qj
#             print("vj ", vj, "\n")
#             print("((conj.(transpose(qj))*vj)[1])*qj ", dot(qj,vj)*qj , "\n")
#             vj = vj - dot(qj,vj)*qj # ((conj.(transpose(qj))*vj)[1])*qj
#             # vk = vk - (conj.(transpose(vk))*qj)*qj
#         end
#         print("dot product 1 ", dot(vj,qj),"\n")
#     end
#     t = t - dot(qj,t)*qj # ((conj.(transpose(qj))*t)[1])*qj
#     print("dot product 2 ", dot(t,qj),"\n")

#     # print("product ", conj.transpose(vk), t,"\n")
#     return t = t - dot(qj,t)*qj
#     # return t = t-(conj.(transpose(t))*vk)*vk
# end 


function modified_gram_schmidt(A, t)
    # orthogonalises the columns of the input matrix
    matrix = Array{Float64}(undef, size(A)[1], size(A)[1]+size(t)[2])
    matrix[:,1:size(A)[2]] = A
    matrix[:,size(A)[2]+1:end] = t

    print("size(matrix) ", size(matrix),"\n")
    num_vectors = size(matrix)[2]
    orth_matrix = copy(matrix)
    for vec_idx = 1:num_vectors
        orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx]/norm(orth_matrix[:, vec_idx])
        for span_base_idx = (vec_idx+1):num_vectors
            # perform block step
            orth_matrix[:, span_base_idx] -= dot(orth_matrix[:, span_base_idx], orth_matrix[:, vec_idx])*orth_matrix[:, vec_idx]
        end
    end
    return orth_matrix[:,end:end]
    # return orth_matrix
end


A = Array{Float64}(undef, 3, 3)
# A = zeros(Int8, 3, 3)
A[1,1] = 1
A[2,1] = 2
A[3,1] = 3
A[1,2] = -1
A[2,2] = 0
A[3,2] = 3
A[1,3] = 0
A[2,3] = 0 
A[3,3] = 1
print("A ", A, "\n")

t = Array{Float64}(undef, 3, 1)
# A = zeros(Int8, 3, 3)
t[1,1] = 1
t[2,1] = 1
t[3,1] = 1

# print("MGS ", MGS(A,t) , "\n")

# MGS_output =  modified_gram_schmidt(A, t)
print("Julia MGS ", modified_gram_schmidt(A, t), "\n")
