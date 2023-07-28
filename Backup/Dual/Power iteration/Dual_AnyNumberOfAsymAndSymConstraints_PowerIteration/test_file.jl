# using Random 

# l = Array{ComplexF64}(undef, 3,1) # L mults related to asym constraints 
# l2 = Array{ComplexF64}(undef, 3, 1) # L mults related to sym constraints
# # Let's attribute random starting Lagrange multipliers that are between 0 and 3 (kinda 
# # arbitrary but we know the L mults are generally small)
# l[1]=2
# l[2]=3
# l[3]=4

# l2[1]=1
# l2[2]=1
# l2[3]=1
# print(l.*l2)

using LinearAlgebra
M = zeros(2,2,2,3)
M[1:Int(2/2), 1:Int(2/2), 1:Int(2/2),:] .= 1.0
P1 = diag(M)
print("P1 ", P1, "\n")