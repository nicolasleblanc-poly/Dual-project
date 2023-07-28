using LinearAlgebra
# The idea is to start from a vector and build the Green function matrix.
# This file is to test the code to do this. Its quicker here than 
# using the full Green function.

cellsA = [2,1,1]
dim = 3*cellsA[1]*cellsA[2]*cellsA[3]
v = Array{ComplexF64}(undef,dim)
G = Array{ComplexF64}(undef,dim,dim)

v[1] = 1
v[2] = 2
v[3] = 3
v[4] = 4
v[5] = 5
v[6] = 6

# Loop version 
for i = 1:dim
    for j = 1:dim
        G[i,j] = v[j]*v[i]
    end 
end

G_hard_coded = Array{ComplexF64}(undef,dim,dim)
# Hard-coded version 
G_hard_coded[1,1] = v[1]*v[1]
G_hard_coded[1,2] = v[1]*v[2]
G_hard_coded[1,3] = v[1]*v[3]
G_hard_coded[1,4] = v[1]*v[4]
G_hard_coded[1,5] = v[1]*v[5]
G_hard_coded[1,6] = v[1]*v[6]
G_hard_coded[2,1] = v[2]*v[1]
G_hard_coded[2,2] = v[2]*v[2]
G_hard_coded[2,3] = v[2]*v[3]
G_hard_coded[2,4] = v[2]*v[4]
G_hard_coded[2,5] = v[2]*v[5]
G_hard_coded[2,6] = v[2]*v[6]
G_hard_coded[3,1] = v[3]*v[1]
G_hard_coded[3,2] = v[3]*v[2]
G_hard_coded[3,3] = v[3]*v[3]
G_hard_coded[3,4] = v[3]*v[4]
G_hard_coded[3,5] = v[3]*v[5]
G_hard_coded[3,6] = v[3]*v[6]
G_hard_coded[4,1] = v[4]*v[1]
G_hard_coded[4,2] = v[4]*v[2]
G_hard_coded[4,3] = v[4]*v[3]
G_hard_coded[4,4] = v[4]*v[4]
G_hard_coded[4,5] = v[4]*v[5]
G_hard_coded[4,6] = v[4]*v[6]
G_hard_coded[5,1] = v[5]*v[1]
G_hard_coded[5,2] = v[5]*v[2]
G_hard_coded[5,3] = v[5]*v[3]
G_hard_coded[5,4] = v[5]*v[4]
G_hard_coded[5,5] = v[5]*v[5]
G_hard_coded[5,6] = v[5]*v[6]
G_hard_coded[6,1] = v[6]*v[1]
G_hard_coded[6,2] = v[6]*v[2]
G_hard_coded[6,3] = v[6]*v[3]
G_hard_coded[6,4] = v[6]*v[4]
G_hard_coded[6,5] = v[6]*v[5]
G_hard_coded[6,6] = v[6]*v[6]

print("G ", G, "\n")
print("G_hard_coded ", G_hard_coded, "\n")
print("norm(G-G_hard_coded) ", norm(G-G_hard_coded), "\n")