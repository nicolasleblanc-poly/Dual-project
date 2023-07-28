using LinearAlgebra
# based on the example code from https://tobydriscoll.net/fnc-julia/krylov/gmres.html
# code for the AA case 

# check after the 20 iterations
# with 5 restarts 

# m is the maximum number of iterations
function GMRES_with_restart(A, b, m=20) # add cellsB for the BA case 
    n = length(b)
    Q = zeros(ComplexF64,n,m+1)
    Q[:,1] = reshape(b, (n,1))/norm(b)
    H = zeros(ComplexF64,m+1,m)
    # Initial solution is zero.
    x = 0
    residual = [norm(b);zeros(m)]
    for j in 1:m
        v = A*Q[:,j]
        for i in 1:j
            H[i,j] = dot(Q[:,i],v)
            v -= H[i,j]*Q[:,i] 
        end
        H[j+1,j] = norm(v)
        Q[:,j+1] = v/H[j+1,j]
        r = [norm(b); zeros(ComplexF64,j)]
        z = H[1:j+1,1:j] \ r # I took out the +1 in the two indices of H.
        x = Q[:,1:j]*z# I removed a +1 in the 2nd index of Q
        # second G|v> type calculation
        value = A*x
        # output(l,x,cellsA)
        residual[j+1] = norm(value - b )
    end
    return x 
end

# Test matrix and vector taken from: 
# https://www.l3harrisgeospatial.com/docs/imsl_sp_gmres.html

A = zeros(Int8, 6, 6)
A[1,1] = 10
A[1,2] = 0
A[1,3] = 0
A[1,4] = 0
A[1,5] = 0
A[1,6] = 0

A[2,1] = 0
A[2,2] = 10
A[2,3] = -3
A[2,4] = -1
A[2,5] = 0
A[2,6] = 0

A[3,1] = 0
A[3,2] = 0
A[3,3] = 15
A[3,4] = 0
A[3,5] = 0
A[3,6] = 0

A[4,1] = -2
A[4,2] = 0
A[4,3] = 0
A[4,4] = 10
A[4,5] = -1
A[4,6] = 0

A[5,1] = -1
A[5,2] = 0
A[5,3] = 0
A[5,4] = -5
A[5,5] = 1
A[5,6] = -3

A[6,1] = -1
A[6,2] = -2
A[6,3] = 0
A[6,4] = 0
A[6,5] = 0
A[6,6] = 6

# print(A, "\n")
x = zeros(Int8, 6, 1)
x[1,1] = 1
x[2,1] = 2
x[3,1] = 3
x[4,1] = 4
x[5,1] = 5
x[6,1] = 6

b = A*x

print(GMRES_with_restart(A,b))

# b = zeros(Int8, 6, 1)
# b[1,1] = 10
# b[2,1] = 7
# b[3,1] = 45
# b[4,1] = 33
# b[5,1] = -34
# b[6,1] = 31
# print(b, "\n")

# print("A*b",A*b, "\n")

# Y = zeros(Int8, 6, 1)

# mul!(Y, A, b)
# print("Y ",Y, "\n")