module gmres
export GMRES_with_restart
using product, LinearAlgebra, vector
# based on the example code from https://tobydriscoll.net/fnc-julia/krylov/gmres.html
# code for the AA case 
i = 1

# check after the 20 iterations
# with 5 restarts 

# m is the maximum number of iterations
function GMRES_with_restart(l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P, m=20) # add cellsB for the BA case 
    n = length(b)
    # n = cellsA[1]*cellsA[2]*cellsA[3]*3
    # print("n ", n, "\n")
    # print("size(b) ", size(b), "\n")
    Q = zeros(ComplexF64,n,m+1)
    # print("size of Q", size(Q))
    # print("b ", b, "\n")
    Q[:,1] = reshape(b, (n,1))/norm(b)
    H = zeros(ComplexF64,m+1,m)
    # Initial solution is zero.
    x = 0
    residual = [norm(b);zeros(m)]
    for j in 1:m
        # print(size(Q[:,j]))

        # first G|v> type calculation
        # Need to change the reshape to get the cells used for greenCircBA
        # cellsB = [1, 1, 1]

        v = (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, Q[:,j])
        # print("v ", v, "\n")

        # output(l,Q[:,j],cellsA)
        # print("Q[:,j] ", Q[:,j], "\n")

        # print(size(v),"\n")

        
        for i in 1:j
            H[i,j] = dot(Q[:,i],v)
            # print(size(H[i,j]), "\n")
            # print(size(Q[:,i]), "\n")

            v -= H[i,j]*Q[:,i]
            
        end
        
        H[j+1,j] = norm(v)
        Q[:,j+1] = v/H[j+1,j]
        # Solve the minimum residual problem.
        # r = zeros(j+1)
        # r[1] = norm(r)

        # r = norm(b)
        r = [norm(b); zeros(ComplexF64,j)]
        # print("H[1:j+1,1:j] ", H[1:j+1,1:j], "\n")
        # print("r ", r, "\n")
        z = H[1:j+1,1:j] \ r # I took out the +1 in the two indices of H.
        # print("z ", z, "\n")
        # print("shape of Q[:,1:j] ", size(Q[:,1:j]), "\n")
        
        # print("length of  ", size(Q[:,1:j]*z), "\n")

        # This is not code to be used. It is simply code that shows what we should have.
        # It is from https://en.wikipedia.org/wiki/Generalized_minimal_residual_method#Regular_GMRES_(MATLAB_/_GNU_Octave)
        # y = H(1:k, 1:k) \ beta(1:k);
        # x = x + Q(:, 1:k) * y;

        
        x = Q[:,1:j]*z# I removed a +1 in the 2nd index of Q
        # second G|v> type calculation
        value = (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, x)
        # output(l,x,cellsA)
        residual[j+1] = norm(value - b )
    end
    return reshape(x,(cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # x
end
end