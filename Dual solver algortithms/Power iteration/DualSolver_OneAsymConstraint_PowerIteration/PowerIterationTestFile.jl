using LinearAlgebra

function power_iteration_first_evaluation(A)
    # Ideally choose a random vector
    # to decrease the chance that our vector
    # is orthogonal to the eigenvector
    b_k = rand(Int8, 3, 1)
    # let's implement an error check using the ratios of different iterations 
    good = false 
    while good == false
        # calculate the n+1 term
        #calculate the operator-vector product (A linear operator and b_k vector)
        # G|v> type calculation
        A_bk = A*b_k
        print("A_bk ",A_bk, "\n")

        # A(greenCircAA, cellsA, chi_inv_coeff, l, P, b_k)
        # output(l,b_k,cellsA) # A*b_k
        # calculate the norm
        A_bk_norm = norm(A_bk)
        # re normalize the vector
        b_k1 = A_bk / A_bk_norm

        # calculate the n+2 term
        # G|v> type calculation
        A_bk1 = A*b_k1
        # A(greenCircAA, cellsA, chi_inv_coeff, l, P, b_k1)
        # output(l,b_k1,cellsA) # A*b_k using the new b_k (the n+1 one)
        # calculate the norm
        A_bk1_norm = norm(A_bk1)
        # re normalize the vector
        b_k = A_bk1 / A_bk1_norm # technically b_k2 but it's called b_k since it will be the
        # b_k for the next iteration of the loop

        # calculate the A*b_k product with the new b_k (the n+2 one)
        # G|v> type calculation
        A_bk2 = A*b_k
        # output(l,cellsA) 

        # calculate the ratios 
        # norm(A^{n+1}*x)/norm(A^{n}*x)
        ratio_n1_n = norm(A_bk1)/norm(A_bk)
        # norm(A^{n+2}*x)/norm(A^{n+1}*x)
        ratio_n2_n1 = norm(A_bk2)/norm(A_bk1)
        # do this for 3 times in a row
        if abs(ratio_n2_n1-ratio_n1_n)/ratio_n1_n < 0.01
            good = true
            # print("Good \n")
        end
        # print("ratio_n1_n ", ratio_n1_n, "\n")
        # print("ratio_n2_n1 ", ratio_n2_n1, "\n")
        # print("abs(ratio_n2_n1-ratio_n1_n)/ratio_n1_n ", abs(ratio_n2_n1-ratio_n1_n)/ratio_n1_n, "\n")
        # print("good ", good, "\n")
        global eigvector = b_k
        global A_eigvector = A_bk2
    end
    # print("Done \n")

    # it is better to implement an error check than a for loop using the number of iterations like below
    # input and output ratio 
    # norm(A^{n+1}*x)/norm(A^{n}*x) -> for a couple in a row -> ratio of the ratios
    # abs({n+2/n+1}-{n+1/n})/{n+1/n}
    # new ratio at each iteration 

    # the last b_k is the largest eigenvector of the linear operator A
    # calculate the eigenvalue corresponding to the largest eigenvector b_k 
    # using the formula: eigenvalue = (Ax * x)/(x*x), where x=b_k in our case
    # A*x=A*b_k is a G|v> type calculation
    #A_bk = output(l,b_k,cellsA)
    A_bk2_conj_tr = conj.(transpose(A_eigvector)) 
    bk_conj_tr = conj.(transpose(eigvector)) 
    eigenvalue = real((A_bk2_conj_tr*eigvector)/(bk_conj_tr*eigvector))[1]
    # print("b_k first ", eigvector, "\n")
    # print("eigenvalue first ", eigenvalue, "\n")

    return b_k, eigenvalue
end

function power_iteration_second_evaluation(A)
    # Ideally choose a random vector
    # to decrease the chance that our vector
    # is orthogonal to the eigenvector
    b_k = rand(Int8, 3, 1)
    # let's do the first run of the power iteration method to get an intial largest eigenvalue
    # and corresponding largest eigenvector
    evaluation = power_iteration_first_evaluation(A)
    eigvect_1 = evaluation[1] 
    eigval_1 = evaluation[2] 
    # # let's define some values
    # # chi coefficient
    # chi_coeff = 3.0 + 0.01im
    # # inverse chi coefficient
    # chi_inv_coeff = 1/chi_coeff 
    # chi_inv_dag_coeff = conj(chi_inv_coeff)
    # define the projection operators
    # I didn't include P in the calculation since P = I for this case
    # b_k = rand(ComplexF64, 3*cellsA[1]*cellsA[2]*cellsA[3], 1)
    good = false 
    while good == false
        # calculate the n+1 term
        # calculate the operator-vector product (A linear operator and b_k vector)
        # G|v> type calculation
        A_bk = A*b_k
        print("A_bk ",A_bk, "\n")
        #A(greenCircAA, cellsA, chi_inv_coeff, l, P, b_k)
        # output(l,b_k,cellsA) # A|v> calculation 
        # let's do the (eigval*I-A)v> = eigval*|v>-A*|v> calculation
        eigval_A_bk = eigval_1*b_k .- A_bk # this the A_v>=A*b_k calculation
        # calculate the norm
        eigval_A_bk_norm = norm(eigval_A_bk)
        # re normalize the vector
        b_k1 = eigval_A_bk / eigval_A_bk_norm # this is the n+1 term

        # calculate the n+2 term
        # G|v> type calculation
        A_bk1 = A*b_k1
        # A(greenCircAA, cellsA, chi_inv_coeff, l, P, b_k1)
        # output(l,b_k1,cellsA) # A*b_k using the new b_k (the n+1 one)
        # let's do the (eigval*I-A)v> = eigval*|v>-A*|v> calculation
        eigval_A_bk1 = eigval_1*b_k1 .- A_bk1 # this the A_v>=A*b_k calculation
        # calculate the norm
        eigval_A_bk1_norm = norm(eigval_A_bk1)
        # re normalize the vector
        b_k = eigval_A_bk1 / eigval_A_bk1_norm # this is the n+2 term
        # technically b_k2 but it's called b_k since it will be the
        # b_k for the next iteration of the loop

        # calculate the A*b_k product with the new b_k (the n+2 one)
        # G|v> type calculation
        A_bk2 = A*b_k
        # A(greenCircAA, cellsA, chi_inv_coeff, l, P, b_k) 
        # let's do the (eigval*I-A)v> = eigval*|v>-A*|v> calculation
        eigval_A_bk2 = eigval_1*b_k .- A_bk2 # this the A_v>=A*b_k calculation
        # calculate the ratios 
        # norm(A^{n+1}*x)/norm(A^{n}*x)
        ratio_n1_n = norm(eigval_A_bk1)/norm(eigval_A_bk)
        # norm(A^{n+2}*x)/norm(A^{n+1}*x)
        ratio_n2_n1 = norm(eigval_A_bk2)/norm(eigval_A_bk1)
        if abs(ratio_n2_n1-ratio_n1_n)/ratio_n1_n < 0.01
            good = true
            # print("Good \n")
        end
        global eigvector = b_k
        global A_eigvector = eigval_A_bk2
    end

    # the last b_k is the largest eigenvector of the linear operator A for the second run 
    # of the power iteration method
    # calculate the eigenvalue corresponding to the largest eigenvector b_k 
    # using the formula: eigenvalue = (Ax * x)/(x*x), where x=b_k in our case
    # A*x=A*b_k is a G|v> type calculation
    #A_bk = output(l,b_k,cellsA)
    A_bk2_conj_tr = conj.(transpose(A_eigvector)) 
    bk_conj_tr = conj.(transpose(eigvector)) 
    eigenvalue_2 = real((A_bk2_conj_tr*eigvector)/(bk_conj_tr*eigvector))[1]

    # print("b_k second ", eigvector, "\n")
    # print("eigenvalue second ", eigenvalue_2, "\n")


    # the minimum eigenvalue will be found by substracting the largest eigenvalue of the second run 
    # and that of the first run => lambda_2-lambda_1
    # will the minimum eigenvector be found by substracting the two eigenvectors?
    print("eigval_1 ", eigval_1, "\n")
    print("eigenvalue_2 ", eigenvalue_2, "\n" )
    # min_eigval = eigenvalue_2 - eigval_1
    # min_eigvec = eigvector - eigvect_1
    min_eigval = eigval_1 - eigenvalue_2
    min_eigvec = eigvect_1 - eigvector 
    print("min_eigval ", min_eigval, "\n")
    # print("min_eigvec ", min_eigvec, "\n")
    return min_eigvec, min_eigval
end

function validityfunc(A)
    eval_2 = power_iteration_second_evaluation(A)
    
    min_eigvec = eval_2[1] # Not needed here but needed later
    min_eigval = eval_2[2]
    # val=mineigfunc(x)
    # min_eval=val[1]
    # min_evec=val[2] # Not needed here but needed later
    # print("min_eval validityfunc ", min_eval,"\n")
    if min_eigval>0
        return 1
    else
        return -1
    end
end


# A = zeros(Int8, 2, 2)
# A[1,1] = 1
# A[1,2] = 2
# A[1,3] = 3
# A[2,1] = 1
# A[2,2] = -5 
# print(A)

A = zeros(Int8, 3, 3)
# A[1,1] = 2
# A[1,2] = -1
# A[1,3] = 0
# A[2,1] = -1
# A[2,2] = 2
# A[2,3] = -1 
# A[3,1] = 0
# A[3,2] = -1
# A[3,3] = 2
A[1,1] = 2
A[1,2] = -1
A[1,3] = 0
A[2,1] = -1
A[2,2] = 3
A[2,3] = -1 
A[3,1] = 0
A[3,2] = -1
A[3,3] = 4
# print(A)

# print(eigen(A).values)

print(validityfunc(A), "\n")

data = eigen(A)
print("Julia eigenvalues ", data.values, "\n")
# print("Julia eigenvectors ", data.vectors, "\n")

# Function output:
# eigval_1 4.722078795581014
# eigenvalue_2 3.4540949752946517
# min_eigval 1.2679838202863625
