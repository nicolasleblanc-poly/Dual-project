module opt_setup 
export Ps, L_mults
using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, bfgs_power_iteration_asym_only, dual_asym_only, gmres

# Start of new code (Saturday, January 21st 2023 at 19:37)
# Let's start by generating the P's. The most simple and logical situation 
# for splitting the cubic domain (so in 3D) is to cut the domain in x, in y
# and in z in half. Let's assume that the amount of cells is always even
# to avoid having decimal terms when we divide by 2. We then get 8 little/baby cubes. 
# The indexes will be [1:half] or [half:end]. For all of the baby cubes, the idea is 
# the same. We first need to create an array that we shall call M. M has the same dimensions of G 
# (the Green function), so (N_x,N_y,N_z,3) where N_i is the amount of cells in
# the i direction. To clarify, Nx=cellsA[1], Ny=cellsA[2], Nz=cellsA[3].
# Next, we need to select a baby cube using linear indexing and 
# add 1's everywhere in this part of the total array (the part associated with the 
# baby cube). We can then use diag(M) to get the diagonal of the diagonal of MaxGAssemblyOpts
# as a vector that we can then use in the rest of the program. Yeah! :)
# Not sure how to do the multiple lines of code below in a loop.
# The loop would be needed to simplify everything and to add 
# the depth parameter that would allow to add another division 
# by 2 in the x, y and z direction, which would be a depth of 2. This would allow to have 64
# constraints instead of 8. We could also do this for a depth of 3, where we could have 
# 512 constraints, which is kinda getting extreme but nonetheless pertinent. 
# First baby cube: [1:cellsA[1]/2, 1:cellsA[2]/2, 1:cellsA[3]/2]

# 0th P. This P is the complex identity matrix and it is considered because it assures 
# duality when using one asym constraint. 


function Ps(cellsA, nb_complex_P, nb_real_P)
    # P = []
    # The first P is always the asym only constraint and it is
    # the complex identity matrix. 
    M = ones(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
    M[:, :, :,:] .= 1.0im
    N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
    P0 = Diagonal(N)
    print("P0 ", P0, "\n")

    print("nb_complex_P ", nb_complex_P, "\n")
    print("nb_complex_P ", nb_complex_P, "\n")

    if nb_complex_P > 1
        # First baby cube [1:cellsA[1]/2, 1:cellsA[2]/2, cellsA[3]/2:end]
        M = zeros(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
        M[1:Int(cellsA[1]/2), 1:Int(cellsA[2]/2), 1:Int(cellsA[3]/2),:] .= 1.0im
        N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
        P1 = Diagonal(N)
        print("P1 ", P1, "\n")

        # Second baby cube: [1:cellsA[1]/2, 1:cellsA[2]/2, cellsA[3]/2+1:end]
        M = zeros(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
        M[1:Int(cellsA[1]/2), 1:Int(cellsA[2]/2), Int(cellsA[3]/2)+1:end,:] .= 1.0im
        N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
        P2 = Diagonal(N)
        print("P2 ", P2, "\n")

        # Third baby cube: [1:cellsA[1]/2, cellsA[2]/2+1:end, 1:cellsA[3]/2]
        M = zeros(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
        M[1:Int(cellsA[1]/2), Int(cellsA[2]/2)+1:end, 1:Int(cellsA[3]/2),:] .= 1.0im
        N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
        P3 = Diagonal(N)
        print("P3 ", P3, "\n")

        # Fourth baby cube: [1:cellsA[1]/2, cellsA[2]/2+1:end, cellsA[3]/2+1:end]
        M = zeros(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
        M[1:Int(cellsA[1]/2), Int(cellsA[2]/2)+1:end, Int(cellsA[3]/2)+1:end,:] .= 1.0im
        N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
        P4 = Diagonal(N)
        print("P4 ", P4, "\n")
    end

    if nb_real_P > 0
        # Fifth baby cube: [cellsA[1]/2+1:end, 1:cellsA[2]/2, 1:cellsA[3]/2]
        M = zeros(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
        M[Int(cellsA[1]/2)+1:end, 1:Int(cellsA[2]/2), 1:Int(cellsA[3]/2),:] .= 1.0
        N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
        P5 = Diagonal(N) 
        print("P5 ", P5, "\n")

        # Sixth baby cube: [cellsA[1]/2+1:end, cellsA[2]/2+1:end, 1:cellsA[3]/2]
        M = zeros(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
        M[Int(cellsA[1]/2)+1:end, Int(cellsA[2]/2)+1:end, 1:Int(cellsA[3]/2),:] .= 1.0
        N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
        P6 = Diagonal(N)
        print("P6 ", P6, "\n")

        # Seventh baby cube: [cellsA[1]/2+1:end, 1:cellsA[2]/2, cellsA[3]/2+1:end]
        M = zeros(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
        M[Int(cellsA[1]/2)+1:end, 1:Int(cellsA[2]/2), Int(cellsA[3]/2)+1:end,:] .= 1.0
        N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
        P7 = Diagonal(N)
        print("P7 ", P7, "\n") 

        # Eigth baby cube: [cellsA[1]/2+1:end, cellsA[2]/2+1:end, cellsA[3]/2+1:end]
        M = zeros(ComplexF64,cellsA[1],cellsA[2],cellsA[3],3)
        M[Int(cellsA[1]/2)+1:end, Int(cellsA[2]/2)+1:end, Int(cellsA[3]/2)+1:end,:] .= 1.0
        N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
        P8 = Diagonal(N)
        print("P8 ", P8, "\n")
    end 

    # For simplicity, I'm going to say that the first constraints are asymmetric and the ones 
    # after are symmetric 
    if nb_complex_P == 1
        P=[P0] 
    elseif nb_complex_P > 1 && nb_real_P == 0 # Asym constraints only 
        P = [P0,P1,P2,P3,P4]
    elseif nb_complex_P > 0 && nb_real_P > 0 # Asym and Sym constraints 
        P = [P0,P1,P2,P3,P4,P5,P6,P7,P8]
    end 
    # Is P always real? -> No 
    # print("size(P1) ",size(P1),"\n")
    # print("size(P2) ",size(P2),"\n")
    # print("size(P3) ",size(P3),"\n")
    # print("size(P4) ",size(P4),"\n")
    # print("size(P5) ",size(P5),"\n")
    # print("size(P6) ",size(P6),"\n")
    # print("size(P7) ",size(P7),"\n")
    # print("size(P8) ",size(P8),"\n")
    print("P ", P, "\n")
    multipliers = L_mults(nb_complex_P,nb_real_P)

    return P, multipliers 
end 

function L_mults(nb_complex_P,nb_real_P)
    # number_asym_constraints = 4
    # number_sym_constraints = 4
    l = Array{ComplexF64}(undef, nb_complex_P,1) # L mults related to asym constraints 
    l2 = Array{ComplexF64}(undef, nb_real_P, 1) # L mults related to sym constraints
    # Let's attribute random starting Lagrange multipliers that are between 0 and 3 (kinda 
    # arbitrary but we know the L mults are generally small)
    rand!(l,(0.01:3))
    rand!(l2,(0.01:3))
    return l, l2
end 
end 
