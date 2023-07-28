module PadeForRitz_GreenOperator_Code_sym_and_asym

using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
    Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
        MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
            product_sym_and_asym, gmres, Plots,Restart_Ritz_jacobiDavidson_operator_sym_and_asym,
                Restart_HarmonicRitz_EigvalClosestToZero_jacobiDavidson_operator_sym_and_asym,
                    gmres,approx_xi_solve_Ritz_HarmonicRitz_GreenOperator_sym_and_asym,
                        b_sym_and_asym, dual_asym_only

export surrogate_function,plot_surrogate_func,root_solve

# Function we want to find the roots of 
function surrogate_function(xi,l,P,ei,b,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff)
    print("Surrogate function calculation \n")
    print("xi ", xi, "\n")
    b = bv_sym_and_asym(ei, xi, l, P) 
    # When GMRES is used as the T solver
    T = GMRES_with_restart(xi,l, b, cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, P)
    return c1(P,ei,T,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff) # Asym constraint
end 

# function plot_surrogate_func(xi,l,P,ei,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff, max_val,ind)
#     xi_vals = LinRange(0,max_val,max_val) 
#     surr_func_vals = Vector{Float64}(undef,max_val)
#     for i in eachindex(xi_vals)
#         surr_func_vals[i] = surrogate_function(xi_vals[i],l,P,ei,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff)
#     end
#     print("size(xi_vals) ", size(xi_vals), "\n")
#     print("size(surr_func_vals) ", size(surr_func_vals), "\n")

#     if ind == 1
#         plot(xi_vals, surr_func_vals, title = "Zoomed-in graph: Surrogate function vs xi",)  
#         xlabel!("Xi")
#         ylabel!("Surrogate function")
#         savefig("/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/greenVolumePlay_AsymAndSym_NewSeanGFuncCode_WithPadeRitz&HarmonicRitz/Graphs/zoomed_operator1_graph_surr_func_vs_xi.png")
    
#     elseif ind == 2
#         plot(xi_vals, surr_func_vals, title = "Zoomed-in graph: Surrogate function vs xi",)  
#         xlabel!("Xi")
#         ylabel!("Surrogate function")
#         savefig("/home/nic-molesky_lab/Pad---Ritz-and-Harmonic-Ritz-duality-checker/greenVolumePlay_AsymAndSym_NewSeanGFuncCode_WithPadeRitz&HarmonicRitz/Graphs/zoomed_operator2_graph_surr_func_vs_xi.png")
    
#     end
# end 

################################################ Peak finders
function peak_finder(xs, ys)  # Finds the peaks within the sampled point 
    slope_criteria = 2 # This needs to be more dynamic, to account for the amount of sampled points
    peaks = []
    slopes = [] 
    for i in range(2, length(xs))
        added = false #verify if a peak has been added twice
        slope =  (ys[i]-ys[i-1])/(xs[i]-xs[i-1])
        push!(slopes, slope)

        #Test 1: Checks a sign change in the slope
        if i > 2
            if (slopes[i-2]\slopes[i-1]) < 0 && ys[i-2] < -5 #change of sign of the slope, implying a peak #the ys[i] <-5 is to prevent finding regions between the peaks
                push!(peaks, xs[i-1]) #note: we take xs[i-1] instead of xs[i-2] because the i-1 value is to the right of the peak
                added = true
            end
        end
        #Test 2: Checks if the slope before and after a point stopped growing (indicating that there was a peak)
        if i > 3
            if abs(slopes[i-3]) < abs(slopes[i-2]) && abs(slopes[i-2]) > abs(slopes[i-1]) && added == false && ys[i] <-5 #the ys[i] <-5 is to prevent finding regions between the peaks
                push!(peaks, xs[i-1])
            end
        end
    end
    print("peaks ", peaks, "\n")
    return peaks
end

################################################### Padé function
# x : x values for the sampled points
# y : y values for the sampled points
# N : number of points used to evaluate the approximate once it has been built using x,y 
# xl : left bound from which the Padé is evaluated 
# xr : right bound up to which the Padé is evaluated
# rebuild_with : evaluates the constructed Padé at the x values in this list

# function Pade(x, y; N = 500, xl = 0.0, xr = xmax, rebuild_with = [])
function Pade(x, y, x_start; N = 500, xl = 0.0, rebuild_with = [])
    #Padé approximant algorithm
    x = x
    r = y
    l = length(x)
    R = zeros(l)
    X = zeros(l)
    P = zeros(l)
    S = zeros(l)
    M = zeros(l)

    for i in range(1,l) # First loop 
        R[i] = r[i]
        X[i] = x[i]
    end

    for j in range(1,l) # Second loop
        P[j] = R[j]
        for s in range(j+1, l)
            S[s] = R[j]
            R[s] = R[s] - S[s]
            if R[s] == 0
                println("Houston, we have a problem, ABORT.")
            else
                M[s] = X[s] - X[j]
                R[s] = M[s]/R[s]
                if j-1 == 0 # To avoid indexing at the P[0] position
                    R[s] = R[s]
                else
                    R[s] = R[s] + P[j-1]
                end
            end
        end
    end

    
    global px
    global approx 
    if isempty(rebuild_with)== true
        print("Sampling \n")
        px = [i for i in range(xl, x_start, N)]
        approx = zeros(length(px))
        for i in eachindex(px)
            approx[i] = rebuild(px[i], l, P, X)
        end 
    else
        px = rebuild_with
        approx = zeros(length(px))
        for i in eachindex(px)
            approx[i] = rebuild(px[i], l, P, X)
        end 
    end
    return (px, approx)
end

function rebuild(x,l,P,X)  # Rebuild the approximation from the little blocks ##
    A = zeros(l)
    B = zeros(l)
    A[1] = P[1]
    B[1] = 1

    A[2] = P[2]*P[1] + (x-X[1])
    B[2] = P[2]
    for i in range(2, l-1)
        A[i+1] = A[i]*(P[i+1] -P[i-1]) + A[i-1]*(x-X[i])
        B[i+1] = B[i]*(P[i+1] -P[i-1]) + B[i-1]*(x-X[i])
    end

    if isinf(abs(A[l])) == true || isinf(abs(B[l])) == true || isnan(abs(A[l])) == true || isnan(abs(B[l])) ==true
        print("Problem \n")
        # throw(Error) # Not sure what to do when this happens yet, 
        # problems occur when N exceeds 336
    else
        return A[l]/B[l]
    end
end


################################################### First test that needs to be done
#this test samples the function initially to determine where is the
#region in which the function is definitly above zero

function first_sampling(x_start,l,ei,b,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P)
    print("x_start ", x_start, "\n")
    xs = Any[x_start]
    ys = Any[surrogate_function(x_start,l,P,ei,b,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff)]
    xr = x_start
    local xl = 0
    xn = x_start/2
    # Check if our starting point gives us a value above 0
    if ys[end] <= 0
        while ys[end] <=0
            x_start = 2*x_start
            push!(xs, x_start)
            push!(ys,surrogate_function(x_start,l,P,ei,b,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff))
        end
    else # If our starting point was adequate, samples from right to left
        while ys[1] > 0
            insert!(xs,1,xn)
            insert!(ys,1,surrogate_function(xn,l,P,ei,b,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff))
            xn = xn/2
        end
    end
    # One last check to see if it's still above zero later on
    x_start = 1.5*x_start
    push!(xs, x_start)
    push!(ys, surrogate_function(x_start,l,P,ei,b,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff))
    
    print("xs", xs, "\n")
    print("ys", ys, "\n")
    
    if ys[end] < 0
        println("Need to check this scenario")
    end
    # Check how many points we already have sampled to determine 
    # how much more sampling needs to be done 
    # (we're going to do one sampling in between every already sampled points)
    xns = Any[]
    for i in range(1, length(xs)-1)
        push!(xns, (xs[i+1]+xs[i])/2)
    end
    for i in range(1, length(xns))
        for j in range(2, length(xs))
            if xns[i] < xs[j] && xns[i] > xs[j-1]
                insert!(xs, j, xns[i])
                insert!(ys,j,surrogate_function(xns[i],l,P,ei,b,cellsA, gMemSlfN,
                    gMemSlfA, chi_inv_coeff))    
            end
        end
    end
    # Use the padé to see if peaks are predicted in the area that is supposed to be convexity
    x_pade, y_pade = Pade(xs,ys,x_start)
    print("x_pade 1", x_pade, "\n")
    print("y_pade 1 ", y_pade, "\n")
    peaks = peak_finder(x_pade, y_pade)

    # Determine which region is above the x axis
    above_region = xs
    for i in range(1,length(xs))
        if ys[i] < 0
            above_region = xs[i:end]
        end
    end
    # Check if there is a predicted peak wihtin the "above" sampled region
    if isempty(peaks) == false
        for i in range(1, length(peaks))
            if peaks[i] > above_region[1] && peaks[i] < above_region[end]
                println("There is a predicted peak in the above region, do something about it")
            end
        end
    end
    # Determines the right and left bound in which more sampling might be needed
    print("above_region ", above_region, "\n")
    xr = above_region[2]
    for i in range(1,length(xs))
        if ys[i] < 0
            xl = xs[i]
        end
    end
    return xs, ys, xl, xr
end

################################################### Padé stop criteria test
# This function checks wheter that last sampling changed the Padé approximant
# significantly or not. If not, stop sampling.
# It keeps in memory previous sampling to compare them
function pade_stop_criteria(xl,x_start,big_x_sampled,big_y_sampled,errors)
    local pade_x1, pade_y1, pade_x2, pade_y2, error
    stop_crit = false
    if length(big_x_sampled) <= 1
        #do nothing
        return errors,stop_crit
    elseif length(big_x_sampled) > 1
        # Compare samplings
        N = 1000
        pade_x1, pade_y1 = Pade(big_x_sampled[end],big_y_sampled[end],x_start,N=N,xl=xl)
        pade_x2, pade_y2 = Pade(big_x_sampled[end-1],big_y_sampled[end-1],x_start,N=N,xl=xl)
        # Calculate error between the two Padé's in between a relevant x range
        error = 0
        for i in range(1,length(pade_x1))
            error += abs(pade_y2[i] - pade_y1[i])/N
        end
        push!(errors, error)
        # Critera to see by how much the error shrunk from one iteration to the next
        if length(errors)>1
            ratio = errors[end-1]/errors[end]
            print("ratio ", ratio, "\n")
            push!(ratios, ratio)
            # Checks if the error is diminishing with each extra sampling (needs to diminish twice in a row)
            if length(ratios)>2
                if ratios[end]<ratios[end-1] && ratios[end-1] < ratios[end-2]
                    stop_crit = true
                    println("Done \n")
                    return errors,stop_crit
                else
                    return errors,stop_crit # stop_crit is false in this case
                end
            end
        end
    end
end


############################ Bissection root finding ##########################
# x1: starting point to the left
# x2 : starting point to the right
# ϵ: Maximum error we want between the actual zero and the approximated one
# N: Maximum number of iteration before returing a value
function bissection(x1,x2,x_start,ϵ,N,big_x_sampled,big_y_sampled)
    fxm = 0.0
    xm = (x1 + x2)/2
    compt = 0
    while abs(x2-x1)/(2*abs(xm)) > ϵ && compt < N
        xm = (x1 + x2)/2
        ans = Pade(big_x_sampled[end],big_y_sampled[end],x_start,rebuild_with=[x1,x2,xm])
        ys = ans[2]
        fx1 = ys[1]
        fx2 = ys[2]
        fxm = ys[3]
        if  fx1*fxm < 0
            x2 = xm
        elseif fxm*fx2<0
            x1 = xm
        end
        compt +=1
    end
    return (xm, fxm)
end

function root_solve(xi,l,P,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,ei,b,
        innerLoopDim,restartDim,tol_MGS,tol_conv,tol_eigval,tol_bissection,
            tol_bicgstab)

    # Need to update the inputs of xi_update below
    x_start = xi_update(xi,l,gMemSlfN,gMemSlfA,
        cellsA,chi_inv_coeff,P,innerLoopDim,restartDim,tol_MGS,
            tol_conv,tol_eigval,tol_bicgstab)
    print("xi such that smallest eigval of A is positive ", x_start, "\n")
    surr_func_x_start = surrogate_function(x_start,l,P,ei,b,cellsA, gMemSlfN,
        gMemSlfA, chi_inv_coeff)[1]
    print("surr_func_x_start: ", surr_func_x_start, "\n")
    
    # plot_surrogate_func(xi,l,P,ei,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff,
    #     Int(round(x_start))+100,1)

    print("Started first sampling \n")
    xs, ys , xl, xr = first_sampling(x_start,l,ei,b,cellsA,gMemSlfN,gMemSlfA,
        chi_inv_coeff,P)
    print(("End of first sampling "))

    x_sampled = xs
    y_sampled = ys
    
    xs = 0
    ys = 0
    xr = 0
    xl = 0
    big_x_sampled = Any[]
    big_y_sampled = Any[]
    index = 1
    print("Start of Padé \n")
    pade_x, pade_y = Pade(x_sampled, y_sampled, x_start,N =5000)
    print("End of Padé \n")

    print("Start of peak_finder \n")
    peaks = peak_finder(pade_x, pade_y)
    print("End of peak_finder \n")
       
    xn = last(peaks)
    xs = [xn]
    ys = [surrogate_function(xn,l,P,ei,b,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff)]

    # Adds the new sampled values in order to the already sampled ones (in ascending x order)
    for i in range(1, length(xs))
        for j in range(2, length(x_sampled))
            if xs[i] < x_sampled[j] && xs[i] > x_sampled[j-1]
                insert!(x_sampled, j, xs[i])
                insert!(y_sampled,j,ys[i])
                index = j
            end
            if xs[i] < x_sampled[j] && j ==2
                insert!(x_sampled, 1, xs[i])
                insert!(y_sampled,1,ys[i])
                index = j
            end
            if xs[i] > x_sampled[j] && j==length(x_sampled)
                insert!(x_sampled, j+1, xs[i])
                insert!(y_sampled,j+1,ys[i])
                index = j
            end
        end
    end
    # Checks if this sampling has changed something about our evaluation
    push!(big_x_sampled,x_sampled[index:end])
    push!(big_y_sampled,y_sampled[index:end])

    # We can narrow the position of the zero using our sampled points
    got_xl = false
    for i in range(1, length(x_sampled)-1)
        sleep(1)
        if y_sampled[i]<0
            xl = x_sampled[i]
            got_xl = true
        end
        if y_sampled[i+1] > 0 && got_xl == true
            xr = x_sampled[i+1]
        end
        got_xl = false
    end
    # Finds the zero using the last Padé approximant (bissection method)
    print("Start of bissection \n")
    zeros = bissection(xl,xr,x_start,10^(-10),100,big_x_sampled,big_y_sampled)
    print("End of bissection \n")
    print("bissection zeros ", zeros, "\n")

    surr_func_xi_zeros = surrogate_function(zeros[1],l,P,ei,b,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff)
    print("surr_func_xi_zeros : ", surr_func_xi_zeros, "\n")

    for it = 1:1000
        print("it ", it, "\n")
        if abs(surr_func_xi_zeros) > tol_bissection
            if surr_func_xi_zeros < 0
                xl = zeros[1]  
            else # Essentially if surr_func_xi_zeros > 0
                xr = zeros[1]
            end

            push!(big_x_sampled[1],zeros[1])
            push!(big_y_sampled[1],surr_func_xi_zeros)
            zeros = bissection(xl,xr,x_start,10^(-10),100,big_x_sampled,big_y_sampled)
            surr_func_xi_zeros = surrogate_function(zeros[1],l,P,ei,b,cellsA, gMemSlfN,
                gMemSlfA, chi_inv_coeff)
           
        elseif abs(surr_func_xi_zeros) < tol_bissection
            print("The guessed root is: ", zeros[1])
            # plot_surrogate_func(xi,l,P,ei,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff,Int(round(zeros[1]))+50,2)
            return zeros[1],surr_func_xi_zeros
        end
        
    end
    print("Went through all of the iterations \n")
    print("The guessed root is: ", zeros[1])
    # plot_surrogate_func(xi,l,P,ei,cellsA, gMemSlfN,gMemSlfA, chi_inv_coeff,Int(round(zeros[1]))+50,2)
    return zeros[1],surr_func_xi_zeros
end

end
