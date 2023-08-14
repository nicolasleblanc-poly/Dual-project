"""
This module contains the code for a root solver that finds 
the last crossing of a function called the surrogate. The 
code uses a Padé approximate and a bisection algorithm.

Author: Nicolas Leblanc
"""

module PadeForRitz_Code
using Plots, Distributed, Random, LinearAlgebra, Peaks
export surrogate_function,plot_surrogate_func,root_solve


# Function we want to find the roots of 
function surrogate_function(alpha_0,alpha_1,D,P1,xi,s)
    P0 = adjoint(D)*D
    # Matrix
    A = (alpha_1+xi)*P0+alpha_0*P1
    # Vector direct solve
    x = A\s
    # Surrogate function 
    surr_func = real(real(adjoint(x)*s)-adjoint(x)*P0*x)
    return surr_func
end

function plot_surrogate_func(alpha_0,alpha_1,D,P1,max_val,s,ind)
    P0 = adjoint(D)*D
    xi_vals = LinRange(0,max_val,4*max_val) 
    surr_func_vals = Vector{Float64}(undef,4*max_val)
    for i in eachindex(xi_vals)
        # Matrix
        A = (alpha_1+xi_vals[i])*I + alpha_0*inv(adjoint(D))*P1*inv(D)
        # Vector direct solve
        x = A\s
        surr_func_vals[i] = real(imag(adjoint(x)*s)-adjoint(x)*P0*x)
    end

    # Good graph
    plot(xi_vals, surr_func_vals, title = "Zoomed-in graph: Surrogate function vs xi") # ,xlim=(0,max_val+60),ylim=(-100,2)  
    xlabel!("Xi")
    ylabel!("Surrogate function")

    if ind == 1
        savefig("Pade_Ritz_HarmonicRitz_algorithms/Combining Padé, Ritz and harmonic Ritz/Matrix version/Graphs/Pade surrogate graphs/zoomed1_in_graph_surr_func_vs_xi.png")
    elseif ind == 2
        savefig("Pade_Ritz_HarmonicRitz_algorithms/Combining Padé, Ritz and harmonic Ritz/Matrix version/Graphs/Pade surrogate graphs/zoomed2_in_graph_surr_func_vs_xi.png")
    end

    pks, vals = findmaxima(-surr_func_vals)
    print("pks ", pks, "\n")
    print("vals ", vals, "\n")
    plotpeaks(xi_vals, surr_func_vals, peaks=pks,  widths=true)  # prominences=true, # see above plot for result
    savefig("Pade_Ritz_HarmonicRitz_algorithms/Combining Padé, Ritz and harmonic Ritz/Matrix version/Graphs/Pade surrogate graphs/minimas_graph_surr_func_vs_xi.png")

    xi_val_peaks = []
    for index in pks 
        push!(xi_val_peaks,xi_vals[index])
    end
    print("xi_val_peaks ", xi_val_peaks, "\n")
end 

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
                # println("Houston, we have a problem, ABORT.")
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
        # approx = map(rebuild, px, l, P)
        for i in eachindex(px)
            approx[i] = rebuild(px[i], l, P, X)
        end 
    else
        px = rebuild_with
        approx = zeros(length(px))
        # approx = map(rebuild, px, l, P)
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


function first_sampling(alpha_0,alpha_1,D,P1,x_start,s)

    # Seperation between values instead of doubling
    # s = 0.1
    # s_n = 2*s_{n+1}

    # Weak version of Padé 
    # Converges if all the eigenvalues are initially (so right after
    # Ritz and harmonic Ritz) positive
    # Strong version of Padé
    # Doesn't converge if all of the eigenvalues are initially (so right after 
    # Ritz and harmonic Ritz)positive since the xi value found might be overshot. 
    # It will therefore go backwards and try to find 
    # the smallest xi value.

    # If all of the eigenvalues are negative, we'll want to sample to the right.


    print("x_start ", x_start, "\n")
    xs = Any[x_start]
    ys = Any[surrogate_function(alpha_0,alpha_1,D,P1,x_start,s)]
    # ys = Any[f(x_start)]
    xr = x_start
    local xl = 0
    xn = x_start/2
    # Check if our starting point gives us a value above 0
    if ys[end] <= 0
        while ys[end] <=0
            x_start = 2*x_start
            push!(xs, x_start)
            push!(ys,surrogate_function(alpha_0,alpha_1,D,P1,x_start,s))
        end
    else # If our starting point was adequate, samples from right to left
        while ys[1] > 0
            insert!(xs,1,xn)
            insert!(ys,1,surrogate_function(alpha_0,alpha_1,D,P1,xn,s))
            xn = xn/2
        end
    end
    # One last check to see if it's still above zero later on
    x_start = 1.5*x_start
    push!(xs, x_start)
    push!(ys, surrogate_function(alpha_0,alpha_1,D,P1,x_start,s))
    
    print("xs", xs, "\n")
    print("ys", ys, "\n")
    
    if ys[end] < 0
        print("There a problem since are last point doesn't give a positive 
            value. Need to check this scenario")
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
                insert!(ys,j,surrogate_function(alpha_0,alpha_1,D,P1,xns[i],s))     
            end
        end
    end
    # Use the padé to see if peaks are predicted in the area that is supposed to be convexity
    x_pade, y_pade = Pade(xs,ys,x_start)

    pks, vals = findmaxima(-y_pade)
    print("pks first_sampling ", pks, "\n")
    peaks = []
    for index in pks 
        push!(peaks,x_pade[index])
    end
    print("xi_val_peaks first_sampling ", peaks, "\n")

    if length(peaks) == 0
        x_negative = -x_value
        pushfirst!(xs, x_negative)
        pushfirst!(ys, surrogate_function(alpha_0,alpha_1,D,P1,x_negative,s))
        print("xs", xs, "\n")
        print("ys", ys, "\n")
        # Use the padé to see if peaks are predicted in the area that is supposed to be convexity
        x_pade, y_pade = Pade(xs,ys,x_negative)
        pks, vals = findmaxima(-y_pade)
        pushfirst!(peaks,x_negative)
        # for index in pks 
        #     push!(peaks,x_negative)
        # end
        print("xi_val_peaks first_sampling 2 ", peaks, "\n")
    end

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
    xr = above_region[2]
    for i in range(1,length(xs))
        if ys[i] < 0
            xl = xs[i]
        end
    end
    return xs, ys, xl, xr
end

############################### Bissection root finding ############################### 
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
   
function root_solve(alpha_0,alpha_1,D,P1,x_start,s,tol_bissection)
    xs, ys , xl, xr = first_sampling(alpha_0,alpha_1,D,P1,x_start,s)

    x_sampled = xs
    y_sampled = ys
    index = 1
    
    xs = 0
    ys = 0
    xr = 0
    xl = 0
    big_x_sampled = Any[]
    big_y_sampled = Any[]


    pade_x, pade_y = Pade(x_sampled, y_sampled, x_start,N =5000)

    # Find negative peaks/valleys
    pks, vals = findmaxima(-pade_y)
    peaks = []
    for index in pks 
        push!(peaks,pade_x[index])
    end
    print("xi_val_peaks root_solve ", peaks, "\n")
    
    if length(peaks) == 0
        x_negative = -pade_x[end]
        pushfirst!(x_sampled, x_negative)
        pushfirst!(y_sampled, surrogate_function(alpha_0,alpha_1,D,P1,x_negative,s))
        print("xs", xs, "\n")
        print("ys", ys, "\n")
        # Use the padé to see if peaks are predicted in the area that is supposed to be convexity
        x_pade, y_pade = Pade(x_sampled,y_sampled,x_negative)
        pks, vals = findmaxima(-y_pade)
        pushfirst!(peaks,x_negative)
        # for index in pks 
        #     push!(peaks,x_pade[index])
        # end
        print("xi_val_peaks root_solve 2 ", peaks, "\n")
    end

    xn = last(peaks)
    xs = [xn]
    ys = [surrogate_function(alpha_0,alpha_1,D,P1,xn,s)]

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
    zeros = bissection(xl,xr,x_start,10^(-10),100,big_x_sampled,big_y_sampled)
    print("bissection zeros ", zeros, "\n")

    # Keep calling the bisection algorithm until we are within a certain
    # chosen tolerence 
    for it = 1:100000
        print("zeros[1] ", zeros[1], "\n")
        surr_func_xi_zeros = surrogate_function(alpha_0,alpha_1,D,P1,zeros[1],s)
        print("surr_func_xi_zeros before if ", surr_func_xi_zeros, "\n")
        if abs(surr_func_xi_zeros) > tol_bissection
            if surr_func_xi_zeros < 0
                xl = zeros[1]  
            else # Essentially if surr_func_xi_zeros > 0
                xr = zeros[1]
            end

            push!(big_x_sampled[1],zeros[1])
            push!(big_y_sampled[1],surr_func_xi_zeros)
            zeros = bissection(xl,xr,x_start,10^(-10),100,big_x_sampled,big_y_sampled)
    
        elseif abs(surr_func_xi_zeros) < tol_bissection
            print("zeros[1] ", zeros[1], "\n")
            print("surr_func_xi_zeros = surrogate_function(alpha_0,alpha_1,D,P1,zeros[1])", surrogate_function(alpha_0,alpha_1,D,P1,zeros[1],s), "\n")
            print("surr_func_xi_zeros ", surr_func_xi_zeros, "\n")
            return zeros[1],surr_func_xi_zeros
        end 
    end
    print("Went through all of the iterations \n")
    return zeros[1],surr_func_xi_zeros
end


end
