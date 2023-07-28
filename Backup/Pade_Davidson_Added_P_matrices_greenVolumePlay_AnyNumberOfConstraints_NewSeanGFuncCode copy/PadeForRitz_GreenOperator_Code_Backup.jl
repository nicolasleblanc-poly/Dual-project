module PadeForRitz_GreenOperator_Code
using Plots, Distributed, Random, LinearAlgebra, gmres
export surrogate_function,plot_surrogate_func,root_solve
###############################The function itself############################# 

# A_0 = [-4.202369178258174 0.23656026179610068 0.9967817753194017 1.3621812143410215 0.7730591438014547 0.6795174235124107 1.201085140547982 1.4868076811790276; 0.23656026179610068 7.798410364498693 1.1887914375583606 0.8319234390137311 0.6993492068510417 1.3827615231298125 0.700451511387506 0.7056215986781617; 0.9967817753194017 1.1887914375583606 0.42299075325249924 0.29486161901242103 0.19378176034018868 1.000139458265159 0.9244827781178556 1.1508145100468172; 1.3621812143410215 0.8319234390137311 0.29486161901242103 0.01708738384810582 1.1788883010405404 0.3440989888794501 0.8452830180744944 1.031421035373452; 0.7730591438014547 0.6993492068510417 0.19378176034018868 1.1788883010405404 7.7279086959771925 0.5514991511284997 1.2655895299014635 0.4587696224505302; 0.6795174235124107 1.3827615231298125 1.000139458265159 0.3440989888794501 0.5514991511284997 -4.801460017525201 1.5136937331663745 1.5641726026980587; 1.201085140547982 0.700451511387506 0.9244827781178556 0.8452830180744944 1.2655895299014635 1.5136937331663745 -5.772356595393507 0.8863604625475946; 1.4868076811790276 0.7056215986781617 1.1508145100468172 1.031421035373452 0.4587696224505302 1.5641726026980587 0.8863604625475946 3.8057109577380377]  
# A_1 = [2.868579595078885 2.0017974394942346 2.74322349984502 1.5384739564067065 1.2049901854320149 0.7844614927869878 0.5116454465287238 0.3954765714958245; 2.0017974394942346 2.1989032020338324 2.2633740613445834 1.860144380106243 1.5067979872002306 1.0776467254653175 0.49796391833968345 0.07010833916308676; 2.74322349984502 2.2633740613445834 3.481910164372317 2.2196452124024675 1.8506900180327013 1.308331972989803 0.9886660735458807 0.8016426250970934; 1.5384739564067065 1.860144380106243 2.2196452124024675 2.365509026271365 2.0482719098769335 1.1736822169267422 0.8666751085840902 0.6206594844699613; 1.2049901854320149 1.5067979872002306 1.8506900180327013 2.0482719098769335 2.5849404532937563 0.8654543936056173 1.1260900464597272 0.8144678880412133; 0.7844614927869878 1.0776467254653175 1.308331972989803 1.1736822169267422 0.8654543936056173 1.2779957150876138 0.6283583221225173 0.09773593867003194; 0.5116454465287238 0.49796391833968345 0.9886660735458807 0.8666751085840902 1.1260900464597272 0.6283583221225173 0.9344991358049112 0.616241342068419; 0.3954765714958245 0.07010833916308676 0.8016426250970934 0.6206594844699613 0.8144678880412133 0.09773593867003194 0.616241342068419 1.0923134840223656]
# s_0 = [0.05622552393337055; 0.3954088760578657; 0.8193557283602528; 0.6414037922397975; 0.13623949797842694; 0.30045038855414574; 0.16677768125220171; 0.6697873147229706;;]
# s_1 = [0.4991424981445124; 0.3705476346310461; 0.17111573451940554; 0.5073875443016921; 0.42246573217942185; 0.43721333864318745; 0.937501475360908; 0.5148246209305662;;]

# Function we want to find the roots of 
function surrogate_function(alpha_0,alpha_1,P0,xi,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P)
    # sz = size(D)[1]
    dim = cellsA[1]*cellsA[2]*cellsA[3]*3
    # Matrix
    # print("xi ", xi, "\n")
    # P0 = adjoint(D)*D
    # A = (alpha_1+xi)*P0+alpha_0*P1
    # A = (alpha_1+xi)*I + alpha_0*inv(adjoint(D))*P1*inv(D)
    # RND vector
    s = Vector{ComplexF64}(undef,dim)
    rand!(s)
    # Direct solve
    x = GMRES_Pade(s,xi,alpha_0,alpha_1,P0,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P) 
    # x = A\s
    # Surrogate function 
    # surr_func = real(imag(adjoint(x)*s)-adjoint(x)*P0*x)
    surr_func = real(real(adjoint(x)*s)-adjoint(x)*P0*x)
    # print("surr_func ", surr_func, "\n")
    return surr_func[1]
end

function plot_surrogate_func(alpha_0,alpha_1,P0,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P,max_val,ind)
    # sz = size(D)[1]
    # dim = cellsA[1]*cellsA[2]*cellsA[3]*3
    # P0 = adjoint(D)*D
    xi_vals = LinRange(0,max_val,max_val) 
    surr_func_vals = Vector{Float64}(undef,max_val)
    # RND vector
    # s = Vector{ComplexF64}(undef,dim)
    # rand!(s)
    for i in eachindex(xi_vals)
        # Matrix
        # A = (alpha_1+xi_vals[i])*P0+alpha_0*P1
        # A = (alpha_1+xi_vals[i])*I + alpha_0*inv(adjoint(D))*P1*inv(D)
        # x = A\s
        # x = GMRES_Pade(s,xi_vals[i],alpha_0,alpha_1,P0,cellsA,gMemSlfN,gMemSlfA,
        #     chi_inv_coeff,P) 
        surr_func_vals[i] = surrogate_function(alpha_0,alpha_1,P0,xi_vals[i],cellsA,
            gMemSlfN,gMemSlfA,chi_inv_coeff,P)
        # surr_func_vals[i] = real(imag(adjoint(x)*s)-adjoint(x)*P0*x)
    end
    print("size(xi_vals) ", size(xi_vals), "\n")
    print("size(surr_func_vals) ", size(surr_func_vals), "\n")
    # print("surr_func_vals ", surr_func_vals, "\n")
    
    # plot(xi_vals, surr_func_vals, title = "Full graph: Surrogate function vs xi",)  
    # xlabel!("Xi")
    # ylabel!("Surrogate function")
    # savefig("/home/nic-molesky_lab/Github-Research-2023/Padé and Ritz_Davidson/Graphs/full_graph_surr_func_vs_xi.png")
    
    if ind == 1
        plot(xi_vals, surr_func_vals, title = "Zoomed-in graph: Surrogate function vs xi",)  
        xlabel!("Xi")
        ylabel!("Surrogate function")
        # xlims!([0,300])
        # yaxis(yvals=[-2:2])
        savefig("/home/nic-molesky_lab/Github-Research-2023/Pade_Davidson_Added_P_matrices_greenVolumePlay_AnyNumberOfConstraints_NewSeanGFuncCode copy/Graphs/Pade_Ritz_HarmonicRitz_Graphs/zoomed_operator1_graph_surr_func_vs_xi.png")
    
    elseif ind == 2
        plot(xi_vals, surr_func_vals, title = "Zoomed-in graph: Surrogate function vs xi",)  
        xlabel!("Xi")
        ylabel!("Surrogate function")
        # xlims!([0,300])
        # yaxis(yvals=[-2:2])
        savefig("/home/nic-molesky_lab/Github-Research-2023/Pade_Davidson_Added_P_matrices_greenVolumePlay_AnyNumberOfConstraints_NewSeanGFuncCode copy/Graphs/Pade_Ritz_HarmonicRitz_Graphs/zoomed_operator2_graph_surr_func_vs_xi.png")
    
    end
end 

################################################ Peak finders
function peak_finder(xs, ys)  # Finds the peaks within the sampled point 
    slope_criteria = 2 # This needs to be more dynamic, to account for the amount of sampled points
    peaks = []
    slopes = [] 
    # print("xs ", xs, "\n")
    # print("ys ", ys, "\n")
    for i in range(2, length(xs))
        added = false #verify if a peak has been added twice
        slope =  (ys[i]-ys[i-1])/(xs[i]-xs[i-1])
        push!(slopes, slope)

        # println(i)
        # println(slope)
        # println(slopes)
        # Test : This one seems like a bad test to conduct, doesnt narrow down the position enough
        # if slope > slope_criteria
        #     push!(peaks,xs[i])
        #     # println(slope)
        #     # println(" ")
        # end

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
    # print("xl in pade ", xl, "\n")
    # print("xr in pade ", xr, "\n")
    # print("isempty(rebuild_with)== true ", isempty(rebuild_with)== true, "\n")
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
    # print("x ", x, "\n")
    # print("l ", l, "\n")
    # print("P ", P, "\n")
    # print("X ", X, "\n")

    A = zeros(l)
    B = zeros(l)
    A[1] = P[1]
    B[1] = 1

    # print("P[1] ", P[1], "\n")
    # print("P[2] ", P[2], "\n")
    # print("x ", x, "\n")
    # print("X[1] ", X[1], "\n")

    A[2] = P[2]*P[1] + (x-X[1])
    B[2] = P[2]
    for i in range(2, l-1)
        A[i+1] = A[i]*(P[i+1] -P[i-1]) + A[i-1]*(x-X[i])
        B[i+1] = B[i]*(P[i+1] -P[i-1]) + B[i-1]*(x-X[i])
    end

    # print("A[l] ", A[l], "\n")

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

function first_sampling(alpha_0,alpha_1,P0,x_start,cellsA,gMemSlfN,gMemSlfA,
        chi_inv_coeff,P)
    # surrogate_function(alpha_0,alpha_1,P0,xi,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P)
    print("x_start ", x_start, "\n")
    xs = Any[x_start]
    ys = Any[surrogate_function(alpha_0,alpha_1,P0,x_start,cellsA,gMemSlfN,
        gMemSlfA,chi_inv_coeff,P)]
    # ys = Any[f(x_start)]
    xr = x_start
    local xl = 0
    xn = x_start/2
    # Check if our starting point gives us a value above 0
    ###### Need completing #####
    if ys[end] <= 0
        while ys[end] <=0
            x_start = 2*x_start
            push!(xs, x_start)
            push!(ys,surrogate_function(alpha_0,alpha_1,P0,
                x_start,cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P))
            # push!(ys, f(x_start))
        end
    else # If our starting point was adequate, samples from right to left
        while ys[1] > 0
            insert!(xs,1,xn)
            insert!(ys,1,surrogate_function(alpha_0,alpha_1,P0,xn,cellsA,
                gMemSlfN,gMemSlfA,chi_inv_coeff,P))
            # insert!(ys,1,f(xn))
            xn = xn/2
        end
    end
    # One last check to see if it's still above zero later on
    x_start = 1.5*x_start
    push!(xs, x_start)
    push!(ys, surrogate_function(alpha_0,alpha_1,P0,x_start,cellsA,gMemSlfN,
        gMemSlfA,chi_inv_coeff,P))
    
    print("xs", xs, "\n")
    print("ys", ys, "\n")
    
    # push!(ys, f(x_start))
    if ys[end] < 0
        println("Need to check this scenario")
    end
    #check how many points we already have sampled to determine 
    #how much more sampling needs to be done 
    #(we're going to do one sampling in between every already sampled points)
    xns = Any[]
    for i in range(1, length(xs)-1)
        push!(xns, (xs[i+1]+xs[i])/2)
    end
    for i in range(1, length(xns))
        for j in range(2, length(xs))
            if xns[i] < xs[j] && xns[i] > xs[j-1]
                insert!(xs, j, xns[i])
                insert!(ys,j,surrogate_function(alpha_0,alpha_1,P0,xns[i],
                    cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P))
                # insert!(ys,j,f(xns[i]))       
            end
        end
    end
    #add a random point for test purposes (for the above test)
    # insert!(xs,3, 90)
    # insert!(ys,3,40)
    # insert!(xs,4, 95)
    # insert!(ys,4,70)
    #Use the padé to see if peaks are predicted in the area that is supposed to be convexity
    x_pade, y_pade = Pade(xs,ys,x_start)
    print("x_pade 1", x_pade, "\n")
    print("y_pade 1 ", y_pade, "\n")
    peaks = peak_finder(x_pade, y_pade)

    ########################
    ##### Some investigating needs to be done to determine what causes
    ##### convexity_test to give false positives when checking the padé_approx
    ####  This issue doesnt arrise with Peak_finder
    # convexs1 = convexity_test(xs,ys)
    # convexs2 = convexity_test(x_pade, y_pade, true)
    # println(peaks)
    # println(convexs2)

    #determine which region is above the x axis
    above_region = xs
    for i in range(1,length(xs))
        if ys[i] < 0
            above_region = xs[i:end]
        end
    end
    #check if there is a predicted peak wihtin the "above" sampled region
    if isempty(peaks) == false
        for i in range(1, length(peaks))
            if peaks[i] > above_region[1] && peaks[i] < above_region[end]
                println("There is a predicted peak in the above region, do something about it")
            end
        end
    end
    #Determines the right and left bound in which more sampling might be needed
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
# Should it dump some of them (would that save memory or once it's assigned it's too late?)
# global big_x_sampled = Any[]
# global big_y_sampled = Any[]
# global errors = Any[]
# global ratios = Any[]
# global stop_crit = false

function pade_stop_criteria(xl,x_start,big_x_sampled,big_y_sampled,errors)
    local pade_x1, pade_y1, pade_x2, pade_y2, error
    # big_x_sampled = Any[]
    # big_y_sampled = Any[]
    # errors = Any[]
    stop_crit = false
    # push!(big_x_sampled,xs)
    # push!(big_y_sampled,ys)
    # println(big_x_sampled)
    # println(big_y_sampled)
    # println("xl =", xl)
    if length(big_x_sampled) <= 1
        #do nothing
        return errors,stop_crit
    elseif length(big_x_sampled) > 1
        #compare samplings
        N = 1000
        pade_x1, pade_y1 = Pade(big_x_sampled[end],big_y_sampled[end],x_start,N=N,xl=xl)
        pade_x2, pade_y2 = Pade(big_x_sampled[end-1],big_y_sampled[end-1],x_start,N=N,xl=xl)
        # println(pade_x1)
        # println(pade_x2)
        #calculate error between the two Padé's in between a relevant x range
        error = 0
        for i in range(1,length(pade_x1))
            error += abs(pade_y2[i] - pade_y1[i])/N
        end
        push!(errors, error)
        #Critera to see by how much the error shrunk from one iteration to the next
        if length(errors)>1
            ratio = errors[end-1]/errors[end]
            print("ratio ", ratio, "\n")
            push!(ratios, ratio)
            #checks if the error is diminishing with each extra sampling (needs to diminish twice in a row)
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






################################################## Padé informed sampling

# function pade_sampling(xs,ys, display_plot = false )




# end


###################################################### Bissection root finding
############################### 
###############################
#TO DO : ______________
#Could be rewritten better, modified it to work with the Padé approx
################################
################################
#x1: starting point to the left
#x2 : starting point to the right
#ϵ: Maximum error we want between the actual zero and the approximated one
#N: Maximum number of iteration before returing a value
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





#######################################################
######################################################
#LETS LOOP THESE TESTS !

#Doing
# - Modify the bissection method algorithm to work with the sampled points ✓
#   instead of the function. ✓
# - Add a Padé/or/modify function for rebuilding the Padé from various inputs once the 
#   Padé has already been built.
# - Finish the error between approximates code ✓
# - Add a criteria for the error for the position of zero (stopping criteria) 
# - Add convexity check to first_test ... but need to fix convexity check first, it'S
#   still broken ... :(

#To do
#1)Implement a Padé test that once a lot of points have been sampled, we take a padé approx
#  of our sampled points, then take one more point and build another padé approx).
#  If the difference between the padé isn't big, it means that we don't really needed
#  to sample anymore. This is the stopping criteria. The padé needs to be evaluated only
#  near the zero to avoid disturbance of sampling near the peaks (which could modify the padé greatly)
#2)Create a Padé test
#2.1)Create a Padé test that informs sampling (if a peak is guessed, sample there)
#3)Test peak_finder on the padé approximate that's been built (how good is it at finding where the peak is)
#4)Add convexity check to First_test.
#4.1)Investigate why the convexity check breaks with the padé approx in first_test
#5)Build a routine that utilizes all the tests to make the full algorithm
#6)Test with more matrices
#7)Write and read from file
#8)Make Montecarlo simulation that outputs the success rate, and the systems on which the 
#  test failed
#8.1)Make an algorithm that finds the first zero for sure everytime 
#    (not efficient with uniform sampling with right to left search)


function root_solve(alpha_0,alpha_1,P0,x_start,cellsA,gMemSlfN,gMemSlfA,
    chi_inv_coeff,P)
    # xl = 0
    # xr = 2000 # The search is gonna start here
    # The search is going to start at the xr value
    # xl: Left x-value, xr: right x-value
    xs, ys , xl, xr = first_sampling(alpha_0,alpha_1,P0,x_start,cellsA,gMemSlfN,
        gMemSlfA,chi_inv_coeff,P)
    print("xs ", xs, "\n")
    print("ys ", ys, "\n")
    print("xl ", xl, "\n")
    print("xr ", xr, "\n")
    x_sampled = xs
    y_sampled = ys
    # x_sampled = x_start
    # y_sampled = y_start
    compt = 0
    
    xs = 0
    ys = 0
    xr = 0
    xl = 0
    big_x_sampled = Any[]
    big_y_sampled = Any[]
    errors = Any[]
    local plt
    if display_plot == true
        plt = plot(x_sampled,y_sampled, markershape = :circle, color = :green, linewidth = 0, legend = false, ylim =(ymin,ymax),xlim =(xmin,xmax))
        plot!(px,py)
    end
    # The stopping critera is that the error between padés need to diminish
    # for two consecutive samplings.
    # while stop_crit == false
    for i = 1:100000
        # global xs, ys, compt
        # local index 
        index = 1
        pade_x, pade_y = Pade(x_sampled, y_sampled, x_start,N =5000)
        peaks = peak_finder(pade_x, pade_y)
        print("peaks ", peaks, "\n")
           
        xn = last(peaks)
        xs = [xn]
        ys = [surrogate_function(alpha_0,alpha_1,P0,xn,cellsA,gMemSlfN,gMemSlfA,
            chi_inv_coeff,P)]
        # ys = [f(xn)]
        

        ###### Adds the new sampled values in order to the already sampled ones (in ascending x order)
        ###### This could be made into it's own function ...
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
        print("xn ", xn, "\n")
        print("x_sampled ", x_sampled, "\n")
        #checks if this sampling has changed something about our evaluation
        push!(big_x_sampled,x_sampled[index:end])
        push!(big_y_sampled,y_sampled[index:end])
        errors,stop_crit = pade_stop_criteria(xn,x_start,big_x_sampled,big_y_sampled,errors)
        # pade_stop_criteria(xs,ys,xl,big_x_sampled,big_y_sampled,errors)
        
        print("stop_crit ", stop_crit, "\n")
        if stop_crit == false
            break
        end
        #plots the different padés (if needed)
        if display_plot==true
            for i in range(1,length(big_x_sampled))
                local pade_x, pade_y
                N=5000
                pade_x, pade_y = Pade(big_x_sampled[i],big_y_sampled[i],x_start,N=N,xl=xn)
                plot!(pade_x,pade_y)
                plot!(x_sampled,y_sampled, markershape = :circle, linewidth = 0, color = :green)
            end
            display(plt)
            sleep(1)
        end
        compt+=1
    end

    #We can narrow the position of the zero using our sampled points
    print("x_sampled ", x_sampled, "\n")
    print("y_sampled ", y_sampled, "\n")
    got_xl = false
    # for i in range(1, length(x_sampled))
    for i in range(1, length(x_sampled)-1)
        sleep(1)
        #println(x_sampled[end-i+1])
        if y_sampled[i]<0
            # xl = y_sampled[i]
            xl = x_sampled[i]
            print("xl in if statement ", xl, "\n")
            got_xl = true
        end
        # if y_sampled[end-i+1] < 0
        #     print("x_sampled[end-i+1] ",x_sampled[end-i+1],"\n")
        #     xr = x_sampled[end-i+2]
        #     print("xr in if statement ", xr, "\n")
        # end
        if y_sampled[i+1] > 0 && got_xl == true
            print("x_sampled[end-i+1] ",x_sampled[end-i+1],"\n")
            # xr = x_sampled[end-i+2]
            xr = x_sampled[i+1]
            print("xr in if statement ", xr, "\n")
        end
        got_xl = false
    end
    # print(xl," ", xr)
    print("xl ", xl, "\n")
    print("xr ", xr, "\n")
    # Finds the zero using the last Padé approximant (bissection method)
    # zeros = bissection(xl,xr,10^(-10),100,big_x_sampled,big_y_sampled)
    zeros = bissection(xl,xr,x_start,10^(-10),100,big_x_sampled,big_y_sampled)
    print("bissection zeros ", zeros, "\n")

    surr_func_xi_zeros = surrogate_function(alpha_0,alpha_1,P0,zeros[1],cellsA,
        gMemSlfN,gMemSlfA,chi_inv_coeff,P)
    print("surr_func_xi_zeros : ", surr_func_xi_zeros, "\n")

    for it = 1:1000
        print("it ", it, "\n")
        if abs(surr_func_xi_zeros) > 1e-4 # Add tolerance parameter
            # 1. Change 
            # 2. Corner case code
            # 3. 
            if surr_func_xi_zeros < 0
                xl = zeros[1]  
            else # Essentially if surr_func_xi_zeros > 0
                xr = zeros[1]
            end
            # print("xl ", xl, "\n")
            # print("xr ", xr, "\n")

            # print("big_x_sampled ", big_x_sampled, "\n")
            # print("big_y_sampled ", big_y_sampled, "\n")

            push!(big_x_sampled[1],zeros[1])
            push!(big_y_sampled[1],surr_func_xi_zeros)
            zeros = bissection(xl,xr,x_start,10^(-10),100,big_x_sampled,big_y_sampled)
            # print("bissection zeros 2 ", zeros, "\n")
            surr_func_xi_zeros = surrogate_function(alpha_0,alpha_1,P0,zeros[1],
                cellsA,gMemSlfN,gMemSlfA,chi_inv_coeff,P)
            # print("surr_func_xi_zeros 2 : ", surr_func_xi_zeros, "\n")
    
        #println("The real root is : ", 65.52353857114213)
        else # surr_func_xi_zeros < 0.01
            return zeros[1],surr_func_xi_zeros
        end
        
    end
    print("Went through all of the iterations \n")
    return zeros[1],surr_func_xi_zeros
end

# ans = root_solve(x_sampled,y_sampled,false)
# println("The guessed root is: ", ans)

end
