"""
This module is the original Padé approximate code. This code has many 
problems that I (Nicolas) partially fixed in the Nic_BasicPadeRootSolver.jl file. 
The problems with the code in this current file is that the 
Padé rebuild function is inside the Padé approximate construction function. The 
Padé convergence function also doesn't work. In my version of this code, I 
fixed the convergence condition. In my more advanced versions of the code
where I use matrices and the Green operator, I separated the Padé approximate
and the rebuild function. I also fixed a bunch of other stuff not mentionned 
here (see the Latex document). 

The general idea of this Padé approximate root solver algorithm is to find the 
last crossing of a function.

Neil deGrasse  Tyson 

Author: Félix Bergeron 
"""

using Plots
using Distributed

#####################################################################
#The function itself

A_0 = [-4.202369178258174 0.23656026179610068 0.9967817753194017 1.3621812143410215 0.7730591438014547 0.6795174235124107 1.201085140547982 1.4868076811790276; 0.23656026179610068 7.798410364498693 1.1887914375583606 0.8319234390137311 0.6993492068510417 1.3827615231298125 0.700451511387506 0.7056215986781617; 0.9967817753194017 1.1887914375583606 0.42299075325249924 0.29486161901242103 0.19378176034018868 1.000139458265159 0.9244827781178556 1.1508145100468172; 1.3621812143410215 0.8319234390137311 0.29486161901242103 0.01708738384810582 1.1788883010405404 0.3440989888794501 0.8452830180744944 1.031421035373452; 0.7730591438014547 0.6993492068510417 0.19378176034018868 1.1788883010405404 7.7279086959771925 0.5514991511284997 1.2655895299014635 0.4587696224505302; 0.6795174235124107 1.3827615231298125 1.000139458265159 0.3440989888794501 0.5514991511284997 -4.801460017525201 1.5136937331663745 1.5641726026980587; 1.201085140547982 0.700451511387506 0.9244827781178556 0.8452830180744944 1.2655895299014635 1.5136937331663745 -5.772356595393507 0.8863604625475946; 1.4868076811790276 0.7056215986781617 1.1508145100468172 1.031421035373452 0.4587696224505302 1.5641726026980587 0.8863604625475946 3.8057109577380377]  
A_1 = [2.868579595078885 2.0017974394942346 2.74322349984502 1.5384739564067065 1.2049901854320149 0.7844614927869878 0.5116454465287238 0.3954765714958245; 2.0017974394942346 2.1989032020338324 2.2633740613445834 1.860144380106243 1.5067979872002306 1.0776467254653175 0.49796391833968345 0.07010833916308676; 2.74322349984502 2.2633740613445834 3.481910164372317 2.2196452124024675 1.8506900180327013 1.308331972989803 0.9886660735458807 0.8016426250970934; 1.5384739564067065 1.860144380106243 2.2196452124024675 2.365509026271365 2.0482719098769335 1.1736822169267422 0.8666751085840902 0.6206594844699613; 1.2049901854320149 1.5067979872002306 1.8506900180327013 2.0482719098769335 2.5849404532937563 0.8654543936056173 1.1260900464597272 0.8144678880412133; 0.7844614927869878 1.0776467254653175 1.308331972989803 1.1736822169267422 0.8654543936056173 1.2779957150876138 0.6283583221225173 0.09773593867003194; 0.5116454465287238 0.49796391833968345 0.9886660735458807 0.8666751085840902 1.1260900464597272 0.6283583221225173 0.9344991358049112 0.616241342068419; 0.3954765714958245 0.07010833916308676 0.8016426250970934 0.6206594844699613 0.8144678880412133 0.09773593867003194 0.616241342068419 1.0923134840223656]
s_0 = [0.05622552393337055; 0.3954088760578657; 0.8193557283602528; 0.6414037922397975; 0.13623949797842694; 0.30045038855414574; 0.16677768125220171; 0.6697873147229706;;]
s_1 = [0.4991424981445124; 0.3705476346310461; 0.17111573451940554; 0.5073875443016921; 0.42246573217942185; 0.43721333864318745; 0.937501475360908; 0.5148246209305662;;]


# A_0 = [-1.8388144910338902 1.537162859887386 1.6260537184194015 0.4891493835691737 0.5297729966815711 1.20649075539336 0.4097179307262595 1.455197845410053 1.4006438369639957 0.8902678425138039 0.7554536552950705 0.7935778172337057 0.4614807738709841 1.5286199478492521 0.8379617025390981; 1.537162859887386 1.6237693554150732 1.4674855740746577 0.878491516958877 1.1195265286348568 1.4593569638441068 0.7276471084434755 0.708098722462357 1.8046126240399905 1.8977737001440356 1.5720844959151856 1.431395845434377 0.9773853406768878 0.7719096650926657 1.4134271362374393; 1.6260537184194015 1.4674855740746577 -5.137695510097551 1.6498241714794966 1.260724941727228 0.2952483519993292 1.0998878145149455 1.0421838812543962 0.6275208588220019 0.8901555760477567 0.7077702262564088 0.4211986423258237 1.4092894098192779 0.764042434855345 1.5605714441410763; 0.4891493835691737 0.878491516958877 1.6498241714794966 3.3996096683778028 0.99544925616683 1.0608429925678737 1.007776867538234 0.39924728589816816 0.6322823899924496 0.9879093685349578 1.6898299352934112 0.34019582532501835 1.2548926411496035 0.7143996999144029 0.3783315471940958; 0.5297729966815711 1.1195265286348568 1.260724941727228 0.99544925616683 7.174928540067182 1.1339953713058315 1.3334050212398574 0.7611674349225268 1.0220693162684233 1.8086410345194632 0.9412808029485091 0.4614461690302625 1.103832577280131 1.6048656113041853 0.677720920503899; 1.20649075539336 1.4593569638441068 0.2952483519993292 1.0608429925678737 1.1339953713058315 -6.840799647524717 1.0263856079922902 1.315449447148028 0.14388983675017797 1.016573698098672 1.0569117763894607 1.2070149426157601 1.0178382671542998 0.7448626948244583 0.9357633200204456; 0.4097179307262595 0.7276471084434755 1.0998878145149455 1.007776867538234 1.3334050212398574 1.0263856079922902 3.061798738775609 0.9044054694972389 0.7045151674956021 1.7238382217814205 0.4821371121458906 0.8262871076914845 0.24935167237477018 0.933028255917011 1.0550265501198952; 1.455197845410053 0.708098722462357 1.0421838812543962 0.39924728589816816 0.7611674349225268 1.315449447148028 0.9044054694972389 3.5282949896490314 0.8537653806935072 1.5584567740608448 1.3090466757589923 1.3036749034104176 0.45795069416845124 0.583301595775372 1.1313563284864963; 1.4006438369639957 1.8046126240399905 0.6275208588220019 0.6322823899924496 1.0220693162684233 0.14388983675017797 0.7045151674956021 0.8537653806935072 0.6661643202040226 1.4781512015986564 1.454784139567904 1.691807725206574 0.9886271398335092 0.8477692798679983 1.1109230875484082; 0.8902678425138039 1.8977737001440356 0.8901555760477567 0.9879093685349578 1.8086410345194632 1.016573698098672 1.7238382217814205 1.5584567740608448 1.4781512015986564 10.080456415950174 0.9952019988769143 1.825140061986722 1.0155954323118008 1.5072649602706512 1.4947479345531829; 0.7554536552950705 1.5720844959151856 0.7077702262564088 1.6898299352934112 0.9412808029485091 1.0569117763894607 0.4821371121458906 1.3090466757589923 1.454784139567904 0.9952019988769143 0.9791695456663301 1.3138990675339972 1.2599323893897996 1.0506961263270111 1.0542621667651018; 0.7935778172337057 1.431395845434377 0.4211986423258237 0.34019582532501835 0.4614461690302625 1.2070149426157601 0.8262871076914845 1.3036749034104176 1.691807725206574 1.825140061986722 1.3138990675339972 -3.9451765474202127 0.28328715568804286 1.24640828453176 1.0352750315864325; 0.4614807738709841 0.9773853406768878 1.4092894098192779 1.2548926411496035 1.103832577280131 1.0178382671542998 0.24935167237477018 0.45795069416845124 0.9886271398335092 1.0155954323118008 1.2599323893897996 0.28328715568804286 4.898852812033557 1.325939900911326 1.8677600423810046; 1.5286199478492521 0.7719096650926657 0.764042434855345 0.7143996999144029 1.6048656113041853 0.7448626948244583 0.933028255917011 0.583301595775372 0.8477692798679983 1.5072649602706512 1.0506961263270111 1.24640828453176 1.325939900911326 -3.4937610518380495 1.2600183498022526; 0.8379617025390981 1.4134271362374393 1.5605714441410763 0.3783315471940958 0.677720920503899 0.9357633200204456 1.0550265501198952 1.1313563284864963 1.1109230875484082 1.4947479345531829 1.0542621667651018 1.0352750315864325 1.8677600423810046 1.2600183498022526 10.078403836937921]
# A_1 = [4.663203972559549 2.0771997636132724 3.4739932302625243 1.7482711469746897 2.314368760174372 1.3457353252450293 1.9985930178448659 1.2472647430459909 1.6654716297555494 1.1258506312714391 1.5479394145913505 0.651537448561234 0.8888530838989165 0.5568721944173955 0.7157800730766144; 2.0771997636132724 3.4316677750114963 2.3578194887829196 1.8939074290768896 2.260077260290282 1.4388553481258473 2.3649448443815 1.564641755604683 1.2370501830712364 1.4219427718820201 1.5672311910883114 1.4979958739748662 0.10795388517625501 0.20486399119462212 0.04129363179514801; 3.4739932302625243 2.3578194887829196 5.044153677005728 2.2542235463423452 3.31614479577347 1.324054430537071 2.7027106855441976 0.9569584759760215 0.8090327765746451 0.8753277568633683 1.4223380983160387 0.7459806244842191 0.338841794224685 0.26657962779692523 0.27388577338624803; 1.7482711469746897 1.8939074290768896 2.2542235463423452 2.433624807313832 2.8282958031136105 0.6829014907441352 2.0972679325583545 1.2002017737643933 1.1931250508096305 1.4143682786626208 0.9693345196664157 1.0304918275037054 0.3153626876201161 0.14856102002275792 0.2137784854509833; 2.314368760174372 2.260077260290282 3.31614479577347 2.8282958031136105 5.257095240390917 1.5849870434525692 3.2106520496809074 1.449463040633146 1.7374261283773114 1.397290460263844 1.2344571861650142 0.8173632776350772 0.9230216193818517 0.5212706884953361 0.7602913672384242; 1.3457353252450293 1.4388553481258473 1.324054430537071 0.6829014907441352 1.5849870434525692 1.4934208914877292 1.6233890685853904 0.999400694205906 0.9766677801765057 0.6508723061267347 1.1844900009662693 0.5768815094280522 0.3800461572264665 0.34324495309569514 0.31277369369049607; 1.9985930178448659 2.3649448443815 2.7027106855441976 2.0972679325583545 3.2106520496809074 1.6233890685853904 3.720969940943133 1.8673850000447858 1.591042100120842 1.7745605655125498 2.0249085091999905 1.2650435776440518 0.301710664558573 0.24188862477032083 0.19491765520830195; 1.2472647430459909 1.564641755604683 0.9569584759760215 1.2002017737643933 1.449463040633146 0.999400694205906 1.8673850000447858 1.8283197358812995 1.4708837267160835 1.5525534109621058 1.5093702483354108 1.1726508758263818 0.13971535139293323 0.1672610266679974 0.05786181725869782; 1.6654716297555494 1.2370501830712364 0.8090327765746451 1.1931250508096305 1.7374261283773114 0.9766677801765057 1.591042100120842 1.4708837267160835 2.1884062794177916 1.3287774721732397 1.1676123274143981 0.9358748782621715 0.8849222011852677 0.5682344266805711 0.6830348063674935; 1.1258506312714391 1.4219427718820201 0.8753277568633683 1.4143682786626208 1.397290460263844 0.6508723061267347 1.7745605655125498 1.5525534109621058 1.3287774721732397 1.7493958140931622 1.394084815087912 1.2249865637475776 0.1282162259993865 0.0933857458682471 0.03981581324816783; 1.5479394145913505 1.5672311910883114 1.4223380983160387 0.9693345196664157 1.2344571861650142 1.1844900009662693 2.0249085091999905 1.5093702483354108 1.1676123274143981 1.394084815087912 2.2364412577247434 1.3249327933778527 0.2711211358479164 0.29261382226774296 0.19468976576589597; 0.651537448561234 1.4979958739748662 0.7459806244842191 1.0304918275037054 0.8173632776350772 0.5768815094280522 1.2650435776440518 1.1726508758263818 0.9358748782621715 1.2249865637475776 1.3249327933778527 1.6637093235718365 0.07926720811622653 0.17391929698892106 0.020220564260743725; 0.8888530838989165 0.10795388517625501 0.338841794224685 0.3153626876201161 0.9230216193818517 0.3800461572264665 0.301710664558573 0.13971535139293323 0.8849222011852677 0.1282162259993865 0.2711211358479164 0.07926720811622653 1.0168383734370028 0.5338881706094403 0.779483738535041; 0.5568721944173955 0.20486399119462212 0.26657962779692523 0.14856102002275792 0.5212706884953361 0.34324495309569514 0.24188862477032083 0.1672610266679974 0.5682344266805711 0.0933857458682471 0.29261382226774296 0.17391929698892106 0.5338881706094403 0.4480673075621018 0.45489485517287154; 0.7157800730766144 0.04129363179514801 0.27388577338624803 0.2137784854509833 0.7602913672384242 0.31277369369049607 0.19491765520830195 0.05786181725869782 0.6830348063674935 0.03981581324816783 0.19468976576589597 0.020220564260743725 0.779483738535041 0.45489485517287154 0.768318513423218]
# s_0 = [0.5007263089892319; 0.46613284602713845; 0.3640033465334075; 0.537644600000377; 0.5855820274674169; 0.4341260398039898; 0.37374349949212704; 0.8517318966328955; 0.9498730144731261; 0.7708199592309373; 0.6343579358307085; 0.43850994918992736; 0.5616661832991805; 0.12451314272146718; 0.8135086969693183;;]
# s_1 = [0.9298310209412178; 0.0759223366003331; 0.7989906022414206; 0.9492621915451503; 0.46006539237497734; 0.547790797986051; 0.7753272446729406; 0.6287932954865109; 0.3171200354409057; 0.3701325755851471; 0.0019424581098640425; 0.4199384456013111; 0.03288443549400033; 0.26562843318134655; 0.7928848049244925;;]

# Function we want to find the roots of 
function f(x)
    t = inv(A_0 + x*A_1)*(s_0+x*s_1)
    f_0 =  2*real(adjoint(t)*s_1) - adjoint(t)*A_1*t 
    return f_0[1]
end

# Plots the piecewise function
xmax = 100
xmin = 0.0
N = 200
global px = [i for i in range(xmin, xmax, N)]
global py = map(f, px)
ymax = 50
ymin = -200
lbound = 0
# global plt1 = plot(px,py, ylim=(ymin,ymax), legend = false)
# display(plt1)
# sleep(3)   


################################################ Peak finders #################
function peak_finder(xs, ys)  # Finds the peaks within the sampled point 
    slope_criteria = 2 # This needs to be more dynamic, to account for the amount of sampled points
    peaks = []
    slopes = [] 
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

        # Test 1: Checks a sign change in the slope
        if i > 2
            if (slopes[i-2]\slopes[i-1]) < 0 && ys[i-2] < -5 #change of sign of the slope, implying a peak #the ys[i] <-5 is to prevent finding regions between the peaks
                push!(peaks, xs[i-1]) #note: we take xs[i-1] instead of xs[i-2] because the i-1 value is to the right of the peak
                added = true
            end
        end
        # Test2  : Checks if the slope before and after a point stopped growing (indicating that there was a peak)
        if i > 3
            if abs(slopes[i-3]) < abs(slopes[i-2]) && abs(slopes[i-2]) > abs(slopes[i-1]) && added == false && ys[i] <-5 #the ys[i] <-5 is to prevent finding regions between the peaks
                push!(peaks, xs[i-1])
            end
        end
    end
    return peaks
end



################################################### Padé function #############
# x : x values for the sampled points
# y : y values for the sampled points
# N : number of points used to evaluate the approximate once it has been built using x,y 
# xl : left bound from which the Padé is evaluated 
# xr : right bound up to which the Padé is evaluated
# rebuild_with : evaluates the constructed Padé at the x values in this list

function Pade(x, y; N = 500, xl = 0.0, xr = xmax, rebuild_with = [])
    #Padé approximant algorithm
    x = x
    r = y
    l =  length(x)
    R= zeros(l)
    X= zeros(l)
    P= zeros(l)
    S= zeros(l)
    M= zeros(l)

    for i in range(1,l) #first loop 
        R[i] = r[i]
        X[i] = x[i]
    end

    for j in range(1,l)#second loop
        P[j] = R[j]
        for s in range(j+1, l)
            S[s] = R[j]
            R[s] = R[s] - S[s]
            if R[s] == 0
                #println("Huston, we have a problem, ABORT.")
            else
                M[s] = X[s] - X[j]
                R[s] = M[s]/R[s]
                if j-1 == 0 # to avoid indexing at the P[0] position
                    R[s] = R[s]
                else
                    R[s] = R[s] + P[j-1]
                end
            end
        end
    end

    function rebuild(x)  #rebuild the approximation from the little blocks
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
            throw(Error) #not sure what to do when this happens yet, problems occur when N exceeds 336
        else
            return A[l]/B[l]
        end
    end
    local px
    if isempty(rebuild_with)== true
        px = [i for i in range(xl, xr, N)]
        approx = map(rebuild, px)
    else
        px = rebuild_with
        approx = map(rebuild, px)
    end
    return (px, approx)
end


################################################### Convexity test

function convexity_test(x,y, display_plot = false)
    local xl = 0
    local i = 1
    while i < length(x) #while loop to easily skip iteration once we've found a point of non-convexity (impossible with range statement)
        for j in range(i+2, length(x))
            for k in range(i+1,j-1)
                p1 = (x[i], y[i])
                p2 = (x[k], y[k])
                p3 = (x[j], y[j])
                slope = (p3[2] - p1[2])/(p3[1] - p1[1])
                intercept = p1[2] - slope*(p1[1])
                Y = slope*p2[1] + intercept
                # I commented out the plots (Nic)
                # if display_plot  == true
                #     small_x = [i for i in range(p1[1], p3[1],10)]
                #     small_y = [(slope*i + intercept) for i in small_x]
                #     plt = plot(px,py, ylim=(ymin,ymax), legend = false)
                #     #plot!(x,y, markershape = :circle, color = :black)
                #     plot!(small_x, small_y)
                #     plot!([p1[1]], [p1[2]], markershape = :circle, color = :green)
                #     plot!([p3[1]], [p3[2]], markershape = :circle, color = :blue)
                #     plot!([p2[1]], [p2[2]], markershape = :circle, color = :red)
                #     display(plt)#necessary to display the graph (equivalent to plt.show() in python)
                #     #sleep(0.1)  #delay to allow for the graph to display
                # end
                if Y > p2[2]
                    println("sampled point  ",p2[2])
                    println("pedicted point ", Y)
                    println("____________")
                    xl = x[k]
                    i+=1
                end
            end
        end
        i+=1
    end
    return xl
end

################################################### First test that needs to be done
#this test samples the function initially to determine where is the
#region in which the function is definitly above zero

function First_test(x_start)
    xs = Any[x_start]
    ys = Any[f(x_start)]
    xr = x_start
    local xl = 0
    xn = x_start/2
    #check if our starting point gives us a value above 0
    ###### Need completing #####
    if ys[end] <= 0
        while ys[end] <=0
            x_start = 2*x_start
            push!(xs, x_start)
            push!(ys, f(x_start))
        end
    else #if our starting point was adequate, samples from right to left
        while ys[1] > 0
            insert!(xs,1,xn)
            insert!(ys,1,f(xn))
            xn = xn/2
        end
    end
    #one last check to see if it's still above zero later on
    x_start = 1.5*x_start
    push!(xs, x_start)
    push!(ys, f(x_start))
    if ys[end]<0
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
                insert!(ys,j,f(xns[i]))
            end
        end
    end
    #add a random point for test purposes (for the above test)
    # insert!(xs,3, 90)
    # insert!(ys,3,40)
    # insert!(xs,4, 95)
    # insert!(ys,4,70)
    #Use the padé to see if peaks are predicted in the area that is supposed to be convexity
    x_pade, y_pade = Pade(xs,ys)
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
    xr = above_region[2]
    for i in range(1,length(xs))
        if ys[i] < 0
            xl = xs[i]
        end
    end
    return xs, ys, xl, xr
end

################################################### Padé stop criteria test ###
# This function checks wheter that last sampling changed the Padé approximant
# significantly or not. If not, stop sampling.
# It keeps in memory previous sampling to compare them
# Should it dump some of them (would that save memory or once it's assigned it's too late?)
global big_x_sampled = Any[]
global big_y_sampled = Any[]
global errors = Any[]
global ratios = Any[]
global stop_crit = false

function pade_stop_criteria(xs,ys,xl)
    local pade_x1, pade_y1, pade_x2, pade_y2, error
    global big_x_sampled
    global big_y_sampled
    global errors
    global stop_crit
    push!(big_x_sampled,xs)
    push!(big_y_sampled,ys)
    # println(big_x_sampled)
    # println(big_y_sampled)
    # println("xl =", xl)
    if length(big_x_sampled) <= 1
        #do nothing
    elseif length(big_x_sampled) > 1
        #compare samplings
        N = 1000
        pade_x1, pade_y1 = Pade(big_x_sampled[end],big_y_sampled[end],N=N,xl =xl)
        pade_x2, pade_y2 = Pade(big_x_sampled[end-1],big_y_sampled[end-1],N=N,xl =xl)
        # println(pade_x1)
        # println(pade_x2)
        # Calculate error between the two Padé's in between a relevant x range
        error = 0
        for i in range(1,length(pade_x1))
            error += abs(pade_y2[i] - pade_y1[i])/N
        end
        push!(errors, error)
        #Critera to see by how much the error shrunk from one iteration to the next
        if length(errors)>1
            ratio = errors[end-1]/errors[end]
            push!(ratios, ratio)
            # Checks if the error is diminishing with each extra sampling 
            # (needs to diminish twice in a row)
            if length(ratios)>2
                if ratios[end]<ratios[end-1] && ratios[end-1] < ratios[end-2]
                    stop_crit = true
                    println("done")
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
# TO DO : ______________
# Could be rewritten better, modified it to work with the Padé approx
################################
################################
# x1: starting point to the left
# x2 : starting point to the right
# ϵ: Maximum error we want between the actual zero and the approximated one
# N: Maximum number of iteration before returing a value
function bissection(x1,x2,ϵ,N)
    local fxm
    xm = (x1 + x2)/2
    compt = 0
    while abs(x2-x1)/(2*abs(xm)) > ϵ && compt < N
        xm = (x1 + x2)/2
        ans = Pade(big_x_sampled[end],big_y_sampled[end],rebuild_with=[x1,x2,xm])
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
# - Add convexity check to first_test ... but need to fix convexity check first, it's
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


xl = 0
xr = 500 #the search is gonna start here
xs, ys , xl, xr = First_test(xr)
global x_sampled = xs
global y_sampled = ys
global compt = 0


function big_boiii(display_plot = false)
    local plt
    # I commented the plots (Nic)
    # if display_plot ==true
    #     plt = plot(x_sampled,y_sampled, markershape = :circle, color = :green, linewidth = 0, legend = false, ylim =(ymin,ymax),xlim =(xmin,xmax))
    #     plot!(px,py)
    # end
    #the stopping critera is that the error between padés need to diminish
    #for two consecutives  sampling
    while stop_crit == false
        global xs, ys, compt
        local index 
        index = 1
        pade_x, pade_y = Pade(x_sampled, y_sampled, N =5000)
        peaks = peak_finder(pade_x, pade_y)
        print("peaks ", peaks, "\n")
        # println(peaks)
        #sample near the peak but far enough from last sampled point (inbetween both values)
        # lastpeak = peaks[end]
        # plastpeak = peaks[end-1]
        # next = 0
        # ####finds what sampled point is closest to the suspected peak
        # #### in order to sample close to it (inbetween)
        # for i in x_sampled
        #     if i > lastpeak
        #         next = i
        #         break
        #     end
        # end

        # ####this needs cleaning up (xn1, xn2,xn3), might be too much to sample 3 times each round
        # #### also these variables names are confusing
        # xn1 = (lastpeak+next)/2 #this needs cleaning up
        # xn2 = (plastpeak+next)/2
        # xn3 = lastpeak
        # xs = [xn3, xn1]
        # ys = [f(xn3),f(xn1)]        
        xn = last(peaks)
        xs = [xn]
        ys = [f(xn)]

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
        pade_stop_criteria(x_sampled[index:end], y_sampled[index:end],xn)
        #plots the different padés (if needed)

        # I commented out the plots (Nic)
        # if display_plot==true
        #     for i in range(1,length(big_x_sampled))
        #         local pade_x, pade_y
        #         N=5000
        #         pade_x, pade_y = Pade(big_x_sampled[i],big_y_sampled[i],N=N,xl = xn)
        #         plot!(pade_x,pade_y)
        #         plot!(x_sampled,y_sampled, markershape = :circle, linewidth = 0, color = :green)
        #     end
        #     display(plt)
        #     sleep(1)
        # end
        compt+=1
    end

    #We can narrow the position of the zero using our sampled points
    print("x_sampled ",x_sampled,"\n")
    for i in range(1, length(x_sampled))
        sleep(1)
        #println(x_sampled[end-i+1])
        if y_sampled[i]<0
            xl = y_sampled[i]
        end
        if y_sampled[end-i+1] < 0
            print("x_sampled[end-i+1] ",x_sampled[end-i+1],"\n")
            xr = x_sampled[end-i+2]
        end
    end
    # println(xl," ", xr)
    print("xl ", xl, "\n")
    print("xr ", xr, "\n")
    #Finds the zero using the last Padé approximant (bissection method)
    zeros = bissection(xl,xr,10^(-10),100)
    #println("The real root is : ", 65.52353857114213)
    return zeros[1]
end

#println(x_sampled)
ans = big_boiii(false)
println("The guessed root is: ", ans)




########################################### Testing/Plotting section


xmax = 100
xmin = 0.0
N = 2000
ymax = 500
ymin = -2000


# xmax = 55
# xmin = 40
# ymax = 100
# ymin = -1400
# N = 300
# px = [i for i in range(xmin, xmax, N)]
# py = map(f, px)
# # plt = plot(pade_x1,pade_y1, color= :black)
# plot!(pade_x2, pade_y2,xlim =(xmin,xmax), color = :red)
# #plt = plot(px,py, ylim=(ymin,ymax), legend = false)
# plot!(x_sampled,y_sampled, markershape = :circle)
# #Builds a Padé from this 
# ans = Pade(x_sampled, y_sampled)
# x_pade = ans[1]
# y_pade = ans[2]
# #plot!(x_pade, y_pade)
# display(plt)
# pade_x, pade_y = Pade(big_x_sampled[end],big_y_sampled[end],N=N,)
# plot(pade_x,pade_y,xlim = [xmin,xmax],ylim =[ymin,ymax])
# plot!(x_sampled,y_sampled, markershape = :circle, linewidth = 0, color = :green)
# plot!(px,py)