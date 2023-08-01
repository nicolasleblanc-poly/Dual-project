### Load necessary packages for tests.
include("./tests/preamble.jl")
# Define test volume, all lengths are defined relative to the wavelength. 
# Number of cells in the volume. 
cellsA = [2, 2, 2]
cellsB = [1, 1, 1]
# Edge lengths of a cell relative to the wavelength. 
scaleA = (0.1, 0.1, 0.1)
scaleB = (0.2, 0.2, 0.2)
# Center position of the volume. 
coordA = (0.0, 0.0, 0.0)
coordB = (0.0, 0.0, 1.0)
### Prepare Green functions
println("Green function construction started.")
# Create domains
assemblyInfo = MaxGAssemblyOpts()
aDom = MaxGDom(cellsA, scaleA, coordA) 
bDom = MaxGDom(cellsB, scaleB, coordB)
## Prepare Green function operators. 
# First number is adjoint mode, set to zero for standard operation. 
gMemSlfN = MaxGOprGenSlf(0, assemblyInfo, aDom)
gMemSlfA = MaxGOprGenSlf(1, assemblyInfo, aDom)
gMemExtN = MaxGOprGenExt(0, assemblyInfo, aDom, bDom)

# gMemSlfN.srcVec[:] = blabla 

# # Operator shorthands
# greenActAA! = () -> grnOpr!(gMemSlfN) 
# greenAdjActAA! = () -> grnOpr!(gMemSlfA)
# greenActAB! = () -> grnOpr!(gMemExtN)
# println("Green function construction completed.")

# outpout = memory loc
# gMemSlfN.trgVec.copyto(output)

#=
The following Green functions are now created for testing. 
greenActAA!() 		gMemSlfN.srcVec -> gMemSlfN.trgVec
greenAdjActAA!()	gMemSlfA.srcVec -> gMemSlfA.trgVec
greenActAB!() 		gMemExtN.srcVec -> gMemExtN.trgVec
=#
# greenActAA!() 
# greenAdjActAA!()
# greenActAB!()

## Integral convergence test
# println("Integral convergence test started.")
# include("./tests/intConTest.jl")
# println("Integral convergence test completed.")
## Analytic test
# println("Analytic test started.")
# include("./tests/anaTest.jl")
# println("Analytic test completed.")
## Positive semi-definiteness check
# Constructs real space Green function for volume A. 
# Number of cells of cells should not surpass [16,16,16]
# println("Semi-definiteness test started.")
# include("./tests/posDefTest.jl")
# println("Semi-definiteness test completed.")