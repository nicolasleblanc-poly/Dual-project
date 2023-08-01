# Green function operator has already been complied vecSrcAA -> vecTrgAA,
# vecSrcAA begins as all zeros
# Memory declaration 
greenCells = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
greenDenseMat = Array{ComplexF64}(undef, 3 * prod(cellsA), 3 * prod(cellsA))
greenDenseAsm = Array{ComplexF64}(undef, 3 * prod(cellsA), 3 * prod(cellsA))
gDDAsm = Array{ComplexF64}(undef, 3 * prod(cellsA), 3 * prod(cellsA))

linItr = 0
srcScl = prod(scaleA)

@time for dirItr in 1 : 3, cellZItr in 1 : cellsA[3], cellYItr in 1 : cellsA[2], 
	cellXItr in 1 : cellsA[1] 

	global linItr = LinearIndices(greenCells)[cellXItr, cellYItr, cellZItr, dirItr]

	if linItr > 1
	
		gMemSlfN.srcVec[linItr - 1] = 0.0 + 0.0im

	end
	# Set source.
	gMemSlfN.srcVec[linItr] = (1.0 + 0.0im) 
	# Calculate resulting field.
	greenActAA!()
	# Save field result.
	copyto!(view(greenDenseMat, :, linItr), gMemSlfN.trgVec);
end
# Clean up gMemSlfN.srcVec. 
gMemSlfN.srcVec[3 * prod(cellsA)] = 0.0 + 0.0im
# Discrete dipole asymG
# anaTest.jl must be included prior to this function in order to define
# greenAna. 
function greenAsmDiscreteDipole(vol::MaxGVol, gDDAsm::Array{ComplexF64,2})::Nothing	

	# Memory allocations
	linItr = 1
	srcPos = zeros(3)	
	srcVec = zeros(ComplexF64, 3)
	
	cellsV = vol.cells
	trgRng = copy(vol.grid)
	
	greenCells = Array{ComplexF64}(undef, cellsV[1], cellsV[2], cellsV[3], 3)
	gDDWrk = similar(gDDAsm) 

	for dirItr in 1 : 3, cellZItr in 1 : cellsV[3], cellYItr in 1 : cellsV[2], 
	cellXItr in 1 : cellsV[1] 

		linItr = LinearIndices(greenCells)[cellXItr, cellYItr, cellZItr, dirItr]
		
		if dirItr == 1

			srcVec = [1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im]

		elseif dirItr == 2

			srcVec = [0.0 + 0.0im, 1.0 + 0.0im, 0.0 + 0.0im]

		else

			srcVec = [0.0 + 0.0im, 0.0 + 0.0im, 1.0 + 0.0im]
		end

		srcPos = [vol.grid[1][cellXItr], vol.grid[2][cellYItr], 
		vol.grid[3][cellZItr]]

		greenAna(view(gDDAsm, :, linItr), vol, trgRng, srcPos, srcVec)
	end

	adjoint!(gDDWrk, gDDAsm)
	lmul!(0.0 - 0.5im, gDDAsm)
	axpy!(0.0 + 0.5im, gDDWrk, gDDAsm)	

	return nothing 
end
# Computations
adjoint!(greenDenseAsm, greenDenseMat);
lmul!((0.0 + 0.5im) * conj(assemblyInfo.freqPhase)^2, greenDenseAsm);
axpy!((0.0 - 0.5im) * (assemblyInfo.freqPhase)^2, greenDenseMat, 
	greenDenseAsm);
# Discrete dipole approximation
greenAsmDiscreteDipole(gMemSlfN.srcVol, gDDAsm);
# Numerical eigenvalues of anti-symmetric component of the Green function
sVals = eigvals(greenDenseAsm);