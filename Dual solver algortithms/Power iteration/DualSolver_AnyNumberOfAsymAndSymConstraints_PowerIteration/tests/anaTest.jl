const π = 3.1415926535897932384626433832795028841971693993751058209749445923
const sepTol = 1.0e-9
## Define analytic Green function
function greenAna(out::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	vol::MaxGVol, trgRng::Array{<:StepRangeLen,1}, srcPos::Array{Float64,1}, 
	srcVec::Array{ComplexF64,1})::Nothing
	
	# Memory allocation
	linItrs = zeros(Int64,3)
	greenCells = Array{ComplexF64}(undef, vol.cells[1], 
		vol.cells[2], vol.cells[3], 3)
	# Separation magnitude and unit vector.
	sep = 0.0
	sh = zeros(ComplexF64,3)
	# Operators components.
	shs = zeros(ComplexF64,3,3)
	id = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
	greenPair = zeros(ComplexF64,3,3) 

	for indZ in 1 : length(trgRng[3]), indY in 1 : length(trgRng[2]), 
		indX in 1 : length(trgRng[1])

		# Calculate index positions
		for dirItr in 1 : 3

			linItrs[dirItr] = 
			LinearIndices(greenCells)[indX, indY, indZ, dirItr]
		end

		sep = 2 * π * sqrt((trgRng[1][indX] - srcPos[1])^2 + 
			(trgRng[2][indY] - srcPos[2])^2 + 
			(trgRng[3][indZ] - srcPos[3])^2)

		if sep < sepTol

			mul!(view(out, linItrs), (2 * π^2  * 2 * im / 3) .* id, srcVec)

		else

			sh = (2 * π) .* (trgRng[1][indX] - srcPos[1], 
				trgRng[2][indY] - srcPos[2], trgRng[3][indZ] - srcPos[3]) ./ sep 
			
			sHs = [(sh[1] * sh[1]) (sh[1] * sh[2]) (sh[1] * sh[3]); 
		 		(sh[2] * sh[1]) (sh[2] * sh[2]) (sh[2] * sh[3]);  
		 		(sh[3] * sh[1]) (sh[3] * sh[2]) (sh[3] * sh[3])]

			greenPair = (2 * π^2 * exp(im * sep) / sep) .* 
					(((1 + (im * sep - 1) / sep^2) .* id) .- 
					((1 + 3 * (im * sep - 1) / sep^2) .* sHs))
			
			mul!(view(out, linItrs), greenPair, srcVec)
		end
	end

	return nothing 
end 

function greenAsm(out::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	vol::MaxGVol, trgRng::Array{<:StepRangeLen,1}, srcPos::Array{Float64,1}, 
	srcVec::Array{ComplexF64,1})::Nothing
	
	# Memory allocation
	linItrs = zeros(Int64,3)
	greenCells = Array{ComplexF64}(undef, vol.cells[1], 
		vol.cells[2], vol.cells[3], 3)
	# Separation magnitude and unit vector.
	sep = 0.0
	sh = zeros(ComplexF64,3)
	# Operators components.
	shs = zeros(ComplexF64,3,3)
	id = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
	gAPair = zeros(ComplexF64,3,3) 

	for indZ in 1 : length(trgRng[3]), indY in 1 : length(trgRng[2]), 
		indX in 1 : length(trgRng[1])

		sep = 2 * π * sqrt((trgRng[1][indX] - srcPos[1])^2 + 
			(trgRng[2][indY] - srcPos[2])^2 + 
			(trgRng[3][indZ] - srcPos[3])^2)
		# Calculate index positions
		for dirItr in 1 : 3

			linItrs[dirItr] = 
			LinearIndices(greenCells)[indX, indY, indZ, dirItr]
		end

		if sep < sepTol

			mul!(view(out, linItrs), (2 * π^2  * 2 / 3) .* id, srcVec)
			
		else

			sh = (2 * π) .* (trgRng[1][indX] - srcPos[1], 
				trgRng[2][indY] - srcPos[2], trgRng[3][indZ] - srcPos[3]) ./ sep 
			
			sHs = [(sh[1] * sh[1]) (sh[1] * sh[2]) (sh[1] * sh[3]); 
		 		(sh[2] * sh[1]) (sh[2] * sh[2]) (sh[2] * sh[3]);  
		 		(sh[3] * sh[1]) (sh[3] * sh[2]) (sh[3] * sh[3])]

			gAPair = (((2 * π^2 * exp(im * sep) / sep) .* 
					(((1 + (im * sep - 1) / sep^2) .* id) .- 
					((1 + 3 * (im * sep - 1) / sep^2) .* sHs))) .-
					((2 * π^2 * exp(-im * sep) / sep) .* 
					(((1 + (-im * sep - 1) / sep^2) .* id) .- 
					((1 + 3 * (-im * sep - 1) / sep^2) .* sHs)))) ./ (2 * im)
			
			mul!(view(out, linItrs), gAPair, srcVec)
		end
	end

	return nothing 
end 
# Prepare memory
gMemSlfN.srcVec[1,1,1,3] = (1.0 + 0.0im) / prod(scaleA)
srcVec = [0.0 + 0.0im, 0.0 + 0.0im, (1.0 + 0.0im)]
srcPos = [gMemSlfN.srcVol.grid[1][1], gMemSlfN.srcVol.grid[2][1], 
	gMemSlfN.srcVol.grid[3][1]]
trgRng = copy(gMemSlfN.srcVol.grid)
anaOut = Array{ComplexF64}(undef, 3 * prod(gMemSlfN.srcVol.cells))
numOut = Array{ComplexF64}(undef, length(gMemSlfN.srcVol.grid[1]), 
	length(gMemSlfN.srcVol.grid[2]), length(gMemSlfN.srcVol.grid[3]), 3)
asmOut = Array{ComplexF64}(undef, length(gMemSlfN.srcVol.grid[1]), 
	length(gMemSlfN.srcVol.grid[2]), length(gMemSlfN.srcVol.grid[3]), 3)
## Preform computations
# Finite element
@time greenActAA!()
copyto!(numOut, gMemSlfN.trgVec);
@time greenAna(anaOut, gMemSlfN.srcVol, trgRng, srcPos, srcVec);
@time greenAsm(asmOut, gMemSlfN.srcVol, trgRng, srcPos, srcVec);
# Reset source vector for further tests
gMemSlfN.srcVec[1,1,1,3] = 0.0 + 0.0im; 