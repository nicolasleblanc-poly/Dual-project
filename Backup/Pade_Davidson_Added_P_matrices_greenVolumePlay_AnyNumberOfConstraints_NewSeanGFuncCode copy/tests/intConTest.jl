const cubRelTol = 1e-8;
const cubAbsTol = 1e-12; 

function weakIntCheck(scl::Float64, glOrd::Int64)::Array{ComplexF64,1}

	sclV = (scl,scl,scl)
	opts = MaxGAssemblyOpts()

	ws1 = weakS(sclV, gaussQuad1(glOrd), opts)
	we1 = weakE(sclV, gaussQuad1(glOrd), opts)
	wv1 = weakV(sclV, gaussQuad1(glOrd), opts)

	ws2 = weakS(sclV, gaussQuad1(glOrd + 1), opts)
	we2 = weakE(sclV, gaussQuad1(glOrd + 1), opts)
	wv2 = weakV(sclV, gaussQuad1(glOrd + 1), opts)

	intDiffS = maximum(abs.(ws1 .- ws2) ./ abs.(ws2))
	intDiffE = maximum(abs.(we1 .- we2) ./ abs.(we2))
	intDiffV = maximum(abs.(wv1 .- wv2) ./ abs.(wv2))

	return [intDiffS, intDiffE, intDiffV]
end

intDiff = weakIntCheck(scaleA[1], 48)

if maximum(abs.(intDiff)) < cubRelTol

	println("Integration convergence test passed.")

else

	println("Integration convergence test failed!")	
end