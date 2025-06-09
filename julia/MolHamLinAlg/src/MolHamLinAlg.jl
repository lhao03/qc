module MolHamLinAlg

function extract_eigen(op::Matrix, ev::Vector)
    bv = op * ev
    b = bv ./ ev
    b = filter(!isnan, b)
    all(b[1] .== b) || throw(AssertionError("Expected all elements of eigenvalue vector to be the same."))
    b[1] * ev == bv || throw(AssertionError("O v != b v, we expect these to be equal"))
    b[1]
end

end # module MolHamLinAlg
