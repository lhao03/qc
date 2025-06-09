module MolHamLinAlg

function extract_eigen(op::Matrix, ev::Vector, panic::Bool)::Real
    ev = map((e) -> abs(e) > 1e-10 ? e : 0, ev)
    bv = op * ev
    b = bv ./ ev
    b = filter(!isnan, b)
    eig = b[1]
    if panic
        all(eig .== b) || throw(AssertionError("Expected all elements of eigenvalue vector to be the same, but got: $(b)"))
        eig * ev == bv || throw(AssertionError("O v != b v, got: $(bv) and $(eig * ev)."))
    end
    real_e = real(eig)
    img_e = imag(eig)
    img_e == 0 || throw(AssertionError("Eigenvalues should be real, got $(eig)"))
    real_e
end

end # module MolHamLinAlg
