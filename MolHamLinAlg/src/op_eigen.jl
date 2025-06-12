using LinearAlgebra

export extract_eigen
export UV_eigendecomp

function extract_eigen(op::Matrix, ev::Vector, panic)
    ev = map((e) -> abs(e) > 1e-10 ? e : 0, ev)
    bv = op * ev
    b = bv ./ ev
    b = filter(!isnan, b)
    eig = b[1]
    if panic
        all(isapprox.(b, eig; atol=1e-10)) || error("Expected all elements of eigenvalue vector to be the same, but got: $(b). Original vector was: $(ev)")
        isapprox(eig * ev, bv; atol=1e-8) || error("O v != b v, got: $(bv) and $(eig * ev).")
    end
    real_e = real(eig)
    img_e = imag(eig)
    isapprox(img_e, 0, atol=1e-10) || error("Eigenvalues should be real, got $(eig)")
    real_e
end

function UV_eigendecomp(mat)
    vals, vecs = eigen(mat)
    vecs, vals
end
