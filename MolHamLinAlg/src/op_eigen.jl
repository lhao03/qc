using LinearAlgebra

export extract_eigen
export UV_eigendecomp

function extract_eigen(op::Matrix, ev::Vector, panic)
    ev = map((e) -> abs(e) > 1e-7 ? e : 0, ev)
    bv = op * ev
    b = bv ./ ev
    b_filtered = filter(!isnan, b)
    eig = b_filtered[1]
    if panic
        all(isapprox.(b_filtered, eig; atol=1e-10)) || error("""Expected all elements of eigenvalue vector to be the same, but got: $(b_filtered).
                                                                Original vector was: $(ev).
                                                                After applying operator become $(bv).
                                                                After division got $(b).""")
        isapprox(eig * ev, bv; atol=1e-8) || error("O v != b v, got: $(bv) and $(eig * ev).")
    end
    real_e = real(eig)
    img_e = imag(eig)
    isapprox(img_e, 0, atol=1e-10) || error("Eigenvalues should be real, got $(eig)")
    round(real_e)
end

function UV_eigendecomp(mat)
    vals, vecs = eigen(mat)
    vals, vecs
end
