using LinearAlgebra

export extract_thetas
export make_x_matrix
export make_unitary

function isantisymm(X)
    isapprox(X, -1 .* transpose(X))
end

function extract_thetas(U)
    U[abs.(U) .< eps(real(eltype(U)))] .= zero(eltype(U))
    isapprox(det(U), 1) || error("U must have determinant of 1, got: $(det(U))")
    if isapprox(U, U', rtol=1e-10)
        U = Hermitian(U)
    end
    X = log(U)
    isapprox(X, transpose(X)) || isantisymm(X) || error("X is not symmetric or skew-symmetric, can't extract thetas: $(repr("text/plain", X))")
    n = size(U)[1]
    m = Integer(((n * (n + 1)) / 2) - n)
    thetas = Complex.(zeros(m))
    c = 1
    for i in 1:n-1
        for j in i+1:n
            thetas[c] = X[j, i]
            c += 1
        end
    end
    thetas, diag(X)
end

function re_im_close(n, tol=1e-10)
    n_real = real(n)
    n_im = imag(n)
    re_is_zero = isapprox(n_real, 0, atol=tol)
    im_is_zero = isapprox(n_im, 0, atol=tol)
    if re_is_zero && im_is_zero
        0
    elseif re_is_zero
        n_im * im
    elseif im_is_zero
        n_real
    else
        n
    end
end

function make_x_matrix(thetas, diags, n)
    X = Complex.(zeros((n, n)))
    t = 1
    for x in 1:n
        for y in x+1:n
            v = re_im_close(thetas[t])
            X[x, y] = -real(v) + imag(v) * im
            X[y, x] = v
            t += 1
        end
    end
    for i in 1:n
        X[i, i] = re_im_close(diags[i])
    end
    all(x -> isapprox(real(x), 0), diag(X)) || error("Real values were found on the diagonal of X: $(repr("text/plain", X))")
    m = (n * (n + 1)) / 2 - n
    t >= m || error("X matrix did not use all the theta values: t, which was $(t), was expected to be $(m).")
    X
end

function make_x_matrix(thetas, n)
    X = zeros((n, n))
    t = 1
    for x in 1:n
        for y in x+1:n
            v = re_im_close(thetas[t])
            X[x, y] = -real(v) + imag(v) * im
            X[y, x] = v
            t += 1
        end
    end
    all(x -> isapprox(imag(x), 0), X) || error("Imaginary values were found in X, please use `make_x_matrix(t, d, n)`")
    m = (n * (n + 1)) / 2 - n
    t >= m || error("X matrix did not use all the theta values: t, which was $(t), was expected to be $(m).")
    X
end

function make_unitary(thetas, diags, n)
    X = make_x_matrix(thetas, diags, n)
    U = exp(X)
    U[abs.(U) .< eps(real(float(eltype(U))))] .= zero(eltype(U))
    isapprox(det(U), 1) || error("U must have determinant of 1, got: $(det(U))")
    U
end

function make_unitary(thetas, n)
    X = make_x_matrix(thetas, n)
    U = exp(X)
    U[abs.(U) .< eps(real(float(eltype(U))))] .= zero(eltype(U))
    isapprox(det(U), 1) || error("U must have determinant of 1, got: $(det(U))")
    U
end

