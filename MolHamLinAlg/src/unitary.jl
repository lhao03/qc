using LinearAlgebra

export extract_thetas
export make_x_matrix
export make_unitary

function isantisymm(X)
    isapprox(X, -1 .* transpose(X))
end

function extract_thetas(U)
    U[abs.(U) .< eps(eltype(U))] .= zero(eltype(U))
    isapprox(det(U), 1) || error("U must have determinant of 1, got: $(det(U))")
    if isapprox(U, U', rtol=1e-10)
        # || error("U should be Hermitian: $(repr("text/plain", U))")
        U = Hermitian(U)
    end
    X = log(U)
    (isapprox(X, transpose(X)) || isantisymm(X)) || error("X is not symmetric or skew-symmetric, can't extract thetas.")
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

function make_x_matrix(thetas, diags, n)
    X = Complex.(zeros((n, n)))
    t = 1
    rtol = eps(real(float(thetas[1])))
    for x in 1:n
        for y in x+1:n
            v_real = real(thetas[t])
            v_imag = imag(thetas[t])
            if isapprox(v_real, 0) || v_real <= rtol
                X[x, y] = v_imag * im
                X[y, x] = v_imag * im
            else
                X[x, y] = -v_real + (v_imag * im)
                X[y, x] = v_real + (v_imag * im)
            end
            t += 1
        end
    end
    for i in 1:n
        X[i, i] = diags[i]
    end
    X
end

function make_x_matrix(thetas, n)
    X = zeros((n, n))
    t = 1
    for x in 1:n
        for y in x+1:n
            X[x, y] = -thetas[t]
            X[y, x] = thetas[t]
            t += 1
        end
    end
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

