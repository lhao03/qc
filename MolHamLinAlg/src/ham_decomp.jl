using LinearAlgebra
using PyCall
using Base: collect_preferences, permutecols!!
using Tullio: @einsum
using Test

export lr_decomposition
export rowwise_reshape
export vecs2mat_reshape
export reshape_eigs

function lr_decomposition(tbt)
    n = size(tbt)[1]
    flattened_tbt = rowwise_reshape(tbt, n^2)
    diags, Ls = eigen(flattened_tbt)
    Vs = vecs2mat_reshape(Ls, n)
    lr_fragments = []
    lr_tensors = []
    for i in 1:n^2
        if norm(sqrt(abs(diags[i])) .* Vs[i]) > 1e-6
            display(Vs[i])
            d, U = eigen(Vs[i])
            d = reshape_eigs(d)
            C = diags[i] .* d * transpose(d)
            @einsum A[p, q, r,s] := C[i, j] * U[p, i] * U[q, i] * U[r, j] * U[s, j]
            # display(C)
            display(U)
            # for i in 1:n
            #     for j in 1:n
            #         println("$(i), $(j)")
            #         display(A[i, j, :, :])
            #     end
            # end
            push!(lr_fragments, (C, U, A))
            push!(lr_tensors, A)
        end
    end
    # isapprox(tbt, foldl(.+, lr_tensors)) || error("LR Decomposition Failed")
    lr_fragments
end

function rowwise_reshape(mat, n)
    reshape(permutedims(mat, (2, 1, 4, 3)), (n, n))
end

function vecs2mat_reshape(Ls, n)
    reshaped_vecs = []
    for col in eachcol(Ls)
        col = map((e) -> abs(e) > 1e-10 ? e : 0, col)
        push!(reshaped_vecs, transpose(reshape(col, (n, n))))
    end
    reshaped_vecs
end

function reshape_eigs(d)
    reshape(d, (size(d)[1], 1))
end

