using LinearAlgebra
using PyCall
using Base: collect_preferences
using Tullio: @einsum

export lr_decomposition
export rowwise_reshape_vec_to_mat
export rowwise_reshape_four_rank_to_two_rank

function lr_decomposition(tbt)
    n = size(tbt)[1]
    flattened_tbt =  rowwise_reshape_four_rank_to_two_rank(tbt)
    diags, Ls = eigen(flattened_tbt)
    Vs = map(L -> rowwise_reshape_vec_to_mat(L, n), eachcol(Ls))
    lr_fragments = []
    for i in 1:n^2
        if norm(sqrt(abs(diags[i])) .* Vs[i]) > 1e-6
            d, U = eigen(Vs[i])
            d = reshape(d, (size(d)[1], 1))
            C = diags[i] .* d * transpose(d)
            @einsum A[p, q, r,s ] := C[i, j] * U[p, i] * U[q, i] * U[r, j] * U[s, j]
            push!(lr_fragments, (C, U, A))
        end
    end
    lr_fragments
end

function rowwise_reshape_four_rank_to_two_rank(mat)
    n = size(mat)[1]
    row_flattened_mat = []
    for i in 1:n
        for j in 1:n
            row_slice = mat[i, j, :, :]
            rows = eachrow(row_slice)
            row_vec = transpose(collect(foldl(vcat, rows)))
            push!(row_flattened_mat, row_vec)
        end
    end
    vcat(row_flattened_mat...)
end

function rowwise_reshape_vec_to_mat(v, n)::Matrix{Complex}
    e = 1
    rows = []
    for _ in 1:n
        row = []
        for _ in 1:n
            push!(row, v[e])
            e += 1
        end
        push!(rows, row)
    end
    stack(rows;dims=1)
end
