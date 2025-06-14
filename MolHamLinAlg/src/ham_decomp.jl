using LinearAlgebra
using PyCall
using Base: collect_preferences, permutecols!!
using Tullio: @einsum
using Test

export lr_decomposition
export rowwise_reshape
export vecs2mat_reshape
export reshape_eigs
export jl_print
export check_lr_decomp

function lr_decomposition(tbt)
    n = size(tbt)[1]
    flattened_tbt = rowwise_reshape(tbt, n^2)
    diags, Ls = eigen(Hermitian(flattened_tbt))
    Vs = vecs2mat_reshape(Ls, n)
    lr_fragments = []
    for i in 1:n^2
        if norm(sqrt(abs(diags[i])) .* Vs[i]) > 1e-6
            d, U = eigen(Hermitian(Vs[i]))
            d = reshape_eigs(d)
            C = diags[i] .* d * transpose(d)
            @einsum A[p,q,r,s] := C[i,j] * U[p,i] * U[q,i] * U[r,j] * U[s,j]
            push!(lr_fragments, ((diags[i], d), U, A))
        end
    end
    check_lr_decomp(tbt, map(t -> t[3], lr_fragments))
    lr_fragments
end

function check_lr_decomp(tbt, lr_tensors)
    isapprox(tbt, foldl(.+, lr_tensors)) || error("LR Decomposition Failed")
end

function rowwise_reshape(mat, n)
    reshape(permutedims(mat, (2, 1, 4, 3)), (n, n))
end

function vecs2mat_reshape(Ls, n)
    reshaped_vecs = []
    for col in eachcol(Ls)
        push!(reshaped_vecs, transpose(reshape(col, (n, n))))
    end
    reshaped_vecs
end

function reshape_eigs(d)
    reshape(d, (size(d)[1], 1))
end


function jl_print(m)
    display(m)
end
