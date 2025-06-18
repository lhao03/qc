using LinearAlgebra
using Tullio: @einsum

A = [0 0 0 0;
     0 0 0 0;
     0 0 0 1;
     0 0 1 0]
val, vec = eigen(A)
display(val)
display(vec)

@einsum D[p, q] := val[r] * vec[p, r] * vec[r, q]
display(D)
