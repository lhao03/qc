using LinearAlgebra
using Tullio: @einsum

A = [0 0 0 0;
     0 0 0 0;
     0 0 0 1;
     0 0 1 0]
val, vec = eigen(A)

B = zeros(4,4)
for i in 1:4
    B .+= val[i] .* vec[:,i] * transpose(vec[:,i])
end
display(B)

@einsum C[p,q] := val[i] * vec[i, p] * vec[i, q]
display(C)
