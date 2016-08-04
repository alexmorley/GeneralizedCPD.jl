using Einsum
using GeneralizedCPD

# data parameters
nr = 2          # rank
sz = (15,16,17) # dimensions
N  = length(sz)

# create low-rank data
F = [ randn(nr,sz[n]) for n in 1:N ]
@einsum X[i,j,k] = F[1][r,i]*F[2][r,i]*F[3][r,i]

# add noise
ξ = 0.1*randn(sz)
data = X + ξ

# fit model
model = GenCPD(nr,data,L2DistLoss())
result = fit!(model,data)
