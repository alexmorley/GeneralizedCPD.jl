using Einsum
using GeneralizedCPD

# data parameters
nr = 2          # rank
sz = (15,16,17) # dimensions
N  = length(sz)

# create low-rank data
F = [ randn(sz[n],nr) for n in 1:N ]
@einsum X[i,j,k] := F[1][i,r]*F[2][j,r]*F[3][k,r]

# add noise
ξ = zeros(sz)#0.1*randn(sz)
data = X + ξ
#data = zeros(sz)

# fit model
model = GenCPD(nr,data,L2DistLoss())
# for i = 1:3; copy!(model.cpd.factors[i],F[i]); end
# fill!(model.cpd.λ,1.0)

opt = GenCPDParams(30)
result = fit!(model,data,opt)

using Plots; unicodeplots();
plot(result.trace; xaxis=("iterations"), yaxis=("error"))
