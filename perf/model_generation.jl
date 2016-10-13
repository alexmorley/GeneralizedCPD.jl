using BenchmarkTools
using GeneralizedCPD

# data parameters
nr = 5       # rank
sz = (50,60,70) # dimensions

# create low-rank data
ξ = 0.1*randn(sz) # noise
data = full(cpd_randn(sz,nr)) + ξ

@benchmark GenCPD($data, $nr, L1DistLoss())
