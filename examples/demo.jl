using GeneralizedCPD

# data parameters
nr = 2          # rank
sz = (50,60,70) # dimensions

# create low-rank data
ξ = 1e-6*randn(sz) # noise
const data = cpd_randn(sz,nr) + ξ

# model
const model = GenCPD(data, nr, L2DistLoss())
randn!(model)
result = fit!(model, data, LBFGS(); store_trace=true)

# plot trace
using Plots; gr()
tr = Optim.trace(result)
iter = [ t.iteration for t in tr ]
obj = [ t.value for t in tr ]
∇nrm = [ t.g_norm for t in tr ]

plot(
    plot(iter,obj,xaxis=("iteration"),yaxis=("objective",:log)),
    plot(iter,∇nrm,xaxis=("iteration"),yaxis=("norm of gradient",:log)),
    legend=(nothing)
)
