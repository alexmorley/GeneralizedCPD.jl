using GeneralizedCPD

# data parameters
nr = 2          # rank
sz = (15,16,17) # dimensions

# create low-rank data
ξ = 0.1*randn(sz) # noise
const data = cpd_randn(sz,nr) + ξ

# model
const model = GenCPD(data, nr, L2DistLoss())
result = fit!(model, data, LBFGS(); store_trace=true)

# plot trace
using Plots; gr()
tr = Optim.trace(result)
iter = [ t.iteration for t in tr.states ]
obj = [ t.value for t in tr.states ]
∇nrm = [ t.gradnorm for t in tr.states ]

plot(
    plot(iter,obj,xaxis=("iteration"),yaxis=("objective",:log)),
    plot(iter,∇nrm,xaxis=("iteration"),yaxis=("norm of gradient",:log)),
    legend=(nothing)
)
