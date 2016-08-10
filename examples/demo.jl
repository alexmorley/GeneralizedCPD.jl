using GeneralizedCPD

# data parameters
nr = 2          # rank
sz = (15,16,17) # dimensions

# create low-rank data
ξ = 0.1*randn(sz) # noise
const data = cpd_randn(sz,nr) + ξ

# model
const model = GenCPD(nr,data,L2DistLoss())

# objective function, autodiff gradients
using ForwardDiff
f(x) = sumvalue(model,x,data)
g!(x,∇) = ForwardDiff.gradient!(∇,f,x)

# use Optim to fit
using Optim
x0 = randn(nparams(model))
result = optimize(f,g!,x0;store_trace=true)

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
