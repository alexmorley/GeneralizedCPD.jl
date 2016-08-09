using ForwardDiff
using GeneralizedCPD

# data parameters
nr = 2       # rank
sz = (4,5,6) # dimensions

# create low-rank data
ξ = 0.1*randn(sz) # noise
const data = full(cpd_randn(sz,nr)) + ξ
const model = GenCPD(nr,data,L1DistLoss())

# objective function, autodiff gradients
randn!(model)
f(x) =  sumvalue(model,x,data);
∇a = ForwardDiff.gradient(f,getparams(model))[1:(end-2)]
∇b = grad(model,data)

