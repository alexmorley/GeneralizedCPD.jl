using BenchmarkTools
using GeneralizedCPD

# data parameters
nr = 5       # rank
sz = (100,100,100) # dimensions

# create low-rank data
ξ = 0.1*randn(sz) # noise
data = full(cpd_randn(sz,nr)) + ξ

# create model
model = GenCPD(data, nr, L1DistLoss())
randn!(model)

# benchmark without preallocation
b1 = @benchmark grad($model, $data)

# benchmark with preallocation
∇ = similar(grad(model,data))
b2 = @benchmark grad!($∇, $model, $data)

# benchmark with vector preallocation
∇2 = vec(∇)
b3 = @benchmark grad!($∇2, $model, $data)

# benchmark without preallocation
b4 = @benchmark grad($model, $data, 1)

# benchmark with preallocation
∇3 = similar(grad(model,data,1))
b5 = @benchmark grad!($∇3, $model, $data, 1)

# benchmark with vector preallocation
∇4 = vec(∇3)
b6 = @benchmark grad!($∇4, $model, $data, 1)
