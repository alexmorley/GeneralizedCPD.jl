using Calculus
using GeneralizedCPD

let
    # data parameters
    nr = 2       # rank
    sz = (4,5,6) # dimensions

    # create low-rank data
    ξ = 0.1*randn(sz) # noise
    const data = full(cpd_randn(sz,nr)) + ξ
    const model = GenCPD(data, nr, L1DistLoss())

    # objective function, autodiff gradients
    randn!(model)
    f(x) = sumvalue!(model,x,data);
    ∇a = Calculus.gradient(f,getparams(model))
    ∇b = grad(model,data)

    if isapprox(∇a,∇b)
        info("Gradients match")
    else
        warn("Gradients don't match")
    end
end