using GeneralizedCPD
using Base.Test
using Calculus
using Iterators

@testset "Gradient Calculations" begin
    # data parameters
    nr = 2       # rank
    sz = (4,5,6) # dimensions

    losses = [L2DistLoss(), L1DistLoss(), LogitMarginLoss(), HingeLoss(), PoissonLoss()]

    for loss in losses
        # create model
        data, = GeneralizedCPD.fakedata(sz,nr,loss)
        model = GenCPD(data, nr, loss)
        randn!(model)
        fill!(model.cpd.λ,1.0)

        # objective function, autodiff gradients
        f(x) = sumvalue!(model,x,data);
        ∇a = Calculus.gradient(f,getparams(model))
        ∇b = grad(model,data)
        @test isapprox(∇a,∇b)

        # stochastic gradient 
        ∇c = zeros(nparams(model))
        for idx in product(1:sz[1],1:sz[2],1:sz[3])
            val,∇s = sgrad(model, data, idx...)
            ∇c += ∇s
            @test isapprox(val,value(model,data,idx...))
        end
        @test isapprox(∇a,∇c)
        @test isapprox(∇b,∇c)
    end
end
