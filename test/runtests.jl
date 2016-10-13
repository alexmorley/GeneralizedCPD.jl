using GeneralizedCPD
using Base.Test
using Calculus

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
        
        # objective function, autodiff gradients
        f(x) = sumvalue!(model,x,data);
        ∇a = Calculus.gradient(f,getparams(model))
        ∇b = grad(model,data)
        @test isapprox(∇a,∇b)
    end
end

