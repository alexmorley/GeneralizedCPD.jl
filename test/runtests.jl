using GeneralizedCPD
using Base.Test
using Calculus
import StatsFuns: logistic

# @testset "Basic CPD constructors" begin
#     cpd = CPD(3,(9,10,11));

#     g0 = GenCPD(cpd,L2DistLoss());
#     g1 = GenCPD(cpd,fill(L2DistLoss(),9));
#     g2 = GenCPD(cpd,fill(L2DistLoss(),9,10));
#     g3 = GenCPD(cpd,fill(L2DistLoss(),9,10,11));

#     @test typeof(g0) <: GenCPD{Float64,3,L2DistLoss,0}
#     @test typeof(g1) <: GenCPD{Float64,3,L2DistLoss,1}
#     @test typeof(g2) <: GenCPD{Float64,3,L2DistLoss,2}
#     @test typeof(g3) <: GenCPD{Float64,3,L2DistLoss,3}

#     @test_throws Exception GenCPD(cpd,fill(L2DistLoss(),8));
#     @test_throws Exception GenCPD(cpd,fill(L2DistLoss(),8,11));
#     @test_throws Exception GenCPD(cpd,fill(L2DistLoss(),9,10,11,5));
# end

@testset "Gradient Calculations" begin
    # data parameters
    nr = 2       # rank
    sz = (4,5,6) # dimensions

    losses = [L2DistLoss(), L1DistLoss(), LogitMarginLoss(), HingeLoss(), PoissonLoss()]

    for loss in losses
        # create model
        data = GeneralizedCPD.fakedata(sz,nr,loss)
        model = GenCPD(data, nr, loss)
        randn!(model)
        
        # objective function, autodiff gradients
        f(x) = sumvalue!(model,x,data);
        ∇a = Calculus.gradient(f,getparams(model))
        ∇b = grad(model,data)
        @test isapprox(∇a,∇b)
    end
end

