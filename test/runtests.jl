using GeneralizedCPD
using Base.Test
using ForwardDiff

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

    losses = [L2DistLoss(), L1DistLoss()]
    for loss in losses
        # create low-rank data
        ξ = 0.1*randn(sz) # noise
        data = full(cpd_randn(sz,nr)) + ξ
        model = GenCPD(nr,data,L1DistLoss())
        fill!(model.cpd.λ,1.0)

        # init model, define objective
        randn!(model)
        f(x) = sumvalue(model,x,data);
        
        # check gradients vs ForwardDiff
        ∇a = ForwardDiff.gradient(f,getparams(model))
        ∇b = grad(model,data)
        @test isapprox(∇a,∇b)

        # similar checks
        xinit = copy(getparams(model))
        x = randn(nparams(model))
        ∇a = ForwardDiff.gradient(f,x)
        ∇b = grad(model,x,data)
        @test isapprox(∇a,∇b)
        @test isapprox(xinit,getparams(model))
        
        # check that grad! overwrites
        ∇c = grad!(model,x,data)
        @test isapprox(∇a,∇c)
        @test isapprox(x,getparams(model))
    end
end

