using GeneralizedCPD
using Base.Test

cpd = CPDecomp(3,(9,10,11));

# should run without error
l0 = LossArray(L2DistLoss())
l1 = LossArray(fill(L2DistLoss(),10))
l2 = LossArray(fill(L2DistLoss(),10,10))

@test typeof(l0) <: LossArray{L2DistLoss,0}
@test typeof(l1) <: LossArray{L2DistLoss,1}
@test typeof(l2) <: LossArray{L2DistLoss,2}

# 
@test l2[10,10,100] == L2DistLoss()
@test l2[10,10,100] == L2DistLoss()
@test l2[10,10,100] == L2DistLoss()


g0 = GenCPDecomp(cpd,L2DistLoss());
g1 = GenCPDecomp(cpd,fill(L2DistLoss(),9));
g2 = GenCPDecomp(cpd,fill(L2DistLoss(),9,10));
g3 = GenCPDecomp(cpd,fill(L2DistLoss(),9,10,11));

@test typeof(g0) <: GenCPDecomp{Float64,3,L2DistLoss,0}
@test typeof(g1) <: GenCPDecomp{Float64,3,L2DistLoss,1}
@test typeof(g2) <: GenCPDecomp{Float64,3,L2DistLoss,2}
@test typeof(g3) <: GenCPDecomp{Float64,3,L2DistLoss,3}

@test_throws Exception GenCPDecomp(cpd,fill(L2DistLoss(),8));
@test_throws Exception GenCPDecomp(cpd,fill(L2DistLoss(),8,11));
@test_throws Exception GenCPDecomp(cpd,fill(L2DistLoss(),9,10,11,5));
