using GeneralizedCPD
using Base.Test

cpd = CPD(3,(9,10,11));

g0 = GenCPD(cpd,L2DistLoss());
g1 = GenCPD(cpd,fill(L2DistLoss(),9));
g2 = GenCPD(cpd,fill(L2DistLoss(),9,10));
g3 = GenCPD(cpd,fill(L2DistLoss(),9,10,11));

@test typeof(g0) <: GenCPD{Float64,3,L2DistLoss,0}
@test typeof(g1) <: GenCPD{Float64,3,L2DistLoss,1}
@test typeof(g2) <: GenCPD{Float64,3,L2DistLoss,2}
@test typeof(g3) <: GenCPD{Float64,3,L2DistLoss,3}

@test_throws Exception GenCPD(cpd,fill(L2DistLoss(),8));
@test_throws Exception GenCPD(cpd,fill(L2DistLoss(),8,11));
@test_throws Exception GenCPD(cpd,fill(L2DistLoss(),9,10,11,5));

