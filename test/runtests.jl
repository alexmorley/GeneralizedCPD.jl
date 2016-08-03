using GeneralizedCPD
using Base.Test

cpd = CPDecomp(3,(9,10,11));

# should run without error
GenCPDecomp(cpd,L2DistLoss());
GenCPDecomp(cpd,fill(L2DistLoss(),9));
GenCPDecomp(cpd,fill(L2DistLoss(),9,10));
GenCPDecomp(cpd,fill(L2DistLoss(),9,10,11));

@test_throws Exception GenCPDecomp(cpd,fill(L2DistLoss(),8));
@test_throws Exception GenCPDecomp(cpd,fill(L2DistLoss(),8,11));
@test_throws Exception GenCPDecomp(cpd,fill(L2DistLoss(),9,10,11,5));

