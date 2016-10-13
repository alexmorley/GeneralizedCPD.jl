using GeneralizedCPD
using Plots; gr()

# data parameters
function demo_gen_cpd(loss = L2DistLoss(), sz = (50,60,70), nr = 2; noise=0.01, kwargs...)

    # model and data
    data, ground_truth = GeneralizedCPD.fakedata(sz, nr, loss, noise)
    model = GenCPD(data, nr, loss)

    # fit
    converged,tr = fit!(model, data, AlternatingDescent(); kwargs...)

    # plot cpd
end

demo_gen_cpd(L2DistLoss())
