using GeneralizedCPD
using Plots; gr()

# data parameters
function demo_gen_cpd(loss = L2DistLoss(), sz = (50,60,70), nr = 2; noise=0.01, kwargs...)

    # model and data
    data, ground_truth = GeneralizedCPD.fakedata(sz, nr, loss, noise)
    model = GenCPD(data, nr, loss)
    init_model = deepcopy(model)

    # fit
    f_hist = fit!(model, data, Adam(); kwargs...)

    return init_model, model, f_hist
end

init_model, model, f_hist = demo_gen_cpd(iterations=10000)
