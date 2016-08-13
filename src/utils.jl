function fakedata{N,Ix<:Integer}(
        sz::NTuple{N,Ix},
        nr::Integer,
        loss::SupervisedLoss
    )

    # create random low-rank tensor
    cpd = cpd_randn(sz,nr)

    # add appropriate noise
    error("not done")
end


function default_opt_options(;
        x_tol::Real = 1e-32,
        f_tol::Real = 1e-10,
        g_tol::Real = 1e-8,
        iterations::Integer = 1_000,
        store_trace::Bool = false,
        show_trace::Bool = true,
        extended_trace::Bool = false,
        autodiff::Bool = false,
        show_every::Integer = 5,
        callback = nothing)

    OptimizationOptions{typeof(callback)}(
        Float64(x_tol), Float64(f_tol), Float64(g_tol), Int(iterations),
        store_trace, show_trace, extended_trace, autodiff, Int(show_every),
        callback)
end
