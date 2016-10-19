function fit!{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        updater::ParamUpdater = Adam();
        learning_rate::Float64 = 0.1,
        iterations::Int = 1_000_000,
        show_trace::Bool = true,
        show_every::Int = 10_000
    )

    x = params(model)
    init(updater, model)

    f_hist = zeros(iterations)
    for iter = 1:iterations
        
        f_x, ∇ = sgrad(model, data)
        update!(x, updater, ∇, learning_rate)
        f_hist[iter] = f_x

        if show_trace
            mod(iter,show_every)==0 && print_with_color(:blue,"Iteration $iter, loss  =  $(round(f_x,8))\n")
        end

    end

    return f_hist
end
