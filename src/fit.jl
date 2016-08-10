function fit!{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        mo::Optim.Optimizer,
        o::OptimizationOptions = OptimizationOptions()
    )

    # generate functions for fitting
    f(x) = sumvalue!(model,x,data)
    g!(x) = grad!(∇,model,x,data) 
    function fg!(x,∇)
        f = sumvalue!(model,x,data)
        grad!(∇,model,data)
        return f
    end
    dfunc = DifferentiableFunction(f,g!,fg!)

    # call Optim
    result = optimize(dfunc,getparams(model),mo,o)
    
    # update parameters, return optimization results
    setparams!(model,result.x)
    return result
end
