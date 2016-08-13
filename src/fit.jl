## ---- Use Optim.jl to fit Generalized CPD ---- ##

function fit!{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        mo::Optim.Optimizer;
        kwargs...
    )
    
    o = OptimizationOptions(;kwargs...)

    # prevent initialization at origin (it is a saddle point)
    for fctr in model.cpd.factors
        if vecnorm(fctr) < eps()
            randn!(fctr)
            scale!(fctr,T(0.1))
        end
    end

    # generate functions for Optim
    f(x) = sumvalue!(model,x,data) # updates params before computing objective
    g!(x) = grad!(∇,model,x,data)  # updates params before computing gradient
    function fg!(x,∇)
        # only updates parameters once
        f = sumvalue!(model,x,data)
        grad!(∇,model,data)
        return f
    end
    dfunc = DifferentiableFunction(f,g!,fg!)

    # call Optim
    result = optimize(dfunc,getparams(model),mo,o)
    
    # update parameters, return optimization results
    setparams!(model,Optim.minimizer(result))
    return result
end


## ---- Custom Optimizers ---- ##

# An optimizer that only knows how to fit GenCPD
abstract GenCPDOptimizer <: Optim.Optimizer

## ---- Alternating Gradient Descent ---- ##

immutable AltGradDescent <: GenCPDOptimizer
    linesearch!::Function
end
AltGradDescent(
    ;linesearch!::Function = Optim.hz_linesearch!
    ) = AltGradDescent(linesearch!)

function fit!{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        mo::AltGradDescent,
        o::OptimizationOptions = OptimizationOptions()
    )

    # preallocate storage for gradients
    # ∇ = 

    return nothing
end