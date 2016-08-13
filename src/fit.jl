## ---- Use Optim.jl to fit Generalized CPD ---- ##

function fit!{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        mo::Optim.Optimizer;
        kwargs...
    )
    
    o = default_opt_options(;kwargs...)

    # # prevent initialization at origin (it is a saddle point)
    # for fctr in model.cpd.factors
    #     if vecnorm(fctr) < eps()
    #         randn!(fctr)
    #         # scale!(fctr,T(0.1))
    #     end
    # end

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

# @generated function fit!{T,N}(
#         model::GenCPD{T,N},
#         data::AbstractArray{T,N},
#         mo::AltGradDescent,
#         o::OptimizationOptions = OptimizationOptions()
#     )

#   quote 
#     # preallocate storage for gradients
#     ∇storage = Array(T,maximum(size(model)), rank(model))

#     # views into paramvec and gradient for each factor
#     const x = @ntuple $N n->(factor_view(model,n))
#     const ∇ = @ntuple $N n->(view(∇,1:size(model,n),:))

#     # 
#     f(x) = sumvalue!(model,x[n],data)
#     g!(x) = grad!(∇[n],model,x[n],data)
#     function fg!(x,∇st)
#         # only updates parameters once
#         setparams!(x[n], model, n)
#         f = sumvalue(model, data)
#         grad!(∇st, model, data)
#         return f
#     end
#     dfunc = DifferentiableFunction(f,g!,fg!)


#     for i = 1:o.iterations
#         for n = 1:N
            
#             # vector view in factor n
#             xn = factor_view(model, n)

#             # calc gradient for factor n
#             grad!(∇[n], model, data, n)

#             # do linesearch
#             mo.linesearch!()
#         end
#     end

#     return nothing
#   end # quote

# end
