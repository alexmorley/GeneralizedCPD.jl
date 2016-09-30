## ---- Use Optim.jl to fit Generalized CPD ---- ##

function fit!{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        mo::Optim.Optimizer,
        o::OptimizationOptions = OptimizationOptions()
    )

    # prevent initialization at origin (it is a saddle point)
    for fctr in model.cpd.factors
        if vecnorm(fctr) < eps()
            randn!(fctr)
            # scale!(fctr,T(0.1))
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
abstract GenCPDOptimizer

## ---- Alternating Gradient Descent ---- ##

immutable AltGradDescent <: GenCPDOptimizer
    linesearch!::Function
end
AltGradDescent(
    ;linesearch!::Function = Optim.hz_linesearch!
    ) = AltGradDescent(linesearch!)

@generated function fit!{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        mo::AltGradDescent,
        o::OptimizationOptions = OptimizationOptions()
    )

  quote 
    # preallocate storage for gradients
    ∇storage = Array(T, maximum(size(model))*rank(model))

    # views into paramvec and gradient for each factor
    const x = @ntuple $N n->(Array(T, size(model,n)*rank(model)))
    const ∇ = @ntuple $N n->(view(∇storage, 1:(rank(model)*size(model,n))))

    f = @ntuple $N n->((x)->begin
        setparams!(model, x, n)
        return sumvalue(model, data)
    end
    )
    g! = @ntuple $N n->((x,∇st)->begin
        setparams!(model, x, n)
        grad!(∇st, model, data, n)
    end
    )
    fg! = @ntuple $N n->((x,∇st)->begin
        setparams!(model, x, n)
        f = sumvalue(model, data)
        grad!(∇st, model, data, n)
        return f
    end
    )
    dfunc = @ntuple $N n->DifferentiableFunction(f[n],g![n],fg![n])

    # gd = @ntuple $N n->GradientDescent()
    # s = @ntuple $N n->Optim.initial_state(gd[n], o, dfunc[n], x[n])

    tr = Float64[]
    for iter = 1:10
        @show iter
        for n = 1:($N)
            @show n
            # Optim.update_state!(dfunc[n], s[n], gd[n]) && break
            # Optim.update_g!(dfunc[n], s[n], gd[n]) 
            # copy!(x[n], s[n].x)

            result = optimize(dfunc[n], x[n], GradientDescent(), o)
            copy!(x[n], result.minimum)
            
            # TODO - check if necessary
            setparams!(model, x[n], n)
        end
        push!(tr,sumvalue(model,data))
    end

    return tr
  end # quote

end
