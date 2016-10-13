## Default to AlternatingDescent method
fit!(model::GenCPD, data::AbstractArray) = fit!(model, data, AlternatingDescent())

## ---- Direct backend to Optim.jl ---- ##
function fit!{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        mo::Optim.Optimizer,
        o::OptimizationOptions = OptimizationOptions()
    )

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

## ---- Alternating Descent, using Optim.jl ---- ##
const DEFAULT_ALT_OPTIONS = OptimizationOptions(iterations=10)

immutable AlternatingDescent{O <: Optim.Optimizer}
    optimizer::O
    options::OptimizationOptions
end
AlternatingDescent(o::OptimizationOptions=DEFAULT_ALT_OPTIONS) = AlternatingDescent{LBFGS}(LBFGS(),o)
AlternatingDescent(mo::Optim.Optimizer,o::OptimizationOptions=DEFAULT_ALT_OPTIONS) = AlternatingDescent(mo,o)

function fit!{T,N,O}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        meta::AlternatingDescent{O},
        options::OptimizationOptions = OptimizationOptions(f_tol=1e-5)
    )
    
    dfunc = _decompose_objective(model, data)

    if options.store_trace
        tr = OptimizationState{O}[]
    end

    f_x = sumvalue(model,data) # holds objective
    converged = false

    for iter = 1:options.iterations
        xprev = copy(getparams(model))
        f_x_prev = f_x
        @show f_x
        for n = 1:N
            result = optimize(dfunc[n], copy(getparams(model,n)), meta.optimizer, meta.options)
            # # TODO - remove this check?
            # @assert isapprox(getparams(model,n), result.minimum)
            f_x = Optim.minimum(result)
        end

        x_converged,f_converged,converged = assess_convergence(getparams(model), xprev, f_x, f_x_prev, options.x_tol, options.f_tol)
        converged && break
    end

    return converged,tr

end

####
# 
function assess_convergence(x::AbstractArray,
                            x_previous::AbstractArray,
                            f_x::Real,
                            f_x_previous::Real,
                            x_tol::Real,
                            f_tol::Real)
    x_converged, f_converged = false, false

    if Optim.maxdiff(x, x_previous) < x_tol
        x_converged = true
    end

    # Relative Tolerance
    if abs(f_x - f_x_previous) / (abs(f_x) + f_tol) < f_tol || nextfloat(f_x) >= f_x_previous
        f_converged = true
    end

    converged = x_converged || f_converged

    return x_converged, f_converged, converged
end

####
# Decompose the objective function along each mode of the tensor
@generated function _decompose_objective{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N}
    )
  quote 
    # preallocate storage for gradients
    ∇storage = Array(T, maximum(size(model))*rank(model))

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

    return dfunc
  end
end
