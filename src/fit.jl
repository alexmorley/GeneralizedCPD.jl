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
    result = optimize(dfunc,params(model),mo,o)
    
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
        meta::AlternatingDescent{O};
        iterations::Int = 100,
        f_tol::Float64 = 1e-7,
        x_tol::Float64 = 1e-7,
        show_trace::Bool = true,
        show_every::Int = 1
    )
    
    dfunc = _decompose_objective(model, data)

    # objective and convergence
    f_x = sumvalue(model,data)
    converged = false
    iter = 1

    # objective function history
    f_hist = Float64[]
    sizehint!(f_hist, iterations)

    # optimization loop
    tic()
    while iter < iterations
        # stor previous params and 
        xprev = copy(params(model))
        f_x_prev = f_x
        
        # optimize over each subproblem
        for n = 1:N
            result = optimize(dfunc[n], copy(params(model,n)), meta.optimizer, meta.options)
            # # TODO - remove this check?
            # @assert isapprox(params(model,n), result.minimum)
            f_x = Optim.minimum(result)
        end

        if show_trace
            mod(iter,show_every)==0 && print_with_color(:blue,"Iteration $iter, loss  =  $(round(f_x,8))\n")
        end

        push!(f_hist, f_x)

        x_converged,f_converged,converged = assess_convergence(params(model), xprev, f_x, f_x_prev, x_tol, f_tol)
        converged && break
        iter += 1
    end
    t = toq();

    if show_trace
        converged && print_with_color(:red,"Algorithm converged ")
        !converged && print_with_color(:red,"DID NOT CONVERGE ")
        print_with_color(:red,"after $iter iterations\n")
        print_with_color(:red,"Total time elapsed, $t seconds\n")
    end

    return converged,f_hist

end

####
# Function to assess convergence (similar to Optim.jl)
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
