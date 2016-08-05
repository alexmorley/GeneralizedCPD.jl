## Optimization Types ##
type GenCPDParams
    iterations::Integer
    ftol::Float64
end

GenCPDParams(i=100,f=1e-6) = GenCPDParams(i,f)

type GenCPDFit{T<:AbstractFloat}
    iterations::Integer
    trace::Vector{T}
    objective::T
    ∇norm::T
    converged::Bool
end

## Alternating Gradient Descent ##

function fit!{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        opt::GenCPDParams = GenCPDParams()
    )
    
    cpd = model.cpd
    nr = rank(cpd)
    factors = cpd.factors
    fill!(cpd.λ,one(T))
    
    dims = size(cpd)
    dims != size(data) && error("cpd and data dimensions do not match.")

    ∇norm = 0.0 # todo, calculate this.
    converged = false # todo, convergence
    
    # gradient step sizes
    ρ = ones(T,N)

    # # todo, clever storage for gradient
    # maxI = maximum(dims)
    # ∇store = Array(T,R,maxI)

    # initial loss
    f = sumvalue(model,data)
    fhist = [f]

    # optimization loop
    iter = 1
    while iter < opt.iterations

        for n = 1:N

            # calculate gradient
            ∇ = grad(model,data,n)

            @assert size(∇) == size(factors[n])

            # take gradient step
            ρ[n] = 1.0
            axpy!(-ρ[n],∇,factors[n])
            fnext = sumvalue(model,data)

            # backtracking linesearch
            lsiter = 0
            while (fnext-f)>eps()
                s = ρ[n]*0.5
                axpy!(s,∇,factors[n])
                ρ[n] = ρ[n] - s
                fnext = sumvalue(model,data)
                lsiter += 1
                lsiter > 1000 && break
            end
            f = fnext

            # todo:
            #   - preallocate space for B
            #   - preallocate space for unfolding, add unfold!

            # # todo, clever storage for gradient
            # I = size(cpd,n)
            # ∇ = view(∇store,:,1:I)

            # todo, preallocate B
            #w = prod(dims[idx])
            #B[1:w, :] = reduce(krprod, factors[idx])

        end

        # todo, convergence
        push!(fhist,f)
        normalize!(cpd)
        iter += 1
    end

    @show ρ
    @show fhist

    return GenCPDFit(iter,fhist,f,∇norm,converged)
end
