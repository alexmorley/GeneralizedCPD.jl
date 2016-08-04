function fit!{T,N}(
        model::GenCPDecomp{T,N},
        data::AbstractArray{T,N},
        opt::GenCPDParams = GenCPDParams()
    )
    
    cpd = model.cpd
    dims = size(cpd)
    dims != size(data) && error("cpd and data dimensions do not match.")

    ∇norm = 0.0 # todo, calculate this.
    converged = false # todo, convergence

    R = rank(cpd)
    
    # todo, think harder about transposing...
    factors = [ transpose(F) for F in cpd.factors ]
    ρ = ones(N)
    fill!(cpd.λ)

    # # todo, clever storage for gradient
    # maxI = maximum(dims)
    # ∇store = Array(T,R,maxI)

    # initial loss
    f0 = sumvalue(cpd,data)

    # optimization loop
    for iter = 1:opt.iterations

        for n = 1:N

            # form estimate of unfolded tensor
            idx = [N:-1:i + 1; i - 1:-1:1]
            B = reduce(krprod, factors[idx])            
            est = A_mul_Bt(factors[n],B)

            # unfold tensor along mode n 
            xn = unfold(data,n)
            deriv!(xn,model.loss,xn,est)

            # compute gradient for factor n
            ∇ = xn*B

            # take gradient step
            axpy!(-ρ[n],∇,factors[n])
            transpose!(cpd.factors[n],factors[n])
            f = sumvalue(model.loss,cpd,data)

            # backtracking linesearch
            while (f-f0)>eps()
                s = ρ[n]*0.5
                axpy!(s,∇,factors[n])
                transpose!(cpd.factors[n],factors[n])
                ρ[n] = ρ[n] - s
                f = sumvalue(model.loss,cpd,data)
            end

            # renormalize factors
            for r = 1:R
                λr = norm(factors[n][:,r])
                factors[n][:,r] ./= λr
                cpd.λ[r] *= λr
            end

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
    end

    # copy factors in CPD object
    for n = 1:N
        transpose!(cpd.factors[n],factors[n])
    end

    return GenCPDFit(n_iter,f,∇norm,converged)
end
