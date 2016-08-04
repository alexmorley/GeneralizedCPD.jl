function value(cpd::GenCPDecomp, data::AbstractArray{T,N})

end

function fit!{T,N}(
        cpd::GenCPDecomp{T,N},
        data::AbstractArray{T,N},
        opt::GenCPDParams = GenCPDParams()
    )

    dims = size(cpd)
    dims != size(data) && error("cpd and data dimensions do not match.")

    ∇norm = 0.0 # todo, calculate this.
    converged = false # todo, convergence

    R = rank(cpd)
    
    # todo, think harder about transposing...
    factors = [ transpose(F) for F in cpd.factors ]
    ρ = ones(N)

    # # todo, clever storage for gradient
    # maxI = maximum(dims)
    # ∇store = Array(T,R,maxI)

    for iter = 1:opt.iterations

        for n = 1:N

            # form estimate of unfolded tensor
            idx = [N:-1:i + 1; i - 1:-1:1]
            B = reduce(krprod, factors[idx])            
            est = A_mul_Bt(factors[n],B)

            # unfold tensor along mode n 
            xn = unfold(data,n)
            deriv!(xn,cpd.loss,xn,est)

            # compute gradient for factor n
            ∇ = xn*B

            # simple backtracking linesearch
            axpy!(-ρ[n],∇,factors[n])
            transpose!(cpd.factors[n],factors[n])
            f = sumvalue(cpd,data)

            # renormalize factors
            for r = 1:R
                factors[n][:,r] ./= norm(factors[n][:,r])
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
