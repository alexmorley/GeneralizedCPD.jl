function fit!{T,N}(
        cpd::GenCPDecomp{T,N},
        data::AbstractArray{T,N},
        opt::GenCPDParams = GenCPDParams()
    )

    size(cpd)!=size(data) && error("cpd and data dimensions do not match.")

    ∇norm = 0.0 # todo, calculate this.
    converged = false # todo, convergence

    maxI = maximum(size(data))
    R = rank(cpd)
    ∇store = Array(T,R,maxI) # storage for gradient

    for iter = 1:opt.iterations

        for n = 1:N
            I = size(cpd,n)
            ∇ = view(∇store,:,1:I)
            
        end

        # todo, convergence
    end

    return GenCPDFit(n_iter,f,∇norm,converged)
end
