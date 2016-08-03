module GeneralizedCPD

using Reexport
@reexport using TensorBase
@reexport using Losses

export GenCPDecomp

type GenCPDecomp{T<:Number,N,L<:Loss,M}
    cpd::CPDecomp{T,N}
    loss::Union{L,AbstractArray{L,M}}

    GenCPDecomp(cpd::CPDecomp{T,N},l::L) = new(cpd,l)
    function GenCPDecomp(cpd::CPDecomp{T,N},loss::AbstractArray{L,M})
        M > N && error("loss array has too many dimensions")
        for m = 1:M
            size(cpd,m) != size(loss,m) && error("loss array does not match cpd along dimension $m")
        end
        new(cpd,loss)
    end
end

function GenCPDecomp{T<:Number,N,L<:Loss,M}(
        cpd::CPDecomp{T,N},
        loss::AbstractArray{L,M}
    )
    return GenCPDecomp{T,N,L,M}(cpd,loss)
end

function GenCPDecomp{T<:Number,N,L<:Loss}(
        cpd::CPDecomp{T,N},
        l::L
    )
    return GenCPDecomp{T,N,L,0}(cpd,l)
end


# type GenCPDParams
#     iterations::Integer
#     ftol::Float64
# end

# GenCPDParams(i=100,f=1e-6) = GenCPDParams(i,f)

# type GenCPDFit
#     iterations::Integer
#     objective::Float64
#     ∇norm::Float64
#     converged::Bool
# end

# function fit!{T,N}(
#         cpd::GenCPDecomp{T,N},
#         data::AbstractArray{T,N},
#         opt::GenCPDParams
#     )
#     ∇norm = 0.0 # todo, calculate this.
#     converged = false # todo, convergence

#     for iter = 1:opt.iterations



#         # todo, convergence
#     end

#     return GenCPDFit(n_iter,f,∇norm,converged)
# end

end # module
