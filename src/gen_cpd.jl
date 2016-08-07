"""
    GenCPD(rank,data)

Creates a Generalized Canonical Polyadic Decomposition (CPD) model with specified
rank (an integer) and data (a tensor).
"""
immutable GenCPD{T<:Number,N,L<:Loss,M}
    paramvec::Vector{T}
    cpd::CPD{T,N}
    loss::Union{L,AbstractArray{L,M}}

    # GenCPD(cpd::CPD{T,N},l::L) = new(cpd,l)
    # function GenCPD(cpd::CPD{T,N},loss::AbstractArray{L,M})
    #     M > N && error("loss array has too many dimensions")
    #     for m = 1:M
    #         size(cpd,m) != size(loss,m) && error("loss array does not match cpd along dimension $m")
    #     end
    #     new(cpd,loss)
    # end
end

## Constructors ##
@generated function GenCPD{T<:Number,N,L<:Loss}(
        nr::Integer,
        data::AbstractArray{T,N},
        l::L
    )
    quote 
      # allocate vector for parameters
      factdims = @ntuple $N (n)->(size(data,n),nr)
      x,factors = vecmats(T,factdims)
      cpd = CPD(factors, ones(T,nr))
      return GenCPD{T,N,L,0}(x,cpd,l)
    end
end

## Getting/Setting parameters ##
function setparams!{T}(model::GenCPD{T},x::AbstractVector{T})
    copy!(model.paramvec,x)
end
getparams(model::GenCPD) = model.paramvec
nparams(model::GenCPD) = length(model.paramvec)

