"""
    GenCPD(rank,data,loss)

Creates a Generalized Canonical Polyadic Decomposition (CPD) model with specified
rank (an integer), data (a tensor), and loss.

**Functions to evaluate objective function:**

    sumvalue(model, data)       # eval objective with current params
    sumvalue(model, x, data)    # eval objective with params `x`
    sumvalue!(model, x, data)   # overwrite model with params `x` then eval objective

**Functions to evaluate gradients:**

    grad(model, data)       # eval gradient with current params
    grad(model, x, data)    # eval gradient with params `x`
    grad!(model, x, data)   # overwrite model with params `x` then eval gradient

"""
immutable GenCPD{T,N,L<:Loss,M}
    paramvec::Vector{T}
    factdims::NTuple{N,Tuple{Int,Int}}
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
      # tuple of dimensions of factor matrices
      factdims = @ntuple $N (n)->(size(data,n),nr)
      
      # calculate total number of parameters
      np = 0
      @nexprs $N (n)->np += prod(factdims[n])

      # allocate paramvec and create factors as views into it
      x = Array(T,np)
      factors, = splitview(x,factdims)

      # create CPD and return
      cpd = CPD(factors,ones(T,nr))
      return GenCPD{T,N,L,0}(x,factdims,cpd,l)
    end
end

## CPD with params x, copy dimensions and rank from other model
function GenCPD{Tx,T,N,L,M}(
        x::AbstractVector{Tx},
        gencp::GenCPD{T,N,L,M}
    )
    # create views into x as factors
    factors, = splitview(x,gencp.factdims)
    λ = ones(Tx,rank(gencp))

    # create CPD and return
    cpd = CPD(factors,λ)
    return GenCPD{Tx,N,L,M}(x,gencp.factdims,cpd,gencp.loss)
end

## Convienence functions ##
Base.size(model::GenCPD) = size(model.cpd)
Base.rank(model::GenCPD) = rank(model.cpd)

Base.randn!{T<:AbstractFloat}(model::GenCPD{T}) = randn!(model.paramvec) 
Base.rand!{T<:AbstractFloat}(model::GenCPD{T}) = rand!(model.paramvec)

## get and set model params through paramvec ##
getparams(model::GenCPD) = model.paramvec
nparams(model::GenCPD) = length(model.paramvec)

function setparams!{T}(model::GenCPD{T},x::AbstractVector)
    copy!(model.paramvec,x)
    fill!(model.cpd.λ, one(T))
end 
