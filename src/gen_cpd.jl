"""
    GenCPD(rank,data,loss)

Creates a Generalized Canonical Polyadic Decomposition (CPD) model with specified
rank (an integer), data (a tensor), and loss.
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
      np = nr
      @nexprs $N (n)->np += prod(factdims[n])

      # allocate paramvec and split/create views into paramvec
      x = Array(T,np)
      factors,λ = splitview(x,factdims)

      # create CPD and return
      cpd = CPD(factors,λ)
      return GenCPD{T,N,L,0}(x,factdims,cpd,l)
    end
end

## CPD with params x, copy dimensions and rank from other model
function GenCPD{Tx,T,N,L,M}(
        x::AbstractVector{Tx},
        gencp::GenCPD{T,N,L,M}
    )
    # create views into x as factors
    factors,λ = splitview(x,gencp.factdims)

    # create CPD and return
    cpd = CPD(factors,λ)
    return GenCPD{Tx,N,L,M}(x,gencp.factdims,cpd,gencp.loss)
end

## Getting/Setting parameters ##
function sumvalue(model::GenCPD, x::AbstractVector, data::AbstractArray)
  tmp = GenCPD(x,model)
  setparams!(tmp,x)
  sumvalue(tmp,data)
end

function setparams!(model::GenCPD,x::AbstractVector)
    copy!(model.paramvec,x)
end
# Base.randn!(model::GenCPD) = randn!(model.paramvec)
# Base.rand!(model::GenCPD) = rand!(model.paramvec)

getparams(model::GenCPD) = model.paramvec
nparams(model::GenCPD) = length(model.paramvec)
Base.size(model::GenCPD) = size(model.cpd)

"""
    sumvalue(model::GenCPD, data)

Computes objective function for Generalized CPD model with current parameters.
"""
sumvalue(model::GenCPD,data::AbstractArray) = sumvalue(model.loss,model.cpd,data)

@generated function sumvalue{T,N}(
        loss::Loss,
        cpd::CPD{T,N},
        data::AbstractArray
    )
  quote
    if size(cpd) != size(data)
        error("cpd and data dimensions don't match")
    end
    z = zero(T)
    @nloops $N i data begin
        t = @nref $N data i
        e = @nref $N cpd i
        z += value(loss,t,e)
    end
    return z
  end
end

@generated function sumvalue{L<:Loss,T,N}(
        loss::AbstractArray{L},
        cpd::CPD{T,N},
        data::AbstractArray
    )
  quote
    if size(cpd) != size(data)
        error("cpd and data dimensions don't match")
    end
    ndims(loss) > ndims(data) && error("loss array has more dimensions than data")
    if size(loss) != size(cpd)[1:ndims(loss)]
        error("loss array dimensions don't match data")
    end
    z = zero(T)
    @nloops $N i data begin
        l = @nref $M loss i
        t = @nref $N data i
        e = @nref $N cpd i
        z += value(l,t,e)
    end
    return z
  end
end
