"""
    GenCPD(data, nr, loss)

Creates a Generalized Canonical Polyadic Decomposition (CPD) model with specified
rank (an integer), data (a tensor), and loss (either a SupervisedLoss).

#### To specify a model:

A rank `nr` generalized CPD model with `loss` specifying loss function(s).

    model = GenCPD(data, nr, loss)

The dimensions of the model and the data type (e.g. `Float64`) match the data
tensor. The `loss` should be either a `SupervisedLoss` or an `Array` of 
SupervisedLosses. Common losses include (see `Losses.jl` package for more):

    * `L2DistLoss()`, quadratic error
    * `L1DistLoss()`, linear error
    * `LogitMarginLoss()`, common loss for binary data
    * `HingeLoss()`, another loss for binary data
    * `PoissonLoss()`, common loss for count data
    * `HuberLoss()`, robust loss

If `loss` is an `Array`, then the number of dimensions should be less than or
equal to the `data` tensor, and the sizes of these dimensions should match
`data`.

#### To optimize model parameters:

After specifying the model, you can fit the parameters by:

    fit!(model, data, ::Optimizer, ::OptimizationOptions)

The Optimizer can be:
  - any Optim.jl solver, e.g. LBFGS(), GradientDescent()
  - any Optimizer implemented in this package:
    * AltGradientDescent(), alternating gradient descent

#### Functions to evaluate objective function:

    sumvalue(model, data)       # eval objective with current params
    sumvalue(model, x, data)    # eval objective with params `x`
    sumvalue!(model, x, data)   # overwrite model with params `x` then eval objective

#### Functions to evaluate gradients:

    grad(model, data)       # eval gradient with current params
    grad(model, x, data)    # eval gradient with params `x`
    grad!(model, x, data)   # overwrite model with params `x` then eval gradient

"""
type GenCPD{T,N,L<:Loss,M}
    paramvec::Vector{T}
    fdims::NTuple{N,Tuple{Int,Int}}
    fstart::NTuple{N,Int}
    fstop::NTuple{N,Int}
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
        data::AbstractArray{T,N},
        nr::Integer,
        loss::L
    )
    quote
      # tuple of dimensions of factor matrices
      fdims = @ntuple $N (n)->(size(data,n),nr)
      
      # calculate total number of parameters
      np = 0
      @nexprs $N (n)->np += prod(fdims[n])

      # allocate paramvec and create factors as views into it
      x = Array(T,np)
      factors,fstart,fstop = splitview(x,fdims)

      # create CPD and return
      cpd = CPD(factors,ones(T,nr))
      return GenCPD{T,N,L,0}(x, fdims, fstart, fstop, cpd, loss)
    end
end

# ## CPD with params x, copy dimensions and rank from other model
# @generated function GenCPD{Tx,T,N,L,M}(
#         paramvec::AbstractVector{Tx},
#         model::GenCPD{T,N,L,M}
#     ) 
#     # create views into x as factors
#     factors = splitview(paramvec, model.fdims)
#     λ = ones(Tx, rank(model))

#     # create CPD and return
#     cpd = CPD(factors,λ)
#     return GenCPD{Tx,N,L,M}(paramvec, model.fdims, model.fstart,
#                           model.fstop, cpd, model.loss)
# end

"""
    factor_view(gencpd, n)

Returns a contiguous view into `gencpd.paramvec` that holds factor matrix `n`.
"""
@inline factor_view(model::GenCPD, n::Integer) = view(model.paramvec, model.fstart[n]:model.fstop[n])

## Convienence functions ##
Base.size(model::GenCPD) = size(model.cpd)
Base.rank(model::GenCPD) = rank(model.cpd)

Base.randn!{T<:AbstractFloat}(model::GenCPD{T}) = randn!(model.paramvec) 
Base.rand!{T<:AbstractFloat}(model::GenCPD{T}) = rand!(model.paramvec)

## get and set model params through paramvec ##
getparams(model::GenCPD) = model.paramvec
nparams(model::GenCPD) = length(model.paramvec)

@generated function setparams!{T,N}(model::GenCPD{T,N}, x::AbstractVector)
  quote
    model.cpd.factors = @ntuple $N (n)-> reshape(view(x, model.fstart[n]:model.fstop[n]), model.fdims[n])
  end
end 

# function setparams!{T}(model::GenCPD{T},x::AbstractVector)
#     copy!(model.paramvec,x)
# end 
