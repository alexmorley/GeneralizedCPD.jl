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
end

## Constructors ##
@generated function GenCPD{T<:AbstractFloat,N,L<:Loss}(
        data::AbstractArray{T,N},
        nr::Integer,
        loss::L
    )
    quote
      # tuple of dimensions of factor matrices
      fdims = @ntuple $N (n)->(nr,size(data,n))
      
      # calculate total number of parameters
      np = 0
      @nexprs $N (n)->np += prod(fdims[n])

      # allocate paramvec and create factors as views into it
      x = Array(T,np)
      factors,fstart,fstop = splitview(x,fdims)

      # create CPD and return
      cpd = CPD(factors,ones(T,nr))
      randn!(x) # initialize params to non-zero
      return GenCPD{T,N,L,0}(x, fdims, fstart, fstop, cpd, loss)
    end
end

## Convienence functions ##
Base.size(model::GenCPD) = size(model.cpd)
Base.size(model::GenCPD, i) = size(model.cpd, i)
Base.rank(model::GenCPD) = rank(model.cpd)

Base.randn!{T<:AbstractFloat}(model::GenCPD{T}) = randn!(model.paramvec) 
Base.rand!{T<:AbstractFloat}(model::GenCPD{T}) = rand!(model.paramvec)

## get and set model params through paramvec ##
getparams(model::GenCPD) = model.paramvec
nparams(model::GenCPD) = length(model.paramvec)

@generated function setparams!{T,N}(model::GenCPD{T,N}, x::AbstractVector)
  quote
    @nexprs $N n->copy!(model.cpd.factors[n], reshape(x[model.fstart[n]:model.fstop[n]], model.fdims[n]))
  end
end 

function setparams!{T,N}(model::GenCPD{T,N}, x::AbstractVector, n::Integer)
    copy!(model.cpd.factors[n], reshape(x, model.fdims[n]))
end

getparams{T,N}(model::GenCPD{T,N}, n::Integer) = view(model.paramvec, model.fstart[n]:model.fstop[n])
