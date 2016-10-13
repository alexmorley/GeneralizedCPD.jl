function fakedata{N,Ix<:Integer}(
        sz::NTuple{N,Ix},
        nr::Integer,
        loss::SupervisedLoss,
        σ²::Float64 = 0.1
    )

    # create random low-rank tensor
    cpd = cpd_randn(sz,nr)
    data = Array(Float64,sz)

    # add noise
    f = get_transform(loss,σ²)
    map!(f, data, full(cpd))

    return data,cpd
end

function fakedata{N,Ix<:Integer,L<:SupervisedLoss}(
        sz::NTuple{N,Ix},
        nr::Integer,
        loss::Array{L},
        σ²::Float64 = 0.1
    )

    # loss must match sz for early dimensions
    for (i,dim) in enumerate(sz)
        i > ndims(loss) && break
        size(loss,i) == dim || throw(ArgumentError("Loss array does not match data array size on dimension $i: size(loss,i) == $(size(loss,i)) while size(data,i) == $(dim)"))
    end

    # create random low-rank tensor
    cpd = full(cpd_randn(sz,nr))
    data = similar(cpd)

    # add noise
    for i in eachindex(data)
        idx = ind2sub(sz,i)
        lidx = idx[1:ndims(loss)]
        l = get_transform(loss[lidx...], σ²)
        data[i] = l(cpd[i])
    end

    return data
end

function get_transform(
        loss::SupervisedLoss,
        σ²::Float64
    )

    if typeof(loss) <: Union{LogitMarginLoss,HingeLoss}
        f = (x) -> rand() > logistic(x) ? 1.0 : -1.0
    elseif typeof(loss) <: PoissonLoss
        f = (x) -> rand(Poisson(exp(x)))
    elseif typeof(loss) <: L2DistLoss
        f = (x) -> rand(Normal(x,σ²))
    elseif typeof(loss) <: L1DistLoss
        f = (x) -> rand(Laplace(x,sqrt(0.5*σ²)))
    else
        throw(ArgumentError("$loss does not have a supported noise distribution."))
    end
end
