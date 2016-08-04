type GenCPD{T<:Number,N,L<:Loss,M}
    cpd::CPD{T,N}
    loss::Union{L,AbstractArray{L,M}}

    GenCPD(cpd::CPD{T,N},l::L) = new(cpd,l)
    function GenCPD(cpd::CPD{T,N},loss::AbstractArray{L,M})
        M > N && error("loss array has too many dimensions")
        for m = 1:M
            size(cpd,m) != size(loss,m) && error("loss array does not match cpd along dimension $m")
        end
        new(cpd,loss)
    end
end

function GenCPD{T<:Number,N,L<:Loss,M}(
        cpd::CPD{T,N},
        loss::AbstractArray{L,M}
    )
    return GenCPD{T,N,L,M}(cpd,loss)
end

function GenCPD{T<:Number,N,L<:Loss}(
        cpd::CPD{T,N},
        l::L
    )
    return GenCPD{T,N,L,0}(cpd,l)
end

function GenCPD{T<:Number,N,L<:Loss}(
        nr::Integer,
        data::AbstractArray{T,N},
        l::L
    )
    cpd = CPD(n)
    return GenCPD{T,N,L,0}(cpd,l)
end

type GenCPDParams
    iterations::Integer
    ftol::Float64
end

GenCPDParams(i=100,f=1e-6) = GenCPDParams(i,f)

type GenCPDFit
    iterations::Integer
    objective::Float64
    ∇norm::Float64
    converged::Bool
end