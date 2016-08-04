type GenCPDecomp{T<:Number,N,L<:Loss,M}
    cpd::CPDecomp{T,N}
    loss::LossArray{L,M}

    function GenCPDecomp(cpd::CPDecomp{T,N},loss::LossArray{L,M})
        M > N && error("loss array has too many dimensions")
        for m = 1:M
            size(cpd,m) != size(loss,m) && error("loss array does not match cpd along dimension $m")
        end
        new(cpd,loss)
    end
end

function GenCPDecomp{T<:Number,N,L<:Loss,M}(
        cpd::CPDecomp{T,N},
        loss::LossArray{L,M}
    )
    return GenCPDecomp{T,N,L,M}(cpd,loss)
end

function GenCPDecomp{T<:Number,N,L<:Loss,M}(
        cpd::CPDecomp{T,N},
        loss::AbstractArray{L,M}
    )
    return GenCPDecomp{T,N,L,M}(cpd,LossArray(loss))
end

function GenCPDecomp{T<:Number,N,L<:Loss}(
        cpd::CPDecomp{T,N},
        l::L
    )
    return GenCPDecomp{T,N,L,0}(cpd,LossArray(l))
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