type LossArray{L<:Loss,M}
    loss::Union{L,AbstractArray{L,M}}

    LossArray(l::L) = new(l)
    LossArray(l::AbstractArray{L,M}) = new(l)
end
LossArray{L<:Loss}(l::L) = LossArray{L,0}(l)
LossArray{L<:Loss,M}(l::AbstractArray{L,M}) = LossArray{L,M}(l)

function Base.size{L,M}(A::LossArray{L,M})
    if M == 0
        return ()
    else
        return size(A.loss)
    end
end

function Base.size{L,M}(A::LossArray{L,M},d::Integer)
    if M == 0
        return ()
    else
        return size(A.loss,d)
    end
end

function Base.getindex{L,M}(A::LossArray{L,M},idx::Integer...)
    if M == 0 
        return A.loss::Loss
    else
        return A.loss[idx[1:M]...]::Loss
    end
end


function sumvalue(
        A::LossArray,
        target::AbstractArray,
        output::AbstractArray
    )
end
