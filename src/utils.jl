@generated function value!{T,N}(
        dest::AbstractArray{T,N},
        loss::Loss,
        target::AbstractArray{T,N},
        estimate::AbstractArray{T,N}
    )
  quote
    if !(size(target) == size(estimate) == size(dest))
        error("dest, target and estimate dimensions don't match")
    end
    @nloops $N i dest begin
        t = @nref $N target i
        e = @nref $N estimate i
        @nref($N,dest,i) = value(loss,t,e)
    end
    return dest
  end
end

@generated function value!{T,N,L<:Loss,M}(
        dest::AbstractArray{T,N},
        loss::AbstractArray{L,M},
        target::AbstractArray{T,N},
        estimate::AbstractArray{T,N}
    )
  quote
    if !(size(target) == size(estimate) == size(dest))
        error("dest, target and estimate dimensions don't match")
    end
    $M > $N && error("loss array has more dimensions than data")
    if size(loss) != size(dest)[1:($M)]
        error("loss array dimensions don't match data")
    end
    @nloops $N i dest begin
        l = @nref $M loss i
        t = @nref $N target i
        e = @nref $N estimate i
        @nref($N,dest,i) = value(loss,t,e)
    end
    return dest
  end
end

@generated function deriv!{T,N}(
        dest::AbstractArray{T,N},
        loss::Loss,
        target::AbstractArray{T,N},
        estimate::AbstractArray{T,N}
    )
  quote
    if !(size(target) == size(estimate) == size(dest))
        error("dest, target and estimate dimensions don't match")
    end
    @nloops $N i dest begin
        t = @nref $N target i
        e = @nref $N estimate i
        @nref($N,dest,i) = deriv(loss,t,e)
    end
    return dest
  end
end

@generated function deriv!{T,N,L<:Loss,M}(
        dest::AbstractArray{T,N},
        loss::AbstractArray{L,M},
        target::AbstractArray{T,N},
        estimate::AbstractArray{T,N}
    )
  quote
    if !(size(target) == size(estimate) == size(dest))
        error("dest, target and estimate dimensions don't match")
    end
    $M > $N && error("loss array has more dimensions than data")
    if size(loss) != size(dest)[1:($M)]
        error("loss array dimensions don't match data")
    end
    @nloops $N i dest begin
        l = @nref $M loss i
        t = @nref $N target i
        e = @nref $N estimate i
        @nref($N,dest,i) = deriv(loss,t,e)
    end
    return dest
  end
end
