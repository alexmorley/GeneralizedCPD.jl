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

@generated function sumvalue{T,N}(
        loss::SupervisedLoss,
        cpd::CPD{T,N},
        data::AbstractArray{T,N}
    )
  quote
    if size(cpd) != size(data)
        error("cpd and data dimensions don't match")
    end
    $M > $N && error("loss array has more dimensions than data")
    if size(loss) != size(cpd)[1:($M)]
        error("loss array dimensions don't match data")
    end
    z = zero(T)
    @nloops $N i dest begin
        t = @nref $N data i
        e = @nref $N cpd i
        z += value(loss,t,e)
    end
    return z
  end
end

@generated function sumvalue{T,N,L<:Loss,M}(
        loss::AbstractArray{L,M},
        cpd::CPD{T,N},
        data::AbstractArray{T,N}
    )
  quote
    if size(cpd) != size(data)
        error("cpd and data dimensions don't match")
    end
    $M > $N && error("loss array has more dimensions than data")
    if size(loss) != size(cpd)[1:($M)]
        error("loss array dimensions don't match data")
    end
    z = zero(T)
    @nloops $N i dest begin
        l = @nref $M loss i
        t = @nref $N data i
        e = @nref $N cpd i
        z += value(l,t,e)
    end
    return z
  end
end

