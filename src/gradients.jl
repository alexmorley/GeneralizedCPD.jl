# @generated function value!{T,N}(
#         dest::AbstractArray{T,N},
#         loss::Loss,
#         target::AbstractArray{T,N},
#         estimate::AbstractArray{T,N}
#     )
#   quote
#     if !(size(target) == size(estimate) == size(dest))
#         error("dest, target and estimate dimensions don't match")
#     end
#     @nloops $N i dest begin
#         t = @nref $N target i
#         e = @nref $N estimate i
#         @nref($N,dest,i) = value(loss,t,e)
#     end
#     return dest
#   end
# end

# @generated function value!{T,N,L<:Loss,M}(
#         dest::AbstractArray{T,N},
#         loss::AbstractArray{L,M},
#         target::AbstractArray{T,N},
#         estimate::AbstractArray{T,N}
#     )
#   quote
#     if !(size(target) == size(estimate) == size(dest))
#         error("dest, target and estimate dimensions don't match")
#     end
#     $M > $N && error("loss array has more dimensions than data")
#     if size(loss) != size(dest)[1:($M)]
#         error("loss array dimensions don't match data")
#     end
#     @nloops $N i dest begin
#         l = @nref $M loss i
#         t = @nref $N target i
#         e = @nref $N estimate i
#         @nref($N,dest,i) = value(loss,t,e)
#     end
#     return dest
#   end
# end

# @generated function deriv!{T,N}(
#         dest::AbstractArray{T,N},
#         loss::Loss,
#         target::AbstractArray{T,N},
#         estimate::AbstractArray{T,N}
#     )
#   quote
#     if !(size(target) == size(estimate) == size(dest))
#         error("dest, target and estimate dimensions don't match")
#     end
#     @nloops $N i dest begin
#         t = @nref $N target i
#         e = @nref $N estimate i
#         @nref($N,dest,i) = deriv(loss,t,e)
#     end
#     return dest
#   end
# end

# @generated function deriv!{T,N,L<:Loss,M}(
#         dest::AbstractArray{T,N},
#         loss::AbstractArray{L,M},
#         target::AbstractArray{T,N},
#         estimate::AbstractArray{T,N}
#     )
#   quote
#     if !(size(target) == size(estimate) == size(dest))
#         error("dest, target and estimate dimensions don't match")
#     end
#     $M > $N && error("loss array has more dimensions than data")
#     if size(loss) != size(dest)[1:($M)]
#         error("loss array dimensions don't match data")
#     end
#     @nloops $N i dest begin
#         l = @nref $M loss i
#         t = @nref $N target i
#         e = @nref $N estimate i
#         @nref($N,dest,i) = deriv(loss,t,e)
#     end
#     return dest
#   end
# end



# function grad{T,N}(
#         model::GenCPD{T,N},
#         data::AbstractArray{T,N},
#         n::Integer
#     )
    
#     factors = model.cpd.factors

#     # form estimate of unfolded tensor
#     idx = [1:n-1; n+1:N]#[N:-1:n+1; n-1:-1:1]
#     B = reduce(krprod, factors[idx])            
#     est = A_mul_Bt(factors[n],B)

#     # unfold tensor along mode n 
#     xn = unfold(data,n)
#     deriv!(xn,model.loss,xn,est)

#     # compute gradient for factor n
#     return xn*B
# end

# # form estimate of unfolded tensor
# idx = [N:-1:n+1; n-1:-1:1]
# B = reduce(krprod, factors[idx])            
# est = A_mul_Bt(factors[n],B)

# # unfold tensor along mode n 
# xn = unfold(data,n)
# deriv!(xn,model.loss,xn,est)

# # compute gradient for factor n
# ∇ = xn*B
