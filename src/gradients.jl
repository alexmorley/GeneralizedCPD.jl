####################################################################################
# Functions to evaluate gradients for Generalized CPD                              #
#                                                                                  #
#    grad(model, data)        # eval gradient with current params                  #
#    grad(model, x, data)     # eval gradient with params `x`                      #
#    grad!(model, x, data)    # overwrite model with params `x` then eval gradient #
#                                                                                  #
#    grad!(∇, model, data)    # eval with current params, store result in `∇`      #
#    grad!(∇, model, x, data) # overwrite model, store result in `∇`               #
#                                                                                  #
####################################################################################

function grad{T,N}(
        model::GenCPD{T,N},
        data::AbstractArray{T,N}
    )
    ∇ = Vector{T}(nparams(model))
    grad!(∇,model,data)
end


function grad(
        model::GenCPD,
        x::AbstractVector,
        data::AbstractArray
    )
    tmp = GenCPD(x,model)
    grad!(tmp,x,data)
end

function grad!{T}(
        model::GenCPD{T},
        x::AbstractVector{T},
        data::AbstractArray
    )
    setparams!(model,x)
    ∇ = Vector{T}(nparams(model))
    grad!(∇,model,data)
end

# ---------------------------- #

@generated function grad!{T,N}(
        ∇::AbstractVector{T},
        model::GenCPD{T,N},
        data::AbstractArray{T,N}
    )
    quote
    F, = splitview(∇,model.factdims)
    @nexprs $N n->(grad!(F[n],model,data,n))
    return ∇
    end
end

function grad!{T,N}(
        ∇::AbstractVector{T},
        model::GenCPD{T,N},
        x::AbstractVector,
        data::AbstractArray{T,N}
    )
    setparams!(model,x)
    grad!(∇,tmp,data)
end

#######################################################################################
# Functions to evaluate gradients for a specified factor matrix. Returns the gradient #
# in a matrix ∇ that has the same dimensions as the mode-n factor matrix.             #
#                                                                                     #
#    grad(model, data, n)      # gradient of mode-n factor with current model params  #
#    grad!(∇, model, data, n)  # same, except writing over ∇ to store result          #
#                                                                                     #
#######################################################################################

function grad(
        model::GenCPD,
        data::AbstractArray,
        n::Integer
    )
    grad!(similar(model.cpd.factors[n]),model,data,n)
end

function grad!{T,N}(
        ∇::AbstractMatrix{T},
        model::GenCPD{T,N},
        data::AbstractArray{T,N},
        n::Integer
    )
    
    factors = model.cpd.factors

    # form estimate of unfolded tensor
    idx = [N:-1:n+1; n-1:-1:1]
    B = reduce(krprod, factors[idx])            
    est = A_mul_Bt(factors[n],B)

    # unfold tensor along mode n 
    xn = unfold(data,n)
    deriv!(xn,model.loss,xn,est)

    # compute gradient for factor n
    A_mul_B!(∇,xn,B)
    return ∇
end
