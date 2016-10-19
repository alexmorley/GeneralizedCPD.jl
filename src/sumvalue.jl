#########################################################################################
#                                                                                       # 
# Functions to evaluate objective function for Generalized CPD.                         #
#                                                                                       #
#     sumvalue(model, data)       # eval objective with current params                  #
#     sumvalue(model, x, data)    # eval objective with params `x`                      #
#     sumvalue!(model, x, data)   # overwrite model with params `x` then eval objective #
#                                                                                       #
#########################################################################################

sumvalue(model::GenCPD,data::AbstractArray) = sumvalue(model.loss,model.cpd,data)

function sumvalue(model::GenCPD, x::AbstractVector, data::AbstractArray)
    temp_model = GenCPD(data, rank(model), model.loss)
    setparams!(temp_model, x)
    sumvalue(temp_model, data)
end

function sumvalue!(model::GenCPD, x::AbstractVector, data::AbstractArray)
    setparams!(model,x)
    sumvalue(model,data)
end

##########################################################################
#                                                                        #
# Lower-level functions to evaluate loss between cpd estimate and data.  #
#                                                                        #
#     sumvalue(::Loss, cpd, data)           # one loss for whole model   #
#     sumvalue(::Array{Loss}, cpd, data)    # uses broadcasting          #
#                                                                        #
##########################################################################

sumvalue(loss::Loss, cpd::CPD, data::AbstractArray) = sumvalue(loss, data, full(cpd))

##########################################################################
#                                                                        #
# Functions to evaluate objective at individual datapoints (for          #
# stochastic methods).                                                   #
#                                                                        #
##########################################################################

@inline value(model::GenCPD, data::AbstractArray, idx::Integer...) = value(model.loss, model.cpd, data, idx...)
@inline value(loss::Loss, cpd::CPD, data::AbstractArray, idx::Integer...) = value(loss, data[idx...], cpd[idx...])

# @generated function sumvalue{L<:Loss,T,N}(
#         loss::AbstractArray{L},
#         cpd::CPD{T,N},
#         data::AbstractArray
#     )
#   quote
#     if size(cpd) != size(data)
#         error("cpd and data dimensions don't match")
#     end
#     ndims(loss) > ndims(data) && error("loss array has more dimensions than data")
#     if size(loss) != size(cpd)[1:ndims(loss)]
#         error("loss array dimensions don't match data")
#     end
#     z = zero(T)
#     @nloops $N i data begin
#         l = @nref $M loss i
#         t = @nref $N data i
#         e = @nref $N cpd i
#         z += value(l,t,e)
#     end
#     return z
#   end
# end
