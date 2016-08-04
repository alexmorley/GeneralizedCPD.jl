module GeneralizedCPD

using Reexport
@reexport using TensorBase
@reexport using Losses
import StatsBase: fit!

export GenCPDecomp,
       GenCPDParams,
       GenCPDFit,
       LossArray,
       fit!

include("loss_array.jl")
include("types.jl")
# include("fit.jl")

end # module
