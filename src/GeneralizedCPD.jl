module GeneralizedCPD

using Reexport
@reexport using TensorBase
importall LearnBase
@reexport using Losses
import StatsBase: fit!

export GenCPD,
       GenCPDParams,
       GenCPDFit,
       LossArray,
       fit!

include("utils.jl")
include("types.jl")
# include("fit.jl")

end # module
