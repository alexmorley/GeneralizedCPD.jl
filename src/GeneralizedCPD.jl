module GeneralizedCPD

using Reexport
@reexport using TensorBase
importall LearnBase
@reexport using Losses
import StatsBase: fit!

export GenCPDecomp,
       GenCPDParams,
       GenCPDFit,
       LossArray,
       fit!

include("types.jl")
# include("fit.jl")

end # module
