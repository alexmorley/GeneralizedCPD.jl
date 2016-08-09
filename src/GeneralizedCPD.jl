module GeneralizedCPD

using Base.Cartesian
using Base.LinAlg
using CatViews

import StatsBase: fit!

using Reexport
@reexport using TensorBase
@reexport using Losses
@reexport using Distributions

export GenCPD,
       GenCPDParams,
       GenCPDFit,
       LossArray,
       fit!,
       setparams!,
       getparams,
       nparams,
       grad

include("gen_cpd.jl")
include("gradients.jl")
# include("fit.jl")

end # module
