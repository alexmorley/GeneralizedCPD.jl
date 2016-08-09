module GeneralizedCPD

using Base.Cartesian
using Base.LinAlg
using CatViews

import StatsBase: fit!

using Reexport
@reexport using TensorBase
@reexport using Losses

export GenCPD,
       GenCPDParams,
       GenCPDFit,
       LossArray,
       fit!,
       setparams!,
       getparams,
       nparams

include("gen_cpd.jl")
# include("gradients.jl")
# include("fit.jl")

end # module
