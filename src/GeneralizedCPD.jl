module GeneralizedCPD

using Base.Cartesian
using Base.LinAlg

importall LearnBase
import StatsBase: fit!
import CatViews: vecmats

using Reexport
@reexport using TensorBase
@reexport using Losses

export GenCPD,
       GenCPDParams,
       GenCPDFit,
       LossArray,
       fit!

include("gen_cpd.jl")
include("gradients.jl")
include("fit.jl")

end # module
