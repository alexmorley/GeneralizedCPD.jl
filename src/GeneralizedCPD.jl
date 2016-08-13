"""
# Generalized Canonical Polyadic Decomposition (GCPD)

Alex H. Williams¹, Tamara G. Kolda²\n
 ¹ Stanford University, Stanford, CA\n
 ² Sandia National Laboratories, Livermore, CA\n
"""
module GeneralizedCPD

using Base.Cartesian
using Base.LinAlg
using CatViews

import StatsBase: fit!
using Optim
using ProgressMeter

# export Optimizers
export GradientDescent,
       LBFGS,
       AltGradDescent

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
include("sumvalue.jl")
include("gradients.jl")
include("utils.jl")
include("fit.jl")

end # module
