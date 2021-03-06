"""
# Generalized Canonical Polyadic Decomposition (GCPD)

Alex H. Williams¹, Tamara G. Kolda²\n
 ¹ Stanford University, Stanford, CA\n
 ² Sandia National Laboratories, Livermore, CA\n
"""
module GeneralizedCPD

# basic utils
using Base.Cartesian
using Base.LinAlg
using CatViews
using Parameters
using Reexport

# JuliaML packages
importall LearnBase
using LossFunctions
import StochasticOptimization:
              ParamUpdater,
              Adam,
              SGD,
              init

# JuliaStats packages
import StatsBase: fit!
using StatsFuns
using Distributions

# JuliaTensors packages
@reexport using TensorBase

# JuliaOpt packages
using Optim

# export Optimizers
export GradientDescent,
       LBFGS,
       AlternatingDescent

# export stochastic optimizers
export ParamUpdater,
       Adam,
       SGD

# reexport some loss functions
export L2DistLoss,
       L1DistLoss,
       LogitMarginLoss,
       HingeLoss,
       PoissonLoss

# reexport main functionality
export GenCPD,
       GenCPDParams,
       GenCPDFit,
       LossArray,
       fit!,
       setparams!,
       params,
       nparams,
       grad,
       sgrad,
       value,
       sumvalue,
       sumvalue!

# generalized cpd code
include("gen_cpd.jl")
include("sumvalue.jl")
include("gradients.jl")
include("utils.jl")
include("fit.jl")
include("stochastic_methods.jl")
include("fakedata.jl")

end # module
