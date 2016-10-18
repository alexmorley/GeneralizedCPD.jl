## Generalized Canonical Polyadic (CP) Tensor Decomposition

[Alex H. Williams](http://alexhwilliams.info/)\*, [Tamara G. Kolda](http://www.sandia.gov/~tgkolda/)

\*Address correspondence to alex.h.willia@gmail.com

The goal of this package is to fit [CP tensor decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) with any reasonable loss function. It leverages [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl) to express the model and [Optim.jl](https://github.com/JuliaOpt/Optim.jl) to fit parameters.

### Installation

Not all of the required packages are in the official registry (yet). So try this and bug me if it doesn't work:

```julia
Pkg.clone("https://github.com/ahwillia/TensorBase.jl")
Pkg.clone("https://github.com/JuliaML/LossFunctions.jl")
Pkg.clone("https://github.com/ahwillia/GeneralizedCPD.jl")
```
