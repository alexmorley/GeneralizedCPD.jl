using BenchmarkTools
using GeneralizedCPD

x = CatView( @ntuple 1000 (n)->randn(10) )

@benchmark begin
    for xi in $x
        nothing
    end
end
