function fakedata{Ix<:Integer}(
        sz::NTuple{N,Ix},
        nr::Integer,
        ξ::AbstractFloat = 0.1
    )

    # create random low-rank tensor
    cpd = cpd_randn(sz,nr)

    # add gaussian noise and return
    return full(cpd) + ξ*randn(sz)
end
