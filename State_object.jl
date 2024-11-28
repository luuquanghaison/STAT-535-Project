using SplittableRandoms
using Distributions
using Random
using LinearAlgebra

mutable struct State_object{T,R}
    """ Current state. """
    state::T
    """ Cached log potential. """
    lp::Float64
    """ RNG. """
    rng::R
end

function create_replica(seed,d,log_potential,init_state = nothing)
    rng = SplittableRandom(seed)
    if isnothing(init_state)
        init_state = randn(rng,d)
    end
    lp_val = log_potential(init_state)
    return State_object(init_state,lp_val,rng)
end