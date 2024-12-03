using SplittableRandoms
using Distributions
using Random
using LinearAlgebra

# object containing state info
mutable struct State_object{T,R,L}
    """ Current state. """
    state::T
    """ Cached log potential. """
    lp::Float64
    """ RNG. """
    rng::R
    """ Other stats."""
    other_stats::L
end

# stored stats for RRWMH
mutable struct RRWMH_stats{T}
    """ Moving mean."""
    moving_mean::T
    """ Moving squared mean."""
    moving_mean2::T
    """ Mean of moving variances."""
    moving_vars_mean::T
    """ Squared mean of moving variances."""
    moving_vars_mean2::T
    """ Mean vector of q"""
    m_vec::T
    """ Variance vector of q"""
    V_vec::T
end

RRWMH_stats(init_state) = RRWMH_stats(init_state,init_state.^2,
fill(0.0,length(init_state)),fill(0.0,length(init_state)),
fill(1.0,length(init_state)),fill(1.0,length(init_state)))

# create state object
function create_replica(seed, d, log_potential; init_state = nothing, other_stats = nothing, scheme = :RWMH)
    rng = SplittableRandom(seed)
    if isnothing(init_state)
        init_state = randn(rng,d)
    end
    lp_val = log_potential(init_state)
    if scheme == :RRWMH
        other_stats = RRWMH_stats(init_state)
    end
    return State_object(init_state,lp_val,rng,other_stats)
end