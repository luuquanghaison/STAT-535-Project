using MCMCChains
include("State_object.jl")

mutable struct RWMH_sampler{R,T}
    """ Step size. """
    step_size::R
    """ Correlation structure. """
    S::T
end

# step function
function step!(explorer::RWMH_sampler, replica::State_object, log_potential, state_update!)
    lp = replica.lp
    lp = RWMH_sample!(explorer, replica, log_potential, lp)    # update chain state
    replica.lp = lp    
    state_update!(explorer, replica)                                       # other state related updates (for adaptation)
    return lp
end


# sample coordinate
function RWMH_sample!(explorer::RWMH_sampler, replica::State_object, log_potential, cached_lp)
    d = length(replica.state)
    rng = replica.rng

    # propose
    state_before = copy(replica.state)    # store previous state
    if d == 1
        replica.state = state_before .+ rand(rng,Normal(0,sqrt(explorer.S))) .* rand(rng,explorer.step_size)
    else
        replica.state = state_before .+ rand(rng,MvNormal(explorer.S)) .* rand(rng,explorer.step_size)
    end
    log_pr_after = log_potential(replica.state)
    # accept-reject step
    accept_ratio = log_pr_after - cached_lp
    if accept_ratio < -randexp(rng)
        # reject: revert the move we just proposed
        replica.state = state_before
        log_pr_after = cached_lp
    end # (nothing to do if accept, we work in-place)
    return log_pr_after
end



# run MCMC chain
function run!(explorer::RWMH_sampler, replica::State_object, N::Int, log_potential, state_update!)
    d = length(replica.state)
    chain = Array{Float64}(undef,N,d+1)
    chain[1,:] = vcat(replica.state,replica.lp)


    for i in 2:N
        step!(explorer, replica, log_potential, state_update!)
        chain[i,:] = vcat(replica.state,replica.lp)
    end

    param_names = ["Parameter $i" for i in 1:d]
    push!(param_names, "lp")

    return Chains(chain, param_names, (internals=["lp"],))
end
