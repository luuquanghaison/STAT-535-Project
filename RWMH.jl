using MCMCChains
include("State_object.jl")
include("explorers.jl")

sample_step(explorer::RWMH, rng, t) = if explorer.dimension == 1
        return randn(rng)*sqrt(explorer.step)
    else
        return rand(rng,MvNormal(explorer.step))
    end

sample_step(explorer::AM, rng, t) = if explorer.dimension == 1
        return randn(rng)*sqrt(explorer.Sigma[1])
    else
        return rand(rng,MvNormal(explorer.Sigma))
    end

function sample_step(explorer::RRWMH, rng, t)
    sigma = Diagonal([sqrt(abs(rand(rng,qi))) for qi in explorer.q])
    sigma = sigma * Diagonal(explorer.Sigma)^(-1/2)
    step = sigma * explorer.Sigma * sigma
    step = (step + step')./2 # to make sure step is positive definite
    #@show step
    if explorer.dimension == 1
        return randn(rng)*sqrt(step[1])
    else
        return rand(rng,MvNormal(step))
    end
end
    


# step function
function step!(explorer, replica::State_object, log_potential, t)
    lp = replica.lp
    lp = RWMH_sample!(explorer, replica, log_potential, lp, t)    # update chain state
    replica.lp = lp    
    return lp
end


# sample coordinate
function RWMH_sample!(explorer, replica::State_object, log_potential, cached_lp, t)
    d = length(replica.state)
    rng = replica.rng

    # propose
    state_before = copy(replica.state)    # store previous state
    eps = sample_step(explorer,rng,t)
    replica.state = state_before .+ eps
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
function run!(explorer, replica::State_object, N::Int, log_potential, update_func!)
    d = length(replica.state)
    chain = Array{Float64}(undef,N,d+1)
    chain[1,:] = vcat(replica.state,replica.lp)


    for i in 2:N
        step!(explorer, replica, log_potential, i)
        chain[i,:] = vcat(replica.state,replica.lp)
        update_func!(explorer, chain, i)                         # adaptation
    end

    param_names = ["Parameter $i" for i in 1:d]
    push!(param_names, "lp")

    return Chains(chain, param_names, (internals=["lp"],))
end
