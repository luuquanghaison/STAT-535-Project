using StatsPlots
using KernelDensity
using Plots

include("RWMH.jl")
include("targets.jl")
include("update_functions.jl")

# function to run experiments
function Run_sampler(targets, d, target_names)
    res = DataFrame(
        dimension = Int[], 
        target = String[], 
        sampler = String[],
        seed = Int[],
        medESS = Float64[]
    )
    for i in 1:d
        res[!, "ESS$(i)"] = Float64[]
        res[!, "mean$(i)"] = Float64[]
        res[!, "var$(i)"] = Float64[]
        res[!, "median$(i)"] = Float64[]
        res[!, "Q1_$(i)"] = Float64[]
        res[!, "Q3_$(i)"] = Float64[]
    end

    for j in eachindex(targets)
        @show target_names[j]
        sampler = :RRWMH
        for seed in 1:10
            @show seed
            replica = create_replica(seed,d,targets[j];scheme=sampler)
            expl = RRWMH(d,100)
            chain = run!(expl,replica,50000,targets[j],RRWMH_update!)
            res = vcat(res,record_stats(target_names[j],d,sampler,seed,chain))
        end
        sampler = :AM
        for seed in 1:10
            replica = create_replica(seed,d,targets[j];scheme=sampler)
            expl = AM(d)
            chain = run!(expl,replica,50000,targets[j],AM_update!)
            res = vcat(res,record_stats(target_names[j],d,sampler,seed,chain[10000:end,:,:]))    
        end
    end

    return res
end

# function to record chain stats
function record_stats(target_name,d,sampler,seed,chain)
    res = DataFrame(
        dimension = d, 
        target = target_name, 
        sampler = sampler,
        seed = seed,
        medESS = median(ess(chain).nt.ess)
    )
    ESS_vec = ess(chain).nt.ess                           # ess vector
    mean_vec = mean(chain).nt.mean                        # mean vector
    sqmean_vec = mean(Chains(Array(chain).^2)).nt.mean    # 2nd moment vector
    quantile_vec = quantile(chain,q = [0.25, 0.5, 0.75])  # quantiles
    for i in 1:d
        res[!, "ESS$(i)"] = [ESS_vec[i]]
        res[!, "mean$(i)"] = [mean_vec[i]]
        res[!, "var$(i)"] = [sqmean_vec[i]-mean_vec[i]^2]
        res[!, "median$(i)"] = [quantile_vec.nt.var"50.0%"[i]]
        res[!, "Q1_$(i)"] = [quantile_vec.nt.var"25.0%"[i]]
        res[!, "Q3_$(i)"] = [quantile_vec.nt.var"75.0%"[i]]
    end

    return res
end

# exact sampling function
function Exact_sampler(dists, d, dist_names, ismodel = false)
    # ESS recording is just to match format with MCMC samplers
    res = DataFrame(
        dimension = Int[], 
        target = String[], 
        sampler = String[],
        seed = Int[],
        medESS = Float64[]
    )
    for i in 1:d
        res[!, "ESS$(i)"] = Float64[]
        res[!, "mean$(i)"] = Float64[]
        res[!, "var$(i)"] = Float64[]
        res[!, "median$(i)"] = Float64[]
        res[!, "Q1_$(i)"] = Float64[]
        res[!, "Q3_$(i)"] = Float64[]
    end

    sampler = :exact
    for j in eachindex(dists)
        seed = 1
        rng = SplittableRandom(seed)
        if ismodel
            chain = sample(rng, dists[j], Prior(), 100000)
        else
            chain = Chains(rand(rng,dists[j], 100000)')
        end
        res = vcat(res,record_stats(dist_names[j],d,sampler,seed,chain))
    end

    return res
end


# kde of first 2 dimension of the chain 
function chain_2d_kde(target, d, sampler, ismodel = false)
    if sampler == :AM
        replica = create_replica(1,d,target;scheme=sampler)
        expl = AM(d)
        chain = run!(expl,replica,50000,target,AM_update!)
        return kde((vec(chain[10000:end,1,:]),vec(chain[10000:end,2,:])))
    elseif sampler == :RRWMH
        replica = create_replica(1,d,target;scheme=sampler)
        expl = RRWMH(d,100)
        chain = run!(expl,replica,50000,target,RRWMH_update!)
        return kde((vec(chain[10000:end,1,:]),vec(chain[10000:end,2,:])))
    elseif sampler == :exact
        if ismodel
            @show target
            chain = sample(SplittableRandom(1), target, Prior(), 100000)
        else
            chain = Chains(rand(SplittableRandom(1), target, 100000)')
        end
        return kde((vec(chain[:,1,:]),vec(chain[:,2,:])))
    else
        error()
    end
end


