using Pkg
Pkg.instantiate()
Pkg.precompile()

using StatsPlots
using KernelDensity
using Plots

include("RWMH.jl")
include("targets.jl")
include("update_functions.jl")


## Normal targets

# 1-d Normal
lp_normal1 = log_density_generator(Normal(2,1))
lp_normal2 = log_density_generator(Normal(0.2,1))
lp_normal3 = log_density_generator(Normal(10,1))
lp_normal4 = log_density_generator(Normal(2,0.2))
lp_normal5 = log_density_generator(Normal(0.2,0.2))
lp_normal6 = log_density_generator(Normal(10,0.2))
lp_normal7 = log_density_generator(Normal(2,5))
lp_normal8 = log_density_generator(Normal(0.2,5))
lp_normal9 = log_density_generator(Normal(10,5))





# 2-d Normal
lp_mvnormal1 = log_density_generator(MvNormal([2,3],[0.25 0.1; 0.1 4]))
lp_mvnormal2 = log_density_generator(MvNormal([2,3],[0.25 -0.3; -0.3 4]))
lp_mvnormal3 = log_density_generator(MvNormal([2,3],[0.25 0.5; 0.5 4]))
lp_mvnormal4 = log_density_generator(MvNormal([2,3],[0.25 -0.8; -0.8 4]))

## Normal mixture



## Banana


## Logistic regression




lp_func = norm_mix_lpdf
d = 2

replica = create_replica(1,d,lp_func;scheme=:AM)
expl_AM = AM(d)
chain_AM = run!(expl_AM,replica,50000,lp_func,AM_update!)
@show ess(chain_AM[10000:end,:,:]).nt.ess

replica = create_replica(1,d,lp_func;scheme=:RRWMH)
expl_RRWMH = RRWMH(d,100)
chain_RRWMH = run!(expl_RRWMH,replica,50000,lp_func,RRWMH_update!)
@show ess(chain_RRWMH[10000:end,:,:]).nt.ess

dens = kde((vec(chain_AM[10000:end,1,:]),vec(chain_AM[10000:end,2,:])))
p = plot(dens,xlims = (-10,5),ylims = (-7,7), title = "AM, ESS = $(Int.(round.(ess(chain_AM[10000:end,:,:]).nt.ess)))")
scatter!([1,-3,2],[2,-1,-4], legend = false)#, label = "true mean locations")
savefig(p,"AM_mix.png")

dens = kde((vec(chain_RRWMH[10000:end,1,:]),vec(chain_RRWMH[10000:end,2,:])))
p = plot(dens,xlims = (-10,5),ylims = (-7,7), title = "RRWMH, ESS = $(Int.(round.(ess(chain_RRWMH[10000:end,:,:]).nt.ess)))")
scatter!([1,-3,2],[2,-1,-4], legend = false)# label = "true mean locations")
savefig(p,"RRWMH_mix.png")

exact_samples = rand(norm_mix_target,100000)
dens = kde((exact_samples[1,:],exact_samples[2,:]))
p = plot(dens,xlims = (-10,5),ylims = (-7,7), title = "Exact samples")
scatter!([1,-3,2],[2,-1,-4], label = "true mean locations")
savefig(p,"true_mix.png")



init_state = [-3.0]
lp_normal = log_density_generator(Normal(1,2))
replica = create_replica(1,1,lp_normal;init_state=init_state,scheme=:AM)

expl_AM = AM(1,[0.0],fill(1.0,(1,1)))
chain_AM = run!(expl_AM,replica,10000,lp_normal,AM_update!)

AM_step = []
for i in 1:10000
    replica = create_replica(i,1,lp_normal;init_state=[-3.0],scheme=:AM)
    step!(expl_AM,replica,lp_normal,1)
    push!(AM_step,replica.state[1])
end

histogram(AM_step)

init_state = [-3.0]
lp_normal = log_density_generator(Normal(1,2))
replica = create_replica(1,1,lp_normal;init_state=init_state,scheme=:RRWMH)

expl_RRWMH = RRWMH(1,[0.0],fill(1.0,(1,1)),[Gamma(1,1)],100)
chain_RRWMH = run!(expl_RRWMH,replica,10000,lp_normal,RRWMH_update!)

RRWMH_step = []
expl_RRWMH = RRWMH(1,[1.0709946326863373],[4.139626115703674;;],
[Gamma(4.139626115703674*2,1/2)],100)
for i in 1:10000
    replica = create_replica(i,1,lp_normal;init_state=[-3.0],scheme=:RRWMH)
    step!(expl_RRWMH,replica,lp_normal,1)
    push!(RRWMH_step,replica.state[1])
end

histogram(RRWMH_step)

a = []
for i in 101:10000
    push!(a,var(chain_AM[(i-100):i,1,:]))
end

b = []
for i in eachindex(a)
    push!(b,var(a[1:i]))
end