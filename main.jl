using Pkg
Pkg.instantiate()
Pkg.precompile()

using StatsPlots
using KernelDensity
using Plots

include("RWMH.jl")
include("targets.jl")

function no_update!(explorer, replica) end
function lp_normal(x) log(pdf(Normal(3,1),x)[1]+pdf(Normal(-2,2),x)[1]) end

expl = RWMH_sampler(Dirac(1),1)


new_state_vec = Vector{Float64}(undef, 10000)
init_state = 0.0
for i in 1:10000
    replica = create_replica(i,1,lp_normal,init_state)
    step!(expl, replica, lp_normal, no_update!)
    new_state_vec[i] = replica.state
end
histogram(new_state_vec,normalize=:pdf,ylim=(0,3.3),xlim=(-2,3))
histogram(0 .+ randn(10000),normalize=:pdf,ylim=(0,3.3),xlim=(-2,3))

init_state = 1.0
replica = create_replica(1,1,lp_normal,init_state)
step!(expl, replica, lp_normal, no_update!)

chain = run!(expl,replica,10000,lp_normal,no_update!)


histogram(chain[:,1,:])

plot(-5:0.01:5,lp_normal.(-5:0.01:5))
@gif for i in 1:500
    plot(1:(20*i),chain[1:(20*i),1,:],
    xlim=(1,10000), ylim=(-7,7),
    legend=false, xlab = "Iteration")
end

@gif for i in 1:500
    plot(-10:0.01:9,exp.(lp_normal.(-10:0.01:9)) ./ 2)
    density!(chain[1:(20*i),1,:],
    xlim=(-10,9), ylim=(0,0.23),
    legend=false)
end

dens = kde((x[1,:],x[2,:]))
plot(dens)
