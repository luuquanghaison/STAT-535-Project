
using StatsPlots

include("RWMH.jl")
include("targets.jl")
include("update_functions.jl")

###############################################################################
# Random step effects on proposals
###############################################################################


s1 = rand(Gamma(1,1),100000) # mean = 1, var = 1
x1 = rand(Normal(),100000) .* sqrt.(s1)
s2 = rand(Gamma(0.2^2,5),100000) # mean = 0.2, var = 1
x2 = rand(Normal(),100000) .* sqrt.(s2)
s3 = rand(Gamma(5^2,0.2),100000) # mean = 5, var = 1
x3 = rand(Normal(),100000) .* sqrt.(s3)
s4 = rand(Gamma(5^2,0.2^2),100000) # mean = 1, var = 0.2²
x4 = rand(Normal(),100000) .* sqrt.(s4)
s5 = rand(Gamma(1,0.2),100000) # mean = 0.2, var = 0.2²
x5 = rand(Normal(),100000) .* sqrt.(s5)
s6 = rand(Gamma(5^4,0.2^3),100000) # mean = 5, var = 0.2²
x6 = rand(Normal(),100000) .* sqrt.(s6)
s7 = rand(Gamma(0.2^2,5^2),100000) # mean = 1, var = 5²
x7 = rand(Normal(),100000) .* sqrt.(s7)
s8 = rand(Gamma(0.2^4,5^3),100000) # mean = 0.2, var = 5²
x8 = rand(Normal(),100000) .* sqrt.(s8)
s9 = rand(Gamma(1,5),100000) # mean = 5, var = 5²
x9 = rand(Normal(),100000) .* sqrt.(s9)

p = density(x1, label = "mean = 1, var = 1")
density!(x4, label = "mean = 1, var = 0.2²", line = :dash)
density!(x7, label = "mean = 1, var = 5²", line = :dashdot)
density!(rand(Normal(0,1),100000), label = "Fixed stepsize = 1", line = :dot)
xlims!(-10,10)
savefig(p,"proposal_behaviour1.png")



density(x2, label = "mean = 0.2, var = 1")
density!(x5, label = "mean = 0.2, var = 0.2²", line = :dash)
density!(x8, label = "mean = 0.2, var = 5²", line = :dashdot)
density!(rand(Normal(0,sqrt(0.2)),100000), label = "Fixed stepsize = 0.2", line = :dot)
xlims!(-10,10)

density(x3, label = "mean = 5, var = 1")
density!(x6, label = "mean = 5, var = 0.2²", line = :dash)
density!(x9, label = "mean = 5, var = 5²", line = :dashdot)
density!(rand(Normal(0,sqrt(5)),100000), label = "Fixed stepsize = 5", line = :dot)
xlims!(-10,10)


###############################################################################
# 1 step behaviour of RRWMH
###############################################################################

lp_normal = log_density_generator(MixtureModel(Normal, [(-3,1), (3,1)], [1/2, 1/2]))

RRWMH_step = []
expl_RRWMH = RRWMH(1,[1],[4;;],[Gamma(2/1,1)],100)
for i in 1:10000
    replica = create_replica(i,1,lp_normal;init_state=[-3.0],scheme=:RRWMH)
    step!(expl_RRWMH,replica,lp_normal,100)
    push!(RRWMH_step,replica.state[1])
end

RWMH_step = []
expl_RWMH = RWMH(1,4)
for i in 1:10000
    replica = create_replica(i,1,lp_normal;init_state=[-3.0],scheme=:RRWMH)
    step!(expl_RWMH,replica,lp_normal,100)
    push!(RWMH_step,replica.state[1])
end

histogram(RRWMH_step, label = "RRWMH step", normalize=:pdf)
density!(rand(Normal(1,2),100000),label = "true density")
histogram(RWMH_step, label = "fixed step", normalize=:pdf, color=2)
density!(rand(Normal(1,2),100000),label = "true density")