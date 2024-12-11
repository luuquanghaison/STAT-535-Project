
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
s5 = rand(Gamma(0.2^2,5^2),100000) # mean = 1, var = 5²
x5 = rand(Normal(),100000) .* sqrt.(s5)

p = density(x4, label = "mean = 1, var = 0.2²")
density!(x1, label = "mean = 1, var = 1", line = :dash)
density!(x5, label = "mean = 1, var = 5²", line = :dashdot)
density!(rand(Normal(0,1),100000), label = "Fixed stepsize = 1", line = :dot)
xlims!(-10,10)
savefig(p,"proposal_behaviour_var.png")


p = density(x2, label = "mean = 0.2, var = 1")
density!(x1, label = "mean = 1, var = 1", line = :dash)
density!(x3, label = "mean = 5, var = 1", line = :dashdot)
density!(rand(Normal(0,1),100000), label = "Fixed stepsize = 1", line = :dot)
xlims!(-10,10)
savefig(p,"proposal_behaviour_mean.png")
