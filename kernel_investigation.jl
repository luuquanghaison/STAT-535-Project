
using StatsPlots

include("RWMH.jl")

s = rand(Gamma(3,3),100000)
x = rand(Normal(),100000) .* s

density(s)
density(x)

s2 = rand(Gamma(1,3*sqrt(6)),100000)
x2 = rand(Normal(),100000) .* s2

density(s2)
density(x2)

s3 = rand(Gamma(9,1),100000)
x3 = rand(Normal(),100000) .* s3

density(s3)
density(x3)

s4 = rand(Pareto(2,1),100000)
x4 = rand(Normal(),100000) .* s4

density(s4)
density(x4,xlim=(-50,50))


