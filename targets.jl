using Distributions
using Turing
using LogDensityProblems
using DataFrames
using CSV
using FillArrays
using StatsFuns: logistic

# log density function generator for known distributions
function log_density_generator(distribution)
    return function log_density(x) 
        logpdf(distribution,x)[1]
    end
end


# Normal mixture target
norm_mix_lpdf = log_density_generator(
    MixtureModel(MvNormal, [([1,2],[1 0.5;0.5 4]), ([-3,-1],[9 2;2 1]), 
    ([2,-4],[0.25 -0.15;-0.15 0.49])], [1/2, 1/3, 1/6]))

# Banana target
@model function Banana()
    θ = Vector{Float64}(undef, 5)
    θ[1] ~ Normal(0,10)
    for i in 2:5
        θ[i] ~ Normal(θ[1]^2,0.1)
    end
end
b = Banana()
banana_func = LogDensityFunction(b)
banana_lpdf = function (θ)
    LogDensityProblems.logdensity(banana_func, θ) 
end

# Logistic regression target
base_folder = dirname(Base.active_project())
data = DataFrame(CSV.File(joinpath(base_folder, "data", "Social_Network_Ads.csv")))
# clean data
data.Gender = [s == "Male" ? 0 : 1 for s in data.Gender] # change gender to binary
data = data[:,2:5]                                       # remove id column
# standardize Age and EstimatedSalary
data.Age = (data.Age .- mean(data.Age)) ./ std(data.Age)
data.EstimatedSalary = (data.EstimatedSalary .- mean(data.EstimatedSalary)) ./ std(data.EstimatedSalary)
# model & density
@model function LR(x,y)
    n = length(y)
    β ~ MvNormal(Diagonal(fill(100,4)))
    for i in 1:n
        v = logistic(β[1] + β[2] * x[i, 1] + β[3] * x[i, 2] + β[4] * x[i, 3])
        y[i] ~ Bernoulli(v)
    end
end
lr = LR(data[:,1:3],data[:,4])
lr_func = LogDensityFunction(lr)
lr_lpdf = function (θ)
    LogDensityProblems.logdensity(lr_func, θ) 
end