include("RWMH.jl")
include("targets.jl")
include("update_functions.jl")
include("utils.jl")

using Plots.PlotMeasures

## Normal targets

# 1-d Normal
lp_vec = fill(log_density_generator(Normal(2,1)),9)
lp_vec[1] = log_density_generator(Normal(2,1))
lp_vec[2] = log_density_generator(Normal(0.2,1))
lp_vec[3] = log_density_generator(Normal(10,1))
lp_vec[4] = log_density_generator(Normal(2,0.2))
lp_vec[5] = log_density_generator(Normal(0.2,0.2))
lp_vec[6] = log_density_generator(Normal(10,0.2))
lp_vec[7] = log_density_generator(Normal(2,5))
lp_vec[8] = log_density_generator(Normal(0.2,5))
lp_vec[9] = log_density_generator(Normal(10,5))

dist_vec = fill(Normal(2,1),9)
dist_vec[1] = Normal(2,1)
dist_vec[2] = Normal(0.2,1)
dist_vec[3] = Normal(10,1)
dist_vec[4] = Normal(2,0.2)
dist_vec[5] = Normal(0.2,0.2)
dist_vec[6] = Normal(10,0.2)
dist_vec[7] = Normal(2,5)
dist_vec[8] = Normal(0.2,5)
dist_vec[9] = Normal(10,5)

target_names = ["N(2,1)", "N(0.2,1)", "N(10,1)",
                "N(2,0.2²)", "N(0.2,0.2²)", "N(10,0.2²)",
                "N(2,5²)", "N(0.2,5²)", "N(10,5²)"]

res_1d = Run_sampler(lp_vec, 1, target_names)
res_1d = vcat(res_1d,Run_sampler([log_density_generator(Cauchy())], 1, ["Cauchy"]))
# add ground truth
for i in 1:9
    quantile_vec = quantile(dist_vec[i],[0.25,0.5,0.75])
    push!(res_1d, (; 
        dimension = 1, 
        target = target_names[i],
        sampler = :exact,
        seed = 1,
        medESS = 0.0,
        ESS1 = 0.0,
        mean1 = mean(dist_vec[i]),
        var1 = var(dist_vec[i]),
        median1 = quantile_vec[2],
        Q1_1 = quantile_vec[1],
        Q3_1 = quantile_vec[3]
    ))
end
quantile_vec = quantile(Cauchy(),[0.25,0.5,0.75])
push!(res_1d, (; 
    dimension = 1, 
    target = "Cauchy",
    sampler = :exact,
    seed = 1,
    medESS = 0.0,
    ESS1 = 0.0,
    mean1 = Inf, # undefined but set to Inf
    var1 = Inf, # undefined but set to Inf
    median1 = quantile_vec[2],
    Q1_1 = quantile_vec[1],
    Q3_1 = quantile_vec[3]
))

CSV.write("deliverables/result_1d.csv", res_1d; quotestrings = true)

df = res_1d[res_1d.sampler .!= :exact,:]
p = groupedboxplot(df.target, df.medESS, group = df.sampler,
xrotation = 45,bar_width=0.7,size = (800, 400),left_margin = [5mm 0mm])
xlabel!("target")
ylabel!("ESS")
savefig(p,"ESS_1d.png")


# check estimation error for mean, var and different quantiles
df_exact = res_1d[res_1d.sampler .== :exact,:]
df_grouped = combine(groupby(df, [:target, :sampler])) do sdf
    DataFrame(target = sdf.target[1], sampler = sdf.sampler[1], 
    mean_err = abs(median(sdf.mean1) - df_exact[df_exact.target .== sdf.target[1],:].mean1[1]), 
    var_err = abs(median(sdf.var1) - df_exact[df_exact.target .== sdf.target[1],:].var1[1]), 
    median_err = abs(median(sdf.median1) - df_exact[df_exact.target .== sdf.target[1],:].median1[1]), 
    Q1_err = abs(median(sdf.Q1_1) - df_exact[df_exact.target .== sdf.target[1],:].Q1_1[1]), 
    Q3_err = abs(median(sdf.Q3_1) - df_exact[df_exact.target .== sdf.target[1],:].Q3_1[1]))
end
CSV.write("deliverables/validity_1d.csv", df_grouped; quotestrings = true)


# 2-d Normal
lp_vec = fill(log_density_generator(MvNormal([2,3],[0.25 0.1; 0.1 4])),4)
lp_vec[1] = log_density_generator(MvNormal([2,3],[0.25 0.1; 0.1 4]))
lp_vec[2] = log_density_generator(MvNormal([2,3],[0.25 -0.3; -0.3 4]))
lp_vec[3] = log_density_generator(MvNormal([2,3],[0.25 0.5; 0.5 4]))
lp_vec[4] = log_density_generator(MvNormal([2,3],[0.25 -0.8; -0.8 4]))

dist_vec = [
    MvNormal([2,3],[0.25 0.1; 0.1 4]),
    MvNormal([2,3],[0.25 -0.3; -0.3 4]),
    MvNormal([2,3],[0.25 0.5; 0.5 4]),
    MvNormal([2,3],[0.25 -0.8; -0.8 4])
]

target_names = ["ρ=0.1", "ρ=-0.3", "ρ=0.5", "ρ=-0.8"]

res_2d = Run_sampler(lp_vec, 2, target_names)
# add ground truth
for i in 1:4
    mean_vec = mean(dist_vec[i])
    var_vec = var(dist_vec[i])
    quantile_vec1 = quantile(Normal(mean_vec[1],sqrt(var_vec[1])),[0.25,0.5,0.75])
    quantile_vec2 = quantile(Normal(mean_vec[2],sqrt(var_vec[2])),[0.25,0.5,0.75])
    push!(res_2d, (; 
        dimension = 2, 
        target = target_names[i],
        sampler = :exact,
        seed = 1,
        medESS = 0.0,
        ESS1 = 0.0,
        mean1 = mean_vec[1],
        var1 = var_vec[1],
        median1 = quantile_vec1[2],
        Q1_1 = quantile_vec1[1],
        Q3_1 = quantile_vec1[3],
        ESS2 = 0.0,
        mean2 = mean_vec[2],
        var2 = var_vec[2],
        median2 = quantile_vec2[2],
        Q1_2 = quantile_vec2[1],
        Q3_2 = quantile_vec2[3]
    ))
end

CSV.write("deliverables/result_2d.csv", res_2d; quotestrings = true)

df = res_2d[res_2d.sampler .!= :exact,:]
p = groupedboxplot(res_2d.target, res_2d.medESS, group = res_2d.sampler,
bar_width=0.7,size = (800, 400),left_margin = [5mm 0mm])
ylabel!("Median ESS")
savefig(p,"ESS_2d.png")

# check estimation error for mean, var and different quantiles
df_exact = res_2d[res_2d.sampler .== :exact,:]
df_grouped = combine(groupby(df, [:target, :sampler])) do sdf
    DataFrame(target = sdf.target[1], sampler = sdf.sampler[1], 
    mean1_err = abs(median(sdf.mean1) - df_exact[df_exact.target .== sdf.target[1],:].mean1[1]), 
    var1_err = abs(median(sdf.var1) - df_exact[df_exact.target .== sdf.target[1],:].var1[1]), 
    median1_err = abs(median(sdf.median1) - df_exact[df_exact.target .== sdf.target[1],:].median1[1]), 
    Q11_err = abs(median(sdf.Q1_1) - df_exact[df_exact.target .== sdf.target[1],:].Q1_1[1]), 
    Q31_err = abs(median(sdf.Q3_1) - df_exact[df_exact.target .== sdf.target[1],:].Q3_1[1]),
    mean2_err = abs(median(sdf.mean2) - df_exact[df_exact.target .== sdf.target[1],:].mean2[1]), 
    var2_err = abs(median(sdf.var2) - df_exact[df_exact.target .== sdf.target[1],:].var2[1]), 
    median2_err = abs(median(sdf.median2) - df_exact[df_exact.target .== sdf.target[1],:].median2[1]), 
    Q12_err = abs(median(sdf.Q1_2) - df_exact[df_exact.target .== sdf.target[1],:].Q1_2[1]), 
    Q32_err = abs(median(sdf.Q3_2) - df_exact[df_exact.target .== sdf.target[1],:].Q3_2[1]))
end
CSV.write("deliverables/validity_2d.csv", df_grouped; quotestrings = true)


## Normal mixture

res_mix = Run_sampler([norm_mix_lpdf], 2, ["Normal mixture"])
res_mix_exact = Exact_sampler([norm_mix_target], 2, ["Normal mixture"])
res_mix = vcat(res_mix,res_mix_exact)
CSV.write("deliverables/result_mix.csv", res_mix; quotestrings = true)

# check estimation error for mean, var and different quantiles
df_exact = res_mix[res_mix.sampler .== :exact,:]
df = res_mix[res_mix.sampler .!= :exact,:]
df_grouped = combine(groupby(df, :sampler)) do sdf
    DataFrame(target = sdf.target[1], sampler = sdf.sampler[1], 
    mean1_err = abs(median(sdf.mean1) - df_exact[df_exact.target .== sdf.target[1],:].mean1[1]), 
    var1_err = abs(median(sdf.var1) - df_exact[df_exact.target .== sdf.target[1],:].var1[1]), 
    median1_err = abs(median(sdf.median1) - df_exact[df_exact.target .== sdf.target[1],:].median1[1]), 
    Q11_err = abs(median(sdf.Q1_1) - df_exact[df_exact.target .== sdf.target[1],:].Q1_1[1]), 
    Q31_err = abs(median(sdf.Q3_1) - df_exact[df_exact.target .== sdf.target[1],:].Q3_1[1]),
    mean2_err = abs(median(sdf.mean2) - df_exact[df_exact.target .== sdf.target[1],:].mean2[1]), 
    var2_err = abs(median(sdf.var2) - df_exact[df_exact.target .== sdf.target[1],:].var2[1]), 
    median2_err = abs(median(sdf.median2) - df_exact[df_exact.target .== sdf.target[1],:].median2[1]), 
    Q12_err = abs(median(sdf.Q1_2) - df_exact[df_exact.target .== sdf.target[1],:].Q1_2[1]), 
    Q32_err = abs(median(sdf.Q3_2) - df_exact[df_exact.target .== sdf.target[1],:].Q3_2[1]))
end
CSV.write("deliverables/validity_mix.csv", df_grouped; quotestrings = true)


dens_exact = chain_2d_kde(norm_mix_target, 2, :exact, false)
dens_AM = chain_2d_kde(norm_mix_lpdf, 2, :AM, false)
dens_RRWMH = chain_2d_kde(norm_mix_lpdf, 2, :RRWMH, false)
p = plot(dens_exact,xlims = (-10,5),ylims = (-7,7), title = "Exact")
scatter!([1,-3,2],[2,-1,-4], label = "true mean locations")
savefig(p,"exact_mix.png")
p = plot(dens_AM,xlims = (-10,5),ylims = (-7,7), title = "AM")
scatter!([1,-3,2],[2,-1,-4], legend = false)
savefig(p,"AM_mix.png")
p = plot(dens_RRWMH,xlims = (-10,5),ylims = (-7,7), title = "RRWMH")
scatter!([1,-3,2],[2,-1,-4], legend = false)
savefig(p,"RRWMH_mix.png")



## Banana

res_banana = Run_sampler([banana_lpdf], 5, ["Banana"])
res_banana_exact = Exact_sampler([b], 5, ["Banana"], true)
res_banana = vcat(res_banana,res_banana_exact)
CSV.write("deliverables/result_banana.csv", res_banana; quotestrings = true)


dens_exact = chain_2d_kde(b, 5, :exact, true)
dens_AM = chain_2d_kde(banana_lpdf, 5, :AM, true)
dens_RRWMH = chain_2d_kde(banana_lpdf, 5, :RRWMH, true)
p = plot(dens_exact,xlims = (-20,20),ylims = (-50,350), title = "Exact")
savefig(p,"exact_banana.png")
p = plot(dens_AM,xlims = (-5,5),ylims = (-5,35), title = "AM")
savefig(p,"AM_banana.png")
p = plot(dens_RRWMH,xlims = (-5,5),ylims = (-5,35), title = "RRWMH")
savefig(p,"RRWMH_banana.png")


## Logistic regression
res_lr = Run_sampler([lr_lpdf], 4, ["Logistic regression"])
CSV.write("deliverables/result_logistic.csv", res_lr; quotestrings = true)

# store summary of results
df = res_lr
df_grouped = combine(groupby(df, :sampler)) do sdf
    DataFrame(target = sdf.target[1], sampler = sdf.sampler[1], 
    mean1 = median(sdf.mean1), 
    var1 = median(sdf.var1), 
    median1 = median(sdf.median1), 
    mean2 = median(sdf.mean2), 
    var2 = median(sdf.var2), 
    median2 = median(sdf.median2), 
    mean3 = median(sdf.mean3), 
    var3 = median(sdf.var3), 
    median3 = median(sdf.median3), 
    mean4 = median(sdf.mean4), 
    var4 = median(sdf.var4), 
    median4 = median(sdf.median4))
end
CSV.write("deliverables/summary_lr.csv", df_grouped; quotestrings = true)

## ESS plot for complex targets
a = vcat(res_mix[1:(end-1),:].target, res_banana[1:(end-1),:].target, res_lr[1:(end-1),:].target)
b = vcat(res_mix[1:(end-1),:].ESS1, res_banana[1:(end-1),:].ESS1, res_lr[1:(end-1),:].ESS1)
c = vcat(res_mix[1:(end-1),:].sampler, res_banana[1:(end-1),:].sampler, res_lr[1:(end-1),:].sampler)
p = groupedboxplot(a, b, group = c, yaxis=:log, legend=:bottomright)
ylabel!("Median ESS")
savefig(p,"ESS_complex.png")
