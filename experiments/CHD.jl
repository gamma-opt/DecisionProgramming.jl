using DelimitedFiles, Plots, Statistics
t1 = zeros(101)
t2 = zeros(101)
val1 = zeros(101)
val2 = zeros(101)
for i in 0:100
    try
        result = readdlm((@__DIR__)*"/results/CHD_"*string(i)*".csv", ',')
        t1[i+1] = result[2]
        val1[i+1] = result[1]
    catch e
        println("Missing file "*(@__DIR__)*"/results/CHD_"*string(i)*".csv")
    end
    try
        result = readdlm((@__DIR__)*"/../../DecProg_0_1_0/experiments/results/CHD_"*string(i+1)*".csv", ',')
        t2[i+1] = result[2]
        val2[i+1] = result[1]
        # println(val1[i+1]*1000/result[1])
    catch e
        t2[i+1] = 7200
        val2[i+1] = Inf
    end
end

#println(sort(t1))
#println(sort(t2))

# log-axis, all data
# scatter(t1, t2, xlabel="v1.2.0", ylabel="v0.1.0", markershape=:x, label="Solution time (seconds)", xlim=[0.001,1800], ylim=[0.001,7200], xaxis=:log, yaxis=:log)
# plot!([0.001,7200], [0.001,7200], ls=:dash, lc=:black, label=false)

# log-axis, >1s
scatter(t1[t2.<7200], t2[t2.<7200], xlabel="(18)-(23)", ylabel="(5)-(10)", markershape=:x, label="Solution time (seconds)", xlim=[1,7200], ylim=[1,7200], legend=:right, xaxis=:log, yaxis=:log, yticks=[10^k for k in 0:3], xticks=[10^k for k in 0:3])
scatter!(t1[t2.>=7200], t2[t2.>=7200], markershape=:x, label=false)
plot!([1,7200], [1,7200], ls=:dash, lc=:black, label=false)
Plots.pdf("CHD_log")

# "regular" axis
scatter(t2[t2.<7200], t1[t2.<7200], ylabel="v1.2.0", xlabel="v0.1.0", markershape=:x, label="Solution time (seconds)", xlim=[0,7200], ylim=[0,3600], legend=:right, size=(1200,400))
scatter!(t2[t2.>=7200], t1[t2.>=7200], markershape=:x, label=false)
plot!([0,7200], [0,7200], ls=:dash, lc=:black, label=false)
Plots.pdf("CHD")