using Printf, Parameters
using JuMP, Gurobi
using DecisionProgramming

# Parameters
no_forgetting = false
const N = 4
const health = [3*k - 2 for k in 1:N]
const test = [3*k - 1 for k in 1:(N-1)]
const treat = [3*k for k in 1:(N-1)]
const cost = [(3*N - 2) + k for k in 1:(N-1)]
const price = [(3*N - 2) + N]

# Influence diagram parameters
C = health ∪ test
D = treat
V = cost ∪ price
A = Vector{Pair{Int, Int}}()
S_j = Vector{Int}(undef, length(C) + length(D))

# Construct arcs
add_arcs(from, to) = append!(A, (i => j for (i, j) in zip(from, to)))
add_arcs(health[1:end-1], health[2:end])
add_arcs(health[1:end-1], test)
add_arcs(treat, health[2:end])
add_arcs(treat, cost)
add_arcs(health[end], price)
if no_forgetting
    append!(A, (test[k] => treat[k2] for k in 1:(N-1) for k2 in k:(N-1)))
    append!(A, (treat[k] => treat[k2] for k in 1:((N-1)-1) for k2 in (k+1):(N-1)))
else
    add_arcs(test, treat)
end

# Construct states
S_j[health] = fill(2, length(health))
S_j[test] = fill(2, length(test))
S_j[treat] = fill(2, length(treat))

# Probabilities
function probabilities(health, treat, test, S_j)
    X = Dict{Int, Array{Float64}}()

    # h_1
    begin
        i = health[1]
        p = zeros(S_j[i])
        p[1] = 0.1
        p[2] = 1.0 - p[1]
        X[i] = p
    end

    # h_i, i≥2
    for (i, j, k) in zip(health[1:end-1], treat, health[2:end])
        p = zeros(S_j[i], S_j[j], S_j[k])
        p[2, 2, 1] = 0.2
        p[2, 2, 2] = 1.0 - p[2, 2, 1]
        p[2, 1, 1] = 0.1
        p[2, 1, 2] = 1.0 - p[2, 1, 1]
        p[1, 2, 1] = 0.9
        p[1, 2, 2] = 1.0 - p[1, 2, 1]
        p[1, 1, 1] = 0.5
        p[1, 1, 2] = 1.0 - p[1, 1, 1]
        X[k] = p
    end

    # t_i
    for (i, j) in zip(health, test)
        p = zeros(S_j[i], S_j[j])
        p[1, 1] = 0.8
        p[1, 2] = 1.0 - p[1, 1]
        p[2, 2] = 0.9
        p[2, 1] = 1.0 - p[2, 2]
        X[j] = p
    end
    return X
end

# Consequences
function consequences(cost, price)
    Y = Dict{Int, Array{Float64}}()
    for i in cost
        Y[i] = [-100, 0]
    end
    for i in price
        Y[i] = [300, 1000]
    end
    return Y
end

X = @time probabilities(health, treat, test, S_j)
Y = @time consequences(cost, price)


# Model
specs = Specs(probability_sum_cut=false)
diagram = @time InfluenceDiagram(C, D, V, A, S_j)
params = @time Params(diagram, X, Y)
model = @time DecisionModel(specs, diagram, params)

println("--- Optimization ---")
optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    # "LazyConstraints" =>    1,
)
set_optimizer(model, optimizer)
optimize!(model)

πval = value.(model[:π])
print_results(πval, diagram, params; πtol=0.1)

println("State probabilities:")
probs = state_probabilities(πval, diagram)
print_state_probabilities(probs, health, ["ill", "healthy"])
print_state_probabilities(probs, test, ["positive", "negative"])
print_state_probabilities(probs, treat, ["treat", "pass"])
println()

# Conditional state probabilities when pig is treat or not treated.
for node in treat
    for state in 1:2
        fixed = Dict(node => state)
        prior = probs[node][state]
        (isapprox(prior, 0, atol=1e-4) | isapprox(prior, 1, atol=1e-4)) && continue
        println("Conditional state probabilities")
        probs2 = state_probabilities(πval, diagram, prior, fixed)
        print_state_probabilities(probs2, health, ["ill", "healthy"], fixed)
        print_state_probabilities(probs2, test, ["positive", "negative"], fixed)
        print_state_probabilities(probs2, treat, ["treat", "pass"], fixed)
        println()
    end
end

println("Decision strategy")
z = model[:z]
for i in D
    z_i = value.(z[i])
    println(z_i)
end

# using Plots
# x, y = cumulative_distribution(πval, diagram, params)
# p = plot(x, y, linestyle=:dash)
# savefig(p, "pig-breeding.svg")
