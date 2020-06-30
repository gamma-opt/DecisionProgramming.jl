using JuMP, Gurobi
using DecisionProgramming

health = [1, 4, 7, 10] # health of the pig
test = [2, 5, 8] # whether to test the pig
treat = [3, 6, 9] # whether to treat the pig
cost = [11, 12, 13] # treatment cost
price = [14] # sell price

C = health ∪ test
D = treat
V = cost ∪ price
A = Vector{Pair{Int, Int}}()
S_j = Vector{Int}()

# Construct arcs
add_arcs(from, to) = append!(A, (i => j for (i, j) in zip(from, to)))

add_arcs(health[1:end-1], health[2:end])
add_arcs(health[1:end-1], test)
add_arcs(treat, health[2:end])
add_arcs(treat, cost)
add_arcs(health[end], price)
# no-forgetting
n = length(test)
append!(A, (test[i] => treat[j] for i in 1:n for j in i:n))
append!(A, (treat[i] => treat[j] for i in 1:(n-1) for j in (i+1):n))

# Construct states
append!(S_j, fill(2, length(health)))
append!(S_j, fill(2, length(test)))
append!(S_j, fill(2, length(treat)))

# Probabilities


# Consequences


# Utilities


# Model
g = InfluenceDiagram(C, D, V, A, S_j)
specs = Specs()
# params = Params()
# model = DecisionModel(specs, g, params)
