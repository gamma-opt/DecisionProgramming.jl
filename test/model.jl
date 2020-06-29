using Test, Random
using Base.Iterators: product
using DecisionProgramming

Random.seed!(111)

"""Generate random probabilities"""
function random_probabilites(states, state)
    X = zeros([states; state]...)
    for p in product(UnitRange.(1, states)...)
        x = rand(state)
        x = x / sum(x)
        for s in 1:state
            X[[[p...]; s]...] = x[s]
        end
    end
    return X
end

"""Generate random utilities"""
function random_consequences(states, consequences_set)
    U = rand(consequences_set, prod(states))
    return reshape(U, states...)
end


C = [2, 4]
D = [1, 3]
V = [5]
A = [1 => 3, 1 => 4, 2 => 3, 2 => 4, 3 => 5, 4 => 5]
S_j = [2, 2, 3, 3]

specs = Specs()
graph = DecisionGraph(C, D, V, A, S_j)
I_j = graph.I_j

X = Dict{Int, Array{Float64}}(
    i => random_probabilites([S_j[j] for j in I_j[i]], S_j[i])
    for i in C)

num_utilities = sum(prod(S_j[j] for j in I_j[i]) for i in V)

Y = Dict{Int, Array{Int}}(
    i => random_consequences([S_j[j] for j in I_j[i]], 1:num_utilities)
    for i in V)

U = rand(num_utilities)

params = Params(graph, X, Y, U)

model = DecisionModel(specs, graph, params)

@test true
