using Test, Random
using Base.Iterators: product
using DecisionProgramming

Random.seed!(111)

"""Generate random probabilities"""
function random_probabilites(states, state)
    X = zeros([states; state]...)
    for s in paths(states)
        x = rand(state)
        x = x / sum(x)
        for s_j in 1:state
            X[[[s...]; s_j]...] = x[s_j]
        end
    end
    return X
end

"""Generate random utilities"""
function random_consequences(states)
    return reshape(rand(prod(states)), states...)
end


C = [2, 4]
D = [1, 3]
V = [5]
A = [1 => 3, 1 => 4, 2 => 3, 2 => 4, 3 => 5, 4 => 5]
S_j = [2, 2, 3, 3]

specs = Specs()
diagram = InfluenceDiagram(C, D, V, A, S_j)
I_j = diagram.I_j

X = Dict{Int, Array{Float64}}(
    i => random_probabilites(S_j[I_j[i]], S_j[i])
    for i in C)

Y = Dict{Int, Array{Float64}}(
    i => random_consequences(S_j[I_j[i]])
    for i in V)

params = Params(diagram, X, Y)

model = DecisionModel(specs, diagram, params)

@test true
