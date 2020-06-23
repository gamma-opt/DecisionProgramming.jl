using Test
using Base.Iterators: product

include(joinpath(dirname(@__DIR__), "src", "model.jl"))


"""Generate random probabilities"""
function random_probabilites(states::AbstractArray, state::Int)
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
function random_utilities(states::AbstractArray)
    Y = rand(prod(states))
    return reshape(Y, states...)
end

C = [2, 4]
D = [1, 3]
V = [5]
A = [
    Pair(1, 3),
    Pair(1, 4),
    Pair(2, 3),
    Pair(2, 4),
    Pair(3, 5),
    Pair(4, 5)
]
S_j = [2, 2, 3, 3]

graph = DecisionGraph(C, D, V, A, S_j)
I_j = graph.I_j

X = Dict{Int, Array{Float64}}(
    i => random_probabilites([S_j[j] for j in I_j[i]], S_j[i])
    for i in C
)
probabilities = Probabilities(graph, X)

Y = Dict{Int, Array{Float64}}(
    i => random_utilities([S_j[j] for j in I_j[i]])
    for i in V
)
utilities = Utilities(graph, Y)

specs = Specs(lazy_constraints=false)
model = DecisionModel(specs, graph, probabilities, utilities)
