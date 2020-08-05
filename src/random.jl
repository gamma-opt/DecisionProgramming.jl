using Random, Parameters

"""Create random influence diagram of given size.

# Examples
```julia
import Random
rng = MersenneTwister(1)
n_C, n_D, n_V = 4, 2, 1  # Number of chance, decision and value states
n_A = 2  # Upper limit on information set size for any node
num_states = [2, 3]  # Set of possible number of states
G = InfluenceDiagram(rng, n_C, n_D, n_V, n_A, num_states)
```
"""
function InfluenceDiagram(rng::AbstractRNG, n_C::Int, n_D::Int, n_V::Int, n_A::Int, num_states::Vector{Int})
    # Create nodes
    n = n_C + n_D
    u = shuffle(rng, 1:n)
    C = u[1:n_C]
    D = u[n_C+1:end]
    V = collect((n+1):(n+n_V))

    # Create arcs between chance and decision nodes.
    A = Vector{Pair{Node, Node}}()
    for i in 1:(n-1)
        js = unique(rand(rng, (i+1):n, rand(rng, 1:n_A)))
        append!(A, i => j for j in js)
    end

    # Create arcs between from nodes chance and decision nodes to value nodes.
    for v in V
        is = unique(rand(rng, 1:(n-1), rand(rng, 0:n_A)))
        append!(A, i => v for i in is)
    end
    # There should be atleast one arc from node n to V
    append!(A, n => v for v in unique(rand(rng, V, rand(rng, 1:n_A))))

    # Create states
    S_j = rand(rng, num_states, n)

    return InfluenceDiagram(C, D, V, A, S_j)
end

function random_probability(rng::AbstractRNG, states::Vector{State}, state::State)
    X = zeros([states; state]...)
    for s in paths(states)
        x = rand(rng, state)
        x = x / sum(x)
        for s_j in 1:state
            X[[[s...]; s_j]...] = x[s_j]
        end
    end
    return X
end

"""Generate random probabilities for an influence diagram.

# Examples
```julia
import Random
rng = MersenneTwister(1)
X = Probabilities(rng, G)
```
"""
function Probabilities(rng::AbstractRNG, G::InfluenceDiagram)
    @unpack C, S_j, I_j = G
    Probabilities(Dict(i => random_probability(rng, S_j[I_j[i]], S_j[i]) for i in C))
end

scale(x::Float64, low::Float64, high::Float64) = x * (high - low) + low

function random_consequence(rng::AbstractRNG, states::Vector{State}, low::Float64, high::Float64)
    x = rand(rng, prod(states))
    x = scale.(x, low, high)
    return reshape(x, states...)
end

"""Generate random consequences for an influence diagram.

# Examples
```julia
import Random
rng = MersenneTwister(1)
Y = Consequences(rng, G; low = -1.0, high = 1.0)
```
"""
function Consequences(rng::AbstractRNG, G::InfluenceDiagram; low::Float64 = -1.0, high::Float64 = 1.0)
    @unpack V, S_j, I_j = G
    high > low || error("")
    Consequences(Dict(i => random_consequence(rng, S_j[I_j[i]], low, high) for i in V))
end

# function random_decision_strategy(rng::AbstractRNG, states::Vector{State}, state::State)
#     # TODO:
# end

# function DecisionStrategy(rng::AbstractRNG, G::InfluenceDiagram)
#     # TODO:
# end
