#=
using Random

"""
    function information_set(rng::AbstractRNG, j::Node, n_I::Int)

Generates random information sets for chance and decision nodes.
"""
function information_set(rng::AbstractRNG, j::Node, n_I::Int)
    m = min(rand(rng, 0:n_I), j-1)
    I_j = shuffle(rng, 1:(j-1))[1:m]
    return sort(I_j)
end

"""
    function information_set(rng::AbstractRNG, leaf_nodes::Vector{Node}, n::Int)

Generates random information sets for value nodes.
"""
function information_set(rng::AbstractRNG, leaf_nodes::Vector{Node}, n::Int)
    @assert n ≥ 1
    non_leaf_nodes = shuffle(rng, setdiff(1:n, leaf_nodes))
    l = length(non_leaf_nodes)
    if isempty(leaf_nodes)
        m = rand(rng, 1:l)
    else
        m = rand(rng, 0:l)
    end
    I_v = [leaf_nodes; non_leaf_nodes[1:m]]
    return sort(I_v)
end


"""
    random_diagram!(rng::AbstractRNG, diagram::InfluenceDiagram, n_C::Int, n_D::Int, n_V::Int, m_C::Int, m_D::Int, states::Vector{Int})

Generate random decision diagram with `n_C` chance nodes, `n_D` decision nodes, and `n_V` value nodes.
Parameter `m_C` and `m_D` are the upper bounds for the size of the information set.

# Arguments
- `rng::AbstractRNG`: Random number generator.
- `diagram::InfluenceDiagram`: The (empty) influence diagram structure that is filled by this function
- `n_C::Int`: Number of chance nodes.
- `n_D::Int`: Number of decision nodes.
- `n_V::Int`: Number of value nodes.
- `m_C::Int`: Upper bound for size of information set for chance nodes.
- `m_D::Int`: Upper bound for size of information set for decision nodes.
- `states::Vector{State}`: The number of states for each chance and decision node
    is randomly chosen from this set of numbers.


# Examples
```julia
rng = MersenneTwister(3)
diagram = InfluenceDiagram()
random_diagram!(rng, diagram, 5, 2, 3, 2, 2, [2,3])
```
"""
function random_diagram!(rng::AbstractRNG, diagram::InfluenceDiagram, n_C::Int, n_D::Int, n_V::Int, m_C::Int, m_D::Int, states::Vector{Int})
    n = n_C + n_D
    n_C ≥ 0 || throw(DomainError("There should be `n_C ≥ 0` chance nodes."))
    n_D ≥ 0 || throw(DomainError("There should be `n_D ≥ 0` decision nodes"))
    n ≥ 1 || throw(DomainError("There should be at least one chance or decision node `n_C+n_D≥1`."))
    n_V ≥ 1 || throw(DomainError("There should be `n_V ≥ 1` value nodes."))
    m_C ≥ 1 || throw(DomainError("Maximum size of information set should be `m_C ≥ 1`."))
    m_D ≥ 1 || throw(DomainError("Maximum size of information set should be `m_D ≥ 1`."))
    all(s > 1 for s in states) || throw(DomainError("Minimum number of states possible should be 2."))

    # Create node indices
    U = shuffle(rng, 1:n)
    diagram.C = [Node(c) for c in sort(U[1:n_C])]
    diagram.D = [Node(d) for d in sort(U[(n_C+1):n])]
    diagram.V = [Node(v) for v in collect((n+1):(n+n_V))]

    diagram.I_j = Vector{Vector{Node}}(undef, n+n_V)
    # Create chance and decision nodes
    for c in diagram.C
        diagram.I_j[c] = information_set(rng, c, m_C)
    end
    for d in diagram.D
        diagram.I_j[d] = information_set(rng, d, m_D)
    end


    # Assign each leaf node to a random value node
    leaf_nodes = setdiff(1:n, (diagram.I_j[c] for c in diagram.C)..., (diagram.I_j[d] for d in diagram.D)...)
    leaf_nodes_v = Dict(v=>Node[] for v in diagram.V)
    for j in leaf_nodes
        v = rand(rng, diagram.V)
        push!(leaf_nodes_v[v], j)
    end

    # Create values nodes
    for v in diagram.V
        diagram.I_j[v] = information_set(rng, leaf_nodes_v[v], n)
    end


    diagram.S = States(State[rand(rng, states, n)...])
    diagram.X = Vector{Probabilities}(undef, n_C)
    diagram.Y = Vector{Utilities}(undef, n_V)

    diagram.Names = ["$(i)" for i in 1:(n+n_V)]
    statelist = []
    for i in 1:n
        push!(statelist, ["$(j)" for j in 1:diagram.S[i]])
    end
    diagram.States = statelist

    for c in diagram.C
        random_probabilities!(rng, diagram, c)
    end

    for v in diagram.V
        random_utilities!(rng, diagram, v)
    end

    diagram.P = DefaultPathProbability(diagram.C, diagram.I_j[diagram.C], diagram.X)
    diagram.U = DefaultPathUtility(diagram.I_j[diagram.V], diagram.Y)

    return diagram
end

"""
    function random_probabilities!(rng::AbstractRNG, diagram::InfluenceDiagram, c::Node; n_inactive::Int=0)

Generate random probabilities for chance node `c`.

# Examples
```julia
rng = MersenneTwister(3)
diagram = InfluenceDiagram()
random_diagram!(rng, diagram, 5, 2, 3, 2, 2, [2,3])
c = diagram.C[1]
random_probabilities!(rng, diagram, c)
```
"""
function random_probabilities!(rng::AbstractRNG, diagram::InfluenceDiagram, c::Node; n_inactive::Int=0)
    if !(c in diagram.C)
        throw(DomainError("Probabilities can only be added for chance nodes."))
    end
    I_c = diagram.I_j[c]
    states = diagram.S[I_c]
    state = diagram.S[c]
    if !(0 ≤ n_inactive ≤ prod([states...; (state - 1)]))
        throw(DomainError("Number of inactive states must be < prod([S[I_j]...;, S[j]-1])"))
    end

    # Create the random probabilities
    data = rand(rng, states..., state)

    # Create inactive chance states
    if n_inactive > 0
        # Count of inactive states per chance stage.
        a = zeros(Int, states...)
        # There can be maximum of (state - 1) inactive chance states per stage.
        b = repeat(vec(CartesianIndices(a)), state - 1)
        # Uniform random sample of n_inactive states.
        r = shuffle(rng, b)[1:n_inactive]
        for s in r
            a[s] += 1
        end
        for s in CartesianIndices(a)
            indices = CartesianIndices(data[s, :])
            i = shuffle(rng, indices)[1:a[s]]
            data[s, i] .= 0.0
        end
    end

    # Normalize the probabilities
    for s in CartesianIndices((states...,))
        data[s, :] /= sum(data[s, :])
    end

    index_c = findfirst(j -> j==c, diagram.C)
    diagram.X[index_c] = Probabilities(c, data)
end

scale(x::Utility, low::Utility, high::Utility) = x * (high - low) + low

"""
    function random_utilities!(rng::AbstractRNG, diagram::InfluenceDiagram, v::Node; low::Float64=-1.0, high::Float64=1.0)

Generate random utilities between `low` and `high` for value node `v`.

# Examples
```julia
rng = MersenneTwister(3)
diagram = InfluenceDiagram()
random_diagram!(rng, diagram, 5, 2, 3, 2, 2, [2,3])
v = diagram.V[1]
random_utilities!(rng, diagram, v)
```
"""
function random_utilities!(rng::AbstractRNG, diagram::InfluenceDiagram, v::Node; low::Float64=-1.0, high::Float64=1.0)
    if !(v in diagram.V)
        throw(DomainError("Utilities can only be added for value nodes."))
    end
    if !(high > low)
        throw(DomainError("high should be greater than low"))
    end
    I_v = diagram.I_j[v]
    data = rand(rng, Utility, diagram.S[I_v]...)
    data = scale.(data, Utility(low), Utility(high))

    index_v = findfirst(j -> j==v, diagram.V)
    diagram.Y[index_v] = Utilities(v, data)
end




"""
    function LocalDecisionStrategy(rng::AbstractRNG, diagram::InfluenceDiagram, d::Node)

Generate random decision strategy for decision node `d`.

# Examples
```julia
rng = MersenneTwister(3)
diagram = InfluenceDiagram()
random_diagram!(rng, diagram, 5, 2, 3, 2, 2, rand(rng, [2,3], 5))
LocalDecisionStrategy(rng, diagram, diagram.D[1])
```


function LocalDecisionStrategy(rng::AbstractRNG, diagram::InfluenceDiagram, d::Node)
    I_d = diagram.I_j[d]
    states = diagram.S[I_d]
    state = diagram.S[d]
    data = zeros(Int, states..., state)
    for s in CartesianIndices((states...,))
        s_j = rand(rng, 1:state)
        data[s, s_j] = 1
    end
    LocalDecisionStrategy(d, data)
end
"""
=#