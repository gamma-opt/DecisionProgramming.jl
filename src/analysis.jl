"""
    struct CompatiblePaths
        S::States
        C::Vector{Node}
        Z::DecisionStrategy
        fixed::FixedPath
    end

CompatiblePaths type.
"""
struct CompatiblePaths
    S::States
    C::Vector{Node}
    Z::DecisionStrategy
    fixed::FixedPath
    function CompatiblePaths(S, C, Z, fixed)
        if !all(k∈Set(C) for k in keys(fixed))
            throw(DomainError("You can only fix chance states."))
        end
        new(S, C, Z, fixed)
    end
end

"""
    CompatiblePaths(diagram::InfluenceDiagram, Z::DecisionStrategy, fixed::FixedPath=Dict{Node, State}())

CompatiblePaths outer construction function. Interface for iterating over paths that are compatible and active given influence diagram and decision strategy.

1) Initialize path `s` of length `n`
2) Fill chance states `s[C]` by generating subpaths `paths(C)`
3) Fill decision states `s[D]` by decision strategy `Z` and path `s`

# Examples
```julia
for s in CompatiblePaths(diagram, Z)
    ...
end
```
"""
function CompatiblePaths(diagram::InfluenceDiagram, Z::DecisionStrategy, fixed::FixedPath=Dict{Node, State}())
    C_indices = indices(diagram.C)
    CompatiblePaths(States(get_values(diagram.S)), C_indices, Z, fixed)
end

function compatible_path(S::States, C::Vector{Node}, Z::DecisionStrategy, s_C::Path)
    s = Array{State}(undef, length(S))
    for (c, s_C_j) in zip(C, s_C)
        s[c] = s_C_j
    end
    for (d, I_d, Z_d) in zip(Z.D, Z.I_d, Z.Z_d)
        s[d] = Z_d((s[I_d]...,))
    end
    return (s...,)
end

function Base.iterate(S_Z::CompatiblePaths)
    if isempty(S_Z.fixed)
        iter = paths(S_Z.S[S_Z.C])
    else
        ks = sort(collect(keys(S_Z.fixed)))
        fixed = Dict{Node, State}(Node(i) => S_Z.fixed[k] for (i, k) in enumerate(S_Z.C) if k in ks)
        iter = paths(S_Z.S[S_Z.C], fixed)
    end
    next = iterate(iter)
    if next !== nothing
        s_C, state = next
        return (compatible_path(S_Z.S, S_Z.C, S_Z.Z, s_C), (iter, state))
    end
end

function Base.iterate(S_Z::CompatiblePaths, gen)
    iter, state = gen
    next = iterate(iter, state)
    if next !== nothing
        s_C, state = next
        return (compatible_path(S_Z.S, S_Z.C, S_Z.Z, s_C), (iter, state))
    end
end

Base.eltype(::Type{CompatiblePaths}) = Path
Base.length(S_Z::CompatiblePaths) = prod(S_Z.S[c] for c in S_Z.C)

"""
    struct UtilityDistribution
        u::Vector{Float64}
        p::Vector{Float64}
    end

UtilityDistribution type.

"""
struct UtilityDistribution
    u::Vector{Float64}
    p::Vector{Float64}
end

"""
    UtilityDistribution(diagram::InfluenceDiagram, Z::DecisionStrategy)

Construct the probability mass function for path utilities on paths that are compatible with given decision strategy.

# Examples
```julia
UtilityDistribution(diagram, Z)
```
"""
function UtilityDistribution(diagram::InfluenceDiagram, Z::DecisionStrategy)
    # Extract utilities and probabilities of active paths
    S_Z = CompatiblePaths(diagram, Z)
    utilities = Vector{Float64}(undef, length(S_Z))
    probabilities = Vector{Float64}(undef, length(S_Z))
    for (i, s) in enumerate(S_Z)
        utilities[i] = diagram.U(s)
        probabilities[i] = diagram.P(s)
    end

    # Filter zero probabilities
    nonzero = @. (!)(iszero(probabilities))
    utilities = utilities[nonzero]
    probabilities = probabilities[nonzero]

    # Sort by utilities
    perm = sortperm(utilities)
    u = utilities[perm]
    p = probabilities[perm]

    # Compute the probability mass function
    u2 = unique(u)
    p2 = similar(u2)
    j = 1
    p2[j] = p[1]
    for k in 2:length(u)
        if u[k] == u2[j]
            p2[j] += p[k]
        else
            j += 1
            p2[j] = p[k]
        end
    end

    UtilityDistribution(u2, p2)
end

"""
    struct StateProbabilities
        probs::Dict{Node, Vector{Float64}}
        fixed::FixedPath
    end

StateProbabilities type.
"""
struct StateProbabilities
    probs::Dict{Node, Vector{Float64}}
    fixed::FixedPath
end

"""
    StateProbabilities(diagram::InfluenceDiagram, Z::DecisionStrategy, node::Node, state::State, prior_probabilities::StateProbabilities)

Associate each node with array of conditional probabilities for each of its states occuring in compatible paths given
    fixed states and prior probability. Fix node and state using their indices.

# Examples
```julia
# Prior probabilities
prior_probabilities = StateProbabilities(diagram, Z)
StateProbabilities(diagram, Z, Node(2), State(1), prior_probabilities)
```
"""
function StateProbabilities(diagram::InfluenceDiagram, Z::DecisionStrategy, node::Node, state::State, prior_probabilities::StateProbabilities)
    prior = prior_probabilities.probs[node][state]
    fixed = deepcopy(prior_probabilities.fixed)

    push!(fixed, node => state)
    probs = Dict(i => zeros(get_values(diagram.S)[i]) for i in 1:length(diagram.S))
    for s in CompatiblePaths(diagram, Z, fixed), i in 1:length(diagram.S)
        probs[i][s[i]] += diagram.P(s) / prior
    end
    StateProbabilities(probs, fixed)
end

"""
    StateProbabilities(diagram::InfluenceDiagram, Z::DecisionStrategy, node::Name, state::Name, prior_probabilities::StateProbabilities)

Associate each node with array of conditional probabilities for each of its states occuring in compatible paths given
    fixed states and prior probability. Fix node and state using their names.

# Examples
```julia
# Prior probabilities
prior_probabilities = StateProbabilities(diagram, Z)

# Select node and fix its state
node = "R"
state = "no test"
StateProbabilities(diagram, Z, node, state, prior_probabilities)
```
"""
function StateProbabilities(diagram::InfluenceDiagram, Z::DecisionStrategy, node::Name, state::Name, prior_probabilities::StateProbabilities)
    node_index = findfirst(j -> j ==node, diagram.Names)
    state_index = findfirst(j -> j == state, get_values(diagram.States)[node_index])

    return StateProbabilities(diagram, Z, Node(node_index), State(state_index), prior_probabilities)
end


"""
    StateProbabilities(diagram::InfluenceDiagram, Z::DecisionStrategy)

Associate each node with array of probabilities for each of its states occuring in compatible paths.

# Examples
```julia
StateProbabilities(diagram, Z)
```
"""
function StateProbabilities(diagram::InfluenceDiagram, Z::DecisionStrategy)
    probs = Dict(i => zeros(get_values(diagram.S)[i]) for i in 1:length(diagram.S))
    for s in CompatiblePaths(diagram, Z), i in 1:length(diagram.S)
        probs[i][s[i]] += diagram.P(s)
    end
    StateProbabilities(probs, Dict{Node, State}())
end

"""
    value_at_risk(U_distribution::UtilityDistribution, α::Float64)

Calculate value-at-risk.
"""
function value_at_risk(U_distribution::UtilityDistribution, α::Float64)
    @assert 0 ≤ α ≤ 1 "We should have 0 ≤ α ≤ 1."
    perm = sortperm(U_distribution.u)
    u, p = U_distribution.u[perm], U_distribution.p[perm]
    index = findfirst(x -> x≥α, cumsum(p))
    return if index === nothing; u[end] else u[index] end
end

"""
    conditional_value_at_risk(U_distribution::UtilityDistribution, α::Float64)

Calculate conditional value-at-risk.
"""
function conditional_value_at_risk(U_distribution::UtilityDistribution, α::Float64)
    x_α = value_at_risk(U_distribution, α)
    if iszero(α)
        return x_α
    else
        tail = U_distribution.u .≤ x_α
        return (sum(U_distribution.u[tail] .* U_distribution.p[tail]) - (sum(U_distribution.p[tail]) - α) * x_α) / α
    end
end
