
struct CompatiblePaths
    S::States
    C::Vector{Node}
    Z::DecisionStrategy
    fixed::Dict{Node, State}
    function CompatiblePaths(diagram, Z, fixed)
        if !all(k∈Set(diagram.C) for k in keys(fixed))
            throw(DomainError("You can only fix chance states."))
        end
        new(diagram.S, diagram.C, Z, fixed)
    end
end

"""
    CompatiblePaths(S::States, C::Vector{ChanceNode}, Z::DecisionStrategy)

Interface for iterating over paths that are compatible and active given influence diagram and decision strategy.

1) Initialize path `s` of length `n`
2) Fill chance states `s[C]` by generating subpaths `paths(C)`
3) Fill decision states `s[D]` by decision strategy `Z` and path `s`

# Examples
```julia
for s in CompatiblePaths(S, C, Z)
    ...
end
```
"""

function CompatiblePaths(diagram::InfluenceDiagram, Z::DecisionStrategy)
    CompatiblePaths(diagram.S, diagram.C, Z, Dict{Node, State}())
end

function compatible_path(S::States, C::Vector{Node}, Z::DecisionStrategy, s_C::Path)
    s = Array{Int}(undef, length(S))
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
        fixed = Dict{Int, Int}(i => S_Z.fixed[k] for (i, k) in enumerate(ks))
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
     UtilityDistribution

UtilityDistribution type.

"""
struct UtilityDistribution
    u::Vector{Float64}
    p::Vector{Float64}
end

"""
    UtilityDistribution(S::States, P::AbstractPathProbability, U::AbstractPathUtility, Z::DecisionStrategy)

Constructs the probability mass function for path utilities on paths that are compatible and active.

# Examples
```julia
UtilityDistribution(S, P, U, Z)
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
    StateProbabilities

StateProbabilities type.
"""
struct StateProbabilities
    probs::Dict{Node, Vector{Float64}}
    fixed::Dict{Node, State}
end

"""
    function StateProbabilities(S::States, P::AbstractPathProbability, Z::DecisionStrategy, node::Node, state::State, prev::StateProbabilities)

Associates each node with array of conditional probabilities for each of its states occuring in active paths given fixed states and prior probability.

# Examples
```julia
# Prior probabilities
prev = StateProbabilities(S, P, Z)

# Select node and fix its state
node = 1
state = 2
StateProbabilities(S, P, Z, node, state, prev)
```
"""
function StateProbabilities(S::States, P::AbstractPathProbability, Z::DecisionStrategy, node::Node, state::State, prev::StateProbabilities)
    prior = prev.probs[node][state]
    fixed = prev.fixed
    push!(fixed, node => state)
    probs = Dict(i => zeros(S[i]) for i in 1:length(S))
    for s in CompatiblePaths(S, P.C, Z, fixed), i in 1:length(S)
        probs[i][s[i]] += P(s) / prior
    end
    StateProbabilities(probs, fixed)
end

"""
    function StateProbabilities(S::States, P::AbstractPathProbability, Z::DecisionStrategy)

Associates each node with array of probabilities for each of its states occuring in active paths.

# Examples
```julia
StateProbabilities(S, P, Z)
```
"""
function StateProbabilities(S::States, P::AbstractPathProbability, Z::DecisionStrategy)
    probs = Dict(i => zeros(S[i]) for i in 1:length(S))
    for s in CompatiblePaths(S, P.C, Z), i in 1:length(S)
        probs[i][s[i]] += P(s)
    end
    StateProbabilities(probs, Dict{Node, State}())
end

"""
    function value_at_risk(u::Vector{Float64}, p::Vector{Float64}, α::Float64)

Value-at-risk.
"""
function value_at_risk(u::Vector{Float64}, p::Vector{Float64}, α::Float64)
    @assert 0 ≤ α ≤ 1 "We should have 0 ≤ α ≤ 1."
    i = sortperm(u)
    u, p = u[i], p[i]
    index = findfirst(x -> x≥α, cumsum(p))
    return if index === nothing; u[end] else u[index] end
end

"""
    function conditional_value_at_risk(u::Vector{Float64}, p::Vector{Float64}, α::Float64)

Conditional value-at-risk.
"""
function conditional_value_at_risk(u::Vector{Float64}, p::Vector{Float64}, α::Float64)
    x_α = value_at_risk(u, p, α)
    if iszero(α)
        return x_α
    else
        tail = u .≤ x_α
        return (sum(u[tail] .* p[tail]) - (sum(p[tail]) - α) * x_α) / α
    end
end
