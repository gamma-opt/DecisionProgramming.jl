using Random, Parameters

"""Generate random decision diagram with `n_C` chance nodes, `n_D` decision nodes, and `n_V` value nodes.
Parameter `n_I` is the upper bound on the size of the information set."""
function random_diagram(rng::AbstractRNG, n_C::Int, n_D::Int, n_V::Int, n_I::Int)
    C = Vector{ChanceNode}()
    D = Vector{DecisionNode}()
    V = Vector{ValueNode}()

    # Create nodes
    n = n_C + n_D
    u = shuffle(rng, 1:n)
    C_j = sort(u[1:n_C])
    D_j = sort(u[n_C+1:end])
    V_j = collect((n+1):(n+n_V))

    for j in C_j
        if j == 1
            I_j = Vector{Node}()
        else
            m = rand(rng, 0:n_I)
            I_j = unique(rand(rng, 1:(j-1), m))
        end
        push!(C, ChanceNode(j, I_j))
    end

    for j in D_j
        if j == 1
            I_j = Vector{Node}()
        else
            m = rand(rng, 0:n_I)
            I_j = unique(rand(rng, 1:(j-1), m))
        end
        push!(D, DecisionNode(j, I_j))
    end

    # TODO: require node n to be in one of the I_j
    for j in V_j
        I_j = unique(rand(rng, 1:n, n_I))
        push!(V, ValueNode(j, I_j))
    end

    return C, D, V
end

"""Generate `n` random states from `states`."""
function States(rng::AbstractRNG, states::Vector{State}, n::Int)
    States(rand(rng, states, n))
end

"""Generate random probabilities for chance node `c` with `S` states."""
function Probabilities(rng::AbstractRNG, c::ChanceNode, S::States)
    states = S[c.I_j]
    state = S[c.j]
    X = zeros(states..., state)
    for s in paths(states)
        x = rand(rng, state)
        x = x / sum(x)
        for s_j in 1:state
            X[s..., s_j] = x[s_j]
        end
    end
    Probabilities(X)
end

scale(x::Float64, low::Float64, high::Float64) = x * (high - low) + low

"""Generate random consequences between `low` and `high` for value node `v` with `S` states."""
function Consequences(rng::AbstractRNG, v::ValueNode, S::States; low::Float64=-1.0, high::Float64=1.0)
    high > low || error("")
    Y = rand(rng, S[v.I_j]...)
    Y = scale.(Y, low, high)
    Consequences(Y)
end

"""Generate random decision strategy for decision node `d` with `S` states."""
function DecisionStrategy(rng::AbstractRNG, d::DecisionNode, S::States)
    states = S[d.I_j]
    state = S[d.j]
    Z = zeros(Int, [states; state]...)
    for s in paths(states)
        s_j = rand(rng, 1:state)
        Z[s..., s_j] = 1
    end
    DecisionStrategy(Z)
end
