using Random, Parameters

"""Create random influence diagram of given size."""
function random_influence_diagram(n_C::Int, n_D::Int, n_V::Int, n_A::Int, num_states::Vector{Int})
    # Create nodes
    n = n_C + n_D
    u = shuffle(1:n)
    C = u[1:n_C]
    D = u[n_C+1:end]
    V = collect((n+1):(n+n_V))

    # Create arcs between chance and decision nodes.
    A = Vector{Pair{Int, Int}}()
    for i in 1:(n-1)
        js = unique(rand((i+1):n, rand(1:n_A)))
        append!(A, i => j for j in js)
    end

    # Create arcs between from nodes chance and decision nodes to value nodes.
    for v in V
        is = unique(rand(1:(n-1), rand(0:n_A)))
        append!(A, i => v for i in is)
    end
    # There should be atleast one arc from node n to V
    append!(A, n => v for v in unique(rand(V, rand(1:n_A))))

    # Create states
    S_j = rand(num_states, n)

    return InfluenceDiagram(C, D, V, A, S_j)
end

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

"""Generate random params."""
function random_params(diagram::InfluenceDiagram)
    @unpack C, V, S_j, I_j = diagram
    X = Dict{Int, Array{Float64}}(
        i => random_probabilites(S_j[I_j[i]], S_j[i]) for i in C)
    Y = Dict{Int, Array{Float64}}(
        i => random_consequences(S_j[I_j[i]]) for i in V)
    return Params(X, Y)
end
