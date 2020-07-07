using Printf, Parameters

"""State probabilities."""
function state_probabilities(πval::Array{Float64}, diagram::InfluenceDiagram, prior::Float64, fixed::Dict{Int, Int}):: Dict{Int, Vector{Float64}}
    @unpack C, D, S_j = diagram
    probs = Dict(i => zeros(S_j[i]) for i in (C ∪ D))
    for s in paths(S_j, fixed), i in (C ∪ D)
        probs[i][s[i]] += πval[s...] / prior
    end
    return probs
end

"""State probabilities."""
function state_probabilities(πval::Array{Float64}, diagram::InfluenceDiagram)
    return state_probabilities(πval, diagram, 1.0, Dict{Int, Int}())
end

"""Cumulative distribution."""
function cumulative_distribution(πval::Array{Float64}, diagram::InfluenceDiagram, params::Params)
    @unpack C, D, V, I_j, S_j = diagram
    @unpack Y = params
    utility(s) = sum(Y[v][s[I_j[v]]...] for v in V)
    u = similar(πval)
    for s in paths(S_j)
        u[s...] = utility(s)
    end
    indices = sortperm(u[:])
    x = u[indices]
    y = cumsum(πval[indices])
    return x, y
end

function print_state_probabilities(probs, nodes, labels, fixed::Dict{Int, Int})
    print("Node")
    for label in labels
        print(" | ", label)
    end
    println()
    for i in nodes
        @printf("%4i", i)
        for prob in probs[i]
            @printf(" | %0.3f", prob)
        end
        if i ∈ keys(fixed)
            @printf(" | Fixed to state %i", fixed[i])
        end
        println()
    end
end

function print_state_probabilities(probs, nodes, labels)
    return print_state_probabilities(probs, nodes, labels, Dict{Int, Int}())
end

function print_results(πval::Array{Float64}, diagram::InfluenceDiagram, params::Params; πtol::Float64=0.0)
    @unpack C, D, V, I_j, S_j = diagram
    @unpack X, Y = params

    utility(s) = sum(Y[v][s[I_j[v]]...] for v in V)

    # Upper bound of number of active paths
    # num_active_upper = prod(S_j[j] for j in C)
    num_active_upper = prod(S_j)

    # Number of active paths.
        # Upper bound of probability of a path.
    probability(s) = prod(X[j][s[[I_j[j]; j]]...] for j in C)

    # Minimum path probability
    ϵ = minimum(probability(s) for s in paths(S_j))

    num_active = sum(π ≥ ϵ for π in πval)

    # Total expected utility of the decision strategy.
    expected_utility = sum(πval[s...] * utility(s) for s in paths(S_j))

    # Average expected utility of active paths
    avg = expected_utility / num_active

    # Active paths
    println("Number of active paths versus all paths:")
    @printf("%i / %i = %f \n", num_active, num_active_upper, num_active / num_active_upper)
    println("Expected expected utility:")
    println(expected_utility)
    println("Average expected utility of active paths:")
    println(avg)
    println()

    println("probability | utility | expected utility | path")
    for s in paths(S_j)
        ut = utility(s)
        eu = πval[s...] * ut
        if (πval[s...] ≥ πtol) | (eu ≥ avg)
            @printf("%.3f | %10.3f | %10.3f | %s \n", πval[s...], ut, eu, s)
        end
    end
    println()
end
