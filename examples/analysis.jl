using Printf, Parameters
using DecisionProgramming

function print_results(πval, diagram::InfluenceDiagram, params::Params; πtol::Real=0)
    @unpack C, D, V, I_j, S_j = diagram
    @unpack Y = params

    utility(s) = sum(Y[v][s[I_j[v]]...] for v in V)

    num_active = sum(π > 0 for π in πval)

    # Total expected utility of the decision strategy.
    expected_utility = sum(πval[s...] * utility(s) for s in paths(S_j))

    # Average expected utility of active paths
    avg = expected_utility / num_active

    # Active paths
    println("Number of active paths versus all paths:")
    @printf("%i / %i = %f \n", num_active, prod(S_j), num_active / prod(S_j))
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

function state_probabilities(πval, diagram::InfluenceDiagram, prior::Float64, fixed::Dict{Int, Int}):: Dict{Int, Vector{Float64}}
    @unpack C, D, S_j = diagram
    probs = Dict(i => zeros(S_j[i]) for i in (C ∪ D))
    for s in paths(S_j, fixed), i in (C ∪ D)
        probs[i][s[i]] += πval[s...] / prior
    end
    return probs
end

function state_probabilities(πval, diagram::InfluenceDiagram)
    return state_probabilities(πval, diagram, 1.0, Dict{Int, Int}())
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
