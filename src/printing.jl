using Printf, Parameters

"""Print decision strategy."""
function print_decision_strategy(Z::DecisionStrategy, G::InfluenceDiagram)
    @unpack C, D, V, I_j, S_j = G
    println("j | s_I(j) | s_j")
    for j in D
        println("I($j) = $(I_j[j])")
        for s_I in paths(S_j[I_j[j]])
            _, s_j = findmax(Z[j][s_I..., :])
            @printf("%i | %s | %s \n", j, s_I, s_j)
        end
    end
end

"""Print state probabilities with fixed states."""
function print_state_probabilities(probs, nodes, states, fixed::Dict{Int, Int})
    print("Node")
    for label in states
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

"""Print state probabilities."""
function print_state_probabilities(probs, nodes, labels)
    return print_state_probabilities(probs, nodes, labels, Dict{Int, Int}())
end
