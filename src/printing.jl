using Printf, Parameters

"""Print decision strategy.

# Examples
```julia
print_decision_strategy(G, Z)
```
"""
function print_decision_strategy(G::InfluenceDiagram, Z::DecisionStrategy)
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

"""Print state probabilities with fixed states.

# Examples
```julia
sprobs = StateProbabilities(G, X, Z)
print_state_probabilities(sprobs, [1, 2], ["A", "B"])
```
"""
function print_state_probabilities(sprobs::StateProbabilities, nodes::Vector{Node}, labels)
    probs = sprobs.probs
    fixed = sprobs.fixed
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
