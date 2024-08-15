using Test, Logging, Random, JuMP
using DecisionProgramming


@info("Creating a pig farm problem with 3 decision stages")
const N = 4

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

add_node!(diagram, ChanceNode("H1", [], ["ill", "healthy"]))
for i in 1:N-1
    # Testing result
    add_node!(diagram, ChanceNode("T$i", ["H$i"], ["positive", "negative"]))
    # Decision to treat
    add_node!(diagram, DecisionNode("D$i", ["T$i"], ["treat", "pass"]))
    # Cost of treatment
    add_node!(diagram, ValueNode("C$i", ["D$i"]))
    # Health of next period
    add_node!(diagram, ChanceNode("H$(i+1)", ["H$(i)", "D$(i)"], ["ill", "healthy"]))
end
add_node!(diagram, ValueNode("MP", ["H$N"]))

generate_arcs!(diagram)

# Add probabilities for node H1
add_probabilities!(diagram, "H1", [0.1, 0.9])

# Declare proability matrix for health nodes H_2, ... H_N-1, which have identical information sets and states
X_H = ProbabilityMatrix(diagram, "H2")
X_H["healthy", "pass", :] = [0.2, 0.8]
X_H["healthy", "treat", :] = [0.1, 0.9]
X_H["ill", "pass", :] = [0.9, 0.1]
X_H["ill", "treat", :] = [0.5, 0.5]

# Declare proability matrix for test result nodes T_1...T_N
X_T = ProbabilityMatrix(diagram, "T1")
X_T["ill", "positive"] = 0.8
X_T["ill", "negative"] = 0.2
X_T["healthy", "negative"] = 0.9
X_T["healthy", "positive"] = 0.1

for i in 1:N-1
    add_probabilities!(diagram, "T$i", X_T)
    add_probabilities!(diagram, "H$(i+1)", X_H)
end

for i in 1:N-1
    add_utilities!(diagram, "C$i", [-100.0, 0.0])
end

add_utilities!(diagram, "MP", [300.0, 1000.0])

generate_diagram!(diagram, positive_path_utility = true)


@info("Creating the decision model.")
model = Model()
z = DecisionVariables(model, diagram)
x_s = PathCompatibilityVariables(model, diagram, z, probability_cut = false)
EV = expected_value(model, diagram, x_s)
@objective(model, Max, EV)


spu = singlePolicyUpdate(diagram, model, z; x_s)
@info("Single policy update found solution $(spu[end][1]) in $(spu[end][2]/1000) seconds.")

@test spu[end][1] <= 726.8121 + 1E-9 # Result is not better than the known optimal value
spu_pairs = zip(spu[1:end-1], spu[2:end])
@test all(pair -> pair[1][1] < pair[2][1], spu_pairs) # Solution values increase
@test all(pair -> pair[1][2] < pair[2][2], spu_pairs) # Times increase