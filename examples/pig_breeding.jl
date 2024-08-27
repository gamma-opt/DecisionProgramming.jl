using Logging
using JuMP, HiGHS
using DecisionProgramming

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
    add_node!(diagram, ValueNode("V$i", ["D$i"]))
    # Health of next period
    add_node!(diagram, ChanceNode("H$(i+1)", ["H$(i)", "D$(i)"], ["ill", "healthy"]))
end
add_node!(diagram, ValueNode("V4", ["H$N"]))

generate_arcs!(diagram)

# Add probabilities for node H1
add_probabilities!(diagram, "H1", [0.1, 0.9])

# Declare probability matrix for health nodes H_2, ... H_N-1, which have identical information sets and states
X_H = ProbabilityMatrix(diagram, "H2")
X_H["healthy", "pass", :] = [0.2, 0.8]
X_H["healthy", "treat", :] = [0.1, 0.9]
X_H["ill", "pass", :] = [0.9, 0.1]
X_H["ill", "treat", :] = [0.5, 0.5]

# Declare probability matrix for test result nodes T_1...T_N
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
    add_utilities!(diagram, "V$i", [-100.0, 0.0])
end


add_utilities!(diagram, "V4", [300.0, 1000.0])

@info("Creating the decision model.")
model, z, x_s = generate_model(diagram, model_type="DP")

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    () -> HiGHS.Optimizer()
)
set_optimizer(model, optimizer)

spu = singlePolicyUpdate(diagram, model, z; x_s)
@info("Single policy update found solution $(spu[end][1]) in $(spu[end][2]/1000) seconds.")

optimize!(model)

@info("Extracting results.")

Z = DecisionStrategy(diagram, z)
S_probabilities = StateProbabilities(diagram, Z)
U_distribution = UtilityDistribution(diagram, Z)

@info("Printing decision strategy:")
print_decision_strategy(diagram, Z, S_probabilities)

@info("Printing utility distribution.")
print_utility_distribution(U_distribution)

@info("Printing statistics")
print_statistics(U_distribution)

@info("State probabilities:")
print_state_probabilities(diagram, S_probabilities, [["H$i" for i in 1:N]...])
print_state_probabilities(diagram, S_probabilities, [["T$i" for i in 1:N-1]...])
print_state_probabilities(diagram, S_probabilities, [["D$i" for i in 1:N-1]...])

@info("Conditional state probabilities")
for state in ["ill", "healthy"]
    S_probabilities2 = StateProbabilities(diagram, Z, "H1", state, S_probabilities)
    print_state_probabilities(diagram, S_probabilities2, [["H$i" for i in 1:N]...])
    print_state_probabilities(diagram, S_probabilities2, [["T$i" for i in 1:N-1]...])
    print_state_probabilities(diagram, S_probabilities2, [["D$i" for i in 1:N-1]...])
end