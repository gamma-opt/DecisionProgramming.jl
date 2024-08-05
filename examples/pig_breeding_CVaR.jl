using Logging
using JuMP, HiGHS
using DecisionProgramming
using DataStructures
using IterTools

const N = 6

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

add_node!(diagram, ChanceNode("H1", [], ["ill", "healthy"]))
for i in 1:N-1
    # Testing result
    add_node!(diagram, ChanceNode("T$i", ["H$i"], ["positive", "negative"]))
    # Decision to treat
    add_node!(diagram, DecisionNode("D$i", ["T$i"], ["treat", "pass"]))
    # Cost of treatment
    # Health of next period
    add_node!(diagram, ChanceNode("H$(i+1)", ["H$(i)", "D$(i)"], ["ill", "healthy"]))
end

node_names = []
for i in 1:N-1
    push!(node_names, "D$i")
end
push!(node_names, "H$N")
add_node!(diagram, ValueNode("MP", node_names))

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

#Define and add utilities
decision_utilities = [-100.0, 0.0]
final_utility = [300.0, 1000.0]

utility_elements = fill(decision_utilities, N-1)
push!(utility_elements, final_utility)

utilities = [sum(comb) for comb in product(utility_elements...)]

add_utilities!(diagram, "MP", utilities)

#positive_path_utility isn't relevant anymore with rjt? slows down generate_diagram! a lot with large N
#generate_diagram!(diagram, positive_path_utility = true)
generate_diagram!(diagram)

@info("Creating the decision model.")
model = Model()

z = DecisionVariables(model, diagram, names=true)

"""
x_s = PathCompatibilityVariables(model, diagram, z, probability_cut = false)
EV = expected_value(model, diagram, x_s)
@objective(model, Max, EV)
"""

μVars = cluster_variables_and_constraints(model, diagram, z)

α = 0.05
CVaR_value = 200.0
p, p_bar, p_u = RJT_conditional_value_at_risk(model, diagram, μVars, α, CVaR_value)

RJT_expected_value(model, diagram, μVars)

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    () -> HiGHS.Optimizer()
)
set_optimizer(model, optimizer)

#spu = singlePolicyUpdate(diagram, model, z; x_s)
spu = singlePolicyUpdate(diagram, model, z)
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

CVaR = sum(value.(ρ_bar) * u for (u, ρ_bar) in p_bar)/0.05
println("CVaR output:")
println(CVaR)
