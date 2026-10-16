using Logging
using JuMP, HiGHS, Gurobi
using DecisionProgramming
using CSV, DataFrames, PrettyTables



@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

const O_states = ["dry", "wet", "soaking"]
const S_states = ["NS", "OS", "CS"]
const T_states = ["test", "no test"]
const D_states = ["drill", "no drill"]
const R_states = ["N/A","NS", "OS", "CS"]


add_node!(diagram, ChanceNode("O", [], O_states))
add_node!(diagram, DecisionNode("T", [], T_states))


add_node!(diagram, ChanceNode("S", ["O"], S_states))
#add_node!(diagram, ChanceNode("R2", ["S","T"], R_states))
add_node!(diagram, ChanceNode("R", ["S","T"], R_states))
add_node!(diagram, DecisionNode("D", ["R"], D_states))


add_node!(diagram, ValueNode("U", ["D", "O"]))
add_node!(diagram, ValueNode("C", ["T"]))
#add_node!(diagram, ValueNode("C2", ["T"]))

generate_arcs!(diagram)

X_O = ProbabilityMatrix(diagram, "O")
X_O["dry"] = 0.5
X_O["wet"] = 0.3
X_O["soaking"] = 0.2
add_probabilities!(diagram,"O",X_O)


##const S_states = ["dry", "medium", "wet"]
#add_node!(diagram, ChanceNode("S", ["O"], S_states))

X_S = ProbabilityMatrix(diagram, "S")

X_S["dry", :]    = [0.95, 0.04, 0.01]
X_S["wet", :] = [0.05, 0.90, 0.05]
X_S["soaking", :]    = [0.01, 0.04, 0.95]

add_probabilities!(diagram, "S", X_S)

#add_node!(diagram, ChanceNode("R1", ["S","T1"], R1_states))

X_R = ProbabilityMatrix(diagram, "R")
# Test performed - R1_states = ["N/A", "dry", "medium", "wet"]
X_R["NS", "test", :] = [0.0, 0.80, 0.15, 0.05]
X_R["OS", "test", :] = [0.0, 0.20, 0.60, 0.20]
X_R["CS", "test", :] = [0.0, 0.05, 0.20, 0.75]

# No test performed - R1_states = ["N/A", "dry", "medium", "wet"]
X_R["NS", "no test", :] = [1.0, 0.0, 0.0, 0.0]
X_R["OS", "no test", :] = [1.0, 0.0, 0.0, 0.0]
X_R["CS", "no test", :] = [1.0, 0.0, 0.0, 0.0]
add_probabilities!(diagram,"R",X_R)

# X_R2 = ProbabilityMatrix(diagram, "R2")

# # Test performed - R2_states = ["N/A", "dry", "medium", "wet"]
# X_R2["dry", "test", :] = [0.0, 0.80, 0.15, 0.05]
# X_R2["medium", "test", :] = [0.0, 0.20, 0.60, 0.20]
# X_R2["wet", "test", :] = [0.0, 0.05, 0.25, 0.70]

# # No test performed - R2_states = ["N/A", "dry", "medium", "wet"]
# X_R2["dry", "no test", :] = [1.0, 0.0, 0.0, 0.0]
# X_R2["medium", "no test", :] = [1.0, 0.0, 0.0, 0.0]
# X_R2["wet", "no test", :] = [1.0, 0.0, 0.0, 0.0]
#add_probabilities!(diagram, "R2", X_R2)

const cost_T = -5_000

const cost_D  = -800000

payoff = Dict(
    "dry" => 0,
    "wet" => 1_000_000,
    "soaking" => 5_000_000
)


Y_U = UtilityMatrix(diagram, "U")
for o in O_states
    Y_U["drill", o] = cost_D + payoff[o]
    Y_U["no drill", o] = 0
end
add_utilities!(diagram, "U", Y_U)

Y_C = UtilityMatrix(diagram, "C")
Y_C["test"] = cost_T
Y_C["no test"] = 0
add_utilities!(diagram, "C", Y_C)



@info("Creating the decision model.")
model, z, x_s = generate_model(
    diagram,
    model_type="DP",
    probability_cut=true
)
@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer()
)
set_optimizer(model, optimizer)

optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(diagram,z)
S_probabilities = StateProbabilities(diagram, Z)
U_distribution = UtilityDistribution(diagram, Z)

@info("Printing decision strategy using tailor made function:")
print_decision_strategy(diagram, Z, S_probabilities)

@info("Printing state probabilities:")
print_state_probabilities(
    diagram,
    S_probabilities,
    ["O", "S", "R"]
)

@info("Printing utility distribution.")
print_utility_distribution(U_distribution)

print_statistics(U_distribution)
