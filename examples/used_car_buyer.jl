using Logging
using JuMP, Gurobi
using DecisionProgramming

const O = 1  # Chance node: lemon or peach
const T = 2  # Decision node: pay stranger for advice
const R = 3  # Chance node: observation of state of the car
const A = 4  # Decision node: purchase alternative
const O_states = ["lemon", "peach"]
const T_states = ["no test", "test"]
const R_states = ["no test", "lemon", "peach"]
const A_states = ["buy without guarantee", "buy with guarantee", "don't buy"]

@info("Creating the influence diagram.")
S = States([
    (length(O_states), [O]),
    (length(T_states), [T]),
    (length(R_states), [R]),
    (length(A_states), [A]),
])
C = Vector{ChanceNode}()
D = Vector{DecisionNode}()
V = Vector{ValueNode}()
X = Vector{Probabilities}()
Y = Vector{Consequences}()

I_O = Vector{Node}()
X_O = [0.2, 0.8]
push!(C, ChanceNode(O, I_O))
push!(X, Probabilities(O, X_O))

I_T = Vector{Node}()
push!(D, DecisionNode(T, I_T))

I_R = [O, T]
X_R = zeros(S[O], S[T], S[R])
X_R[1, 1, :] = [1,0,0]
X_R[1, 2, :] = [0,1,0]
X_R[2, 1, :] = [1,0,0]
X_R[2, 2, :] = [0,0,1]
push!(C, ChanceNode(R, I_R))
push!(X, Probabilities(R, X_R))

I_A = [R]
push!(D, DecisionNode(A, I_A))

I_V1 = [T]
Y_V1 = [0.0, -25.0]
push!(V, ValueNode(5, I_V1))
push!(Y, Consequences(5, Y_V1))

I_V2 = [A]
Y_V2 = [100.0, 40.0, 0.0]
push!(V, ValueNode(6, I_V2))
push!(Y, Consequences(6, Y_V2))

I_V3 = [O, A]
Y_V3 = [-200.0 0.0 0.0;
        -40.0 -20.0 0.0]
push!(V, ValueNode(7, I_V3))
push!(Y, Consequences(7, Y_V3))

validate_influence_diagram(S, C, D, V)
sort!.((C, D, V, X, Y), by = x -> x.j)

P = DefaultPathProbability(C, X)
U = DefaultPathUtility(V, Y)

@info("Creating the decision model.")
model = Model()
U⁺ = PositivePathUtility(S, U)
z = DecisionVariables(model, S, D)
x_s = BinaryPathVariables(model, z, S, P)
EV = expected_value(model, x_s, U⁺, P)
@objective(model, Max, EV)

validate_model(model, x_s, S, P, U, expected_value_objective = true)

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
    "IntFeasTol"      => 1e-9,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(z)

@info("Printing decision strategy:")
print_decision_strategy(S, Z)

@info("Computing utility distribution.")
udist = UtilityDistribution(S, P, U, Z)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing expected utility.")
print_statistics(udist)
