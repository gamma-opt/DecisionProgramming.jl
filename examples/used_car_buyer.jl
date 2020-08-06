using Printf, Random, Logging, Parameters, JuMP, Gurobi
using DecisionProgramming


const O = 1     # Chance node: lemon or peach
const T = 2     # Decision node: pay stranger for advice
const R = 3     # Chance node: observation of state of the car
const A = 4     # Decision node: purchase alternative
const O_states = ["lemon", "peach"]
const T_states = ["no test", "test"]
const R_states = ["no test", "lemon", "peach"]
const A_states = ["buy without guarantee", "buy with guarantee", "don't buy"]

C = Vector{ChanceNode}()
D = Vector{DecisionNode}()
V = Vector{ValueNode}()
S_j = Vector{State}(undef, 4)
S_j[O] = length(O_states)
S_j[T] = length(T_states)
S_j[R] = length(R_states)
S_j[A] = length(A_states)

I_O = Vector{Node}()
X_O = [0.2,0.8]
push!(C, ChanceNode(O, I_O, S_j[O], X_O))

I_T = Vector{Node}()
push!(D, DecisionNode(T, I_T, S_j[T]))

I_R = [O,T]
X_R = zeros(S_j[O], S_j[T], S_j[R])
X_R[1, 1, :] = [1,0,0]
X_R[1, 2, :] = [0,1,0]
X_R[2, 1, :] = [1,0,0]
X_R[2, 2, :] = [0,0,1]
push!(C, ChanceNode(R, I_R, S_j[R], X_R))

I_A = [R]
push!(D, DecisionNode(A, I_A, S_j[A]))

I_V1 = [T]
Y_V1 = [0, -25]
push!(V, ValueNode(5, I_V1, Y_V1))

I_V2 = [A]
Y_V2 = [100, 40, 0]
push!(V, ValueNode(6, I_V2, Y_V2))

I_V3 = [O,A]
Y_V3 = [-200 0 0; -40 -20 0]
push!(V, ValueNode(7, I_V3, Y_V3))


@info("Defining InfluenceDiagram")
G = InfluenceDiagram(C, D, V)

@info("Creating probabilities.")
X = Probabilities(G, C)

@info("Creating consequences.")
Y = Consequences(G, V)

@info("Creating path probability.")
P = PathProbability(G, X)

@info("Creating path utility.")
U = PathUtility(G, Y)

@info("Defining DecisionModel")
U⁺ = PositivePathUtility(U)
@time model = DecisionModel(G, P; positive_path_utility=true)

@info("Creating model objective.")
@time EV = expected_value(model, G, U⁺)
@objective(model, Max, EV)

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(model)

@info("Printing decision strategy:")
print_decision_strategy(G, Z)
println()

@info("Computing utility distribution.")
@time udist = UtilityDistribution(G, P, U, Z)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing expected utility.")
@unpack u, p = udist
@printf("Expected utility: %.1f", sum(u[i]*p[i] for i in 1:length(u)))
