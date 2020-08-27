using Printf, Random, Logging, Parameters, JuMP, Gurobi
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
push!(X, Probabilities(X_O))

I_T = Vector{Node}()
push!(D, DecisionNode(T, I_T))

I_R = [O, T]
X_R = zeros(S[O], S[T], S[R])
X_R[1, 1, :] = [1,0,0]
X_R[1, 2, :] = [0,1,0]
X_R[2, 1, :] = [1,0,0]
X_R[2, 2, :] = [0,0,1]
push!(C, ChanceNode(R, I_R))
push!(X, Probabilities(X_R))

I_A = [R]
push!(D, DecisionNode(A, I_A))

I_V1 = [T]
Y_V1 = [0.0, -25.0]
push!(V, ValueNode(5, I_V1))
push!(Y, Consequences(Y_V1))

I_V2 = [A]
Y_V2 = [100.0, 40.0, 0.0]
push!(V, ValueNode(6, I_V2))
push!(Y, Consequences(Y_V2))

I_V3 = [O, A]
Y_V3 = [-200.0 0.0 0.0;
        -40.0 -20.0 0.0]
push!(V, ValueNode(7, I_V3))
push!(Y, Consequences(Y_V3))

validate_influence_diagram(S, C, D, V)
s_c = sortperm([c.j for c in C])
s_d = sortperm([d.j for d in D])
s_v = sortperm([v.j for v in V])
C = C[s_c]
D = D[s_d]
V = V[s_v]
X = X[s_c]
Y = Y[s_v]

P = DefaultPathProbability(C, X)
U = DefaultPathUtility(V, Y)

@info("Creating the decision model.")
model = Model()
z = decision_variables(model, S, D)
π_s = path_probability_variables(model, z, S, D, P)
EV = expected_value(model, π_s, S, U)
@objective(model, Max, EV)

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(z, D)

@info("Printing decision strategy:")
print_decision_strategy(S, Z)

@info("Computing utility distribution.")
udist = UtilityDistribution(S, P, U, Z)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing expected utility.")
print_statistics(udist)
