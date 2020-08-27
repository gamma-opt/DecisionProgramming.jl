using Printf, Random, Logging, Parameters, JuMP, Gurobi
using DecisionProgramming

Random.seed!(42)

const dᴾ = 1    # Decision node: range for number of patents
const cᵀ = 2    # Chance node:   technical competitiveness
const dᴬ = 3    # Decision node: range for number of applications
const cᴹ = 4    # Chance node:   market share
const DP_states = ["0-3 patents", "3-6 patents", "6-9 patents"]
const CT_states = ["low", "medium", "high"]
const DA_states = ["0-5 applications", "5-10 applications", "10-15 applications"]
const CM_states = ["low", "medium", "high"]

S = States([
    (length(DP_states), [dᴾ]),
    (length(CT_states), [cᵀ]),
    (length(DA_states), [dᴬ]),
    (length(CM_states), [cᴹ]),
])
C = Vector{ChanceNode}()
D = Vector{DecisionNode}()
V = Vector{ValueNode}()
X = Vector{Probabilities}()
Y = Vector{Consequences}()

I_DP = Vector{Node}()
push!(D, DecisionNode(dᴾ, I_DP))

I_CT = [dᴾ]
X_CT = zeros(S[dᴾ], S[cᵀ])
X_CT[1, :] = [1/2, 1/3, 1/6]
X_CT[2, :] = [1/3, 1/3, 1/3]
X_CT[3, :] = [1/6, 1/3, 1/2]
push!(C, ChanceNode(cᵀ, I_CT))
push!(X, Probabilities(X_CT))

I_DA = [dᴾ, cᵀ]
push!(D, DecisionNode(dᴬ, I_DA))

I_CM = [cᵀ, dᴬ]
X_CM = zeros(S[cᵀ], S[dᴬ], S[cᴹ])
X_CM[1, 1, :] = [2/3, 1/4, 1/12]
X_CM[1, 2, :] = [1/2, 1/3, 1/6]
X_CM[1, 3, :] = [1/3, 1/3, 1/3]
X_CM[2, 1, :] = [1/2, 1/3, 1/6]
X_CM[2, 2, :] = [1/3, 1/3, 1/3]
X_CM[2, 3, :] = [1/6, 1/3, 1/2]
X_CM[3, 1, :] = [1/3, 1/3, 1/3]
X_CM[3, 2, :] = [1/6, 1/3, 1/2]
X_CM[3, 3, :] = [1/12, 1/4, 2/3]
push!(C, ChanceNode(cᴹ, I_CM))
push!(X, Probabilities(X_CM))

# Dummy value node
push!(V, ValueNode(5,[cᴹ]))
push!(Y,Consequences(zeros(S[cᴹ])))

@info("Validate influence diagram.")
validate_influence_diagram(S, C, D, V)
s_c = sortperm([c.j for c in C])
s_d = sortperm([d.j for d in D])
s_v = sortperm([v.j for v in V])
C = C[s_c]
D = D[s_d]
V = V[s_v]
X = X[s_c]
Y = Y[s_v]

@info("Creating path probability.")
P = DefaultPathProbability(C, X)

@info("Defining DecisionModel")
model = Model()
z = decision_variables(model, S, D)
π_s = path_probability_variables(model, z, S, D, P)

@info("Creating problem specific constraints and expressions")

n_T = 5                     # number of technology projects
n_A = 5                     # number of application projects
I_t = rand(n_T)*0.1         # costs of technology projects
O_t = rand(1:3,n_T)         # number of patents for each tech project
I_a = rand(n_T)*2           # costs of application projects
O_a = rand(2:4,n_T)         # number of applications for each appl. project

V_A = rand(S[cᴹ], n_A).+0.5 # Value of an application
V_A[1, :] .+= -0.5          # Low market share: less value
V_A[3, :] .+= 0.5           # High market share: more value

x_T = variables(model, [S[dᴾ]...,n_T]; binary=true)
x_A = variables(model, [S[dᴾ]...,S[cᵀ]...,S[dᴬ]..., n_A]; binary=true)

M = 20                      # a large constant
ε = 0.5*minimum([O_t O_a])  # a helper variable, allows using ≤ instead of < in constraints (28b) and (29b)

q_P = [0, 3, 6, 9]          # limits of the technology intervals
q_A = [0, 5, 10, 15]        # limits of the application intervals
z_dP = z[1]
z_dA = z[2]

@constraint(model, [i=1:3],
    sum(x_T[i,t] for t in 1:n_T) <= z_dP[i]*n_T)            #(25)
@constraint(model, [i=1:3, j=1:3, k=1:3],
    sum(x_A[i,j,k,a] for a in 1:n_A) <= z_dP[i]*n_A)        #(26)
@constraint(model, [i=1:3, j=1:3, k=1:3],
    sum(x_A[i,j,k,a] for a in 1:n_A) <= z_dA[i,j,k]*n_A)    #(27)

#(28a)
@constraint(model, [i=1:3],
    q_P[i] - (1 - z_dP[i])*M <= sum(x_T[i,t]*O_t[t] for t in 1:n_T))
#(28b)
@constraint(model, [i=1:3],
    sum(x_T[i,t]*O_t[t] for t in 1:n_T) <= q_P[i+1] + (1 - z_dP[i])*M - ε)
#(29a)
@constraint(model, [i=1:3, j=1:3, k=1:3],
    q_A[k] - (1 - z_dA[i,j,k])*M <= sum(x_A[i,j,k,a]*O_a[a] for a in 1:n_A))
#(29b)
@constraint(model, [i=1:3, j=1:3, k=1:3],
    sum(x_A[i,j,k,a]*O_a[a] for a in 1:n_A) <= q_A[k+1] + (1 - z_dA[i,j,k])*M - ε)

@constraint(model, [i=1:3, j=1:3, k=1:3], x_A[i,j,k,1] <= x_T[i,1])
@constraint(model, [i=1:3, j=1:3, k=1:3], x_A[i,j,k,2] <= x_T[i,1])
@constraint(model, [i=1:3, j=1:3, k=1:3], x_A[i,j,k,2] <= x_T[i,2])

@info("Creating model objective.")
struct PathUtility <: AbstractPathUtility
    expr
end
(U::PathUtility)(s::Path) = value.(U.expr[s])

U = PathUtility(@expression(model, [s = paths(S)],
    sum(x_A[s[1:3]..., a]*(V_A[s[4],a] - I_a[a]) for a in 1:n_A) -
    sum(x_T[s[1],t]*I_t[t] for t in 1:n_T)))
EV = @expression(model, sum(π_s[s...] * U.expr[s] for s in paths(S)))
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

@info("Printing statistics")
print_statistics(udist)
