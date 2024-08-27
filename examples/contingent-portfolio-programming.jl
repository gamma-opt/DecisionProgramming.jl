using Logging, Random
using JuMP, HiGHS
using DecisionProgramming

Random.seed!(42)

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

add_node!(diagram, DecisionNode("DP", [], ["0-3 patents", "3-6 patents", "6-9 patents"]))
add_node!(diagram, ChanceNode("CT", ["DP"], ["low", "medium", "high"]))
add_node!(diagram, DecisionNode("DA", ["DP", "CT"], ["0-5 applications", "5-10 applications", "10-15 applications"]))
add_node!(diagram, ChanceNode("CM", ["CT", "DA"], ["low", "medium", "high"]))

generate_arcs!(diagram)

X_CT = ProbabilityMatrix(diagram, "CT")
X_CT[1, :] = [1/2, 1/3, 1/6]
X_CT[2, :] = [1/3, 1/3, 1/3]
X_CT[3, :] = [1/6, 1/3, 1/2]
add_probabilities!(diagram, "CT", X_CT)

X_CM = ProbabilityMatrix(diagram, "CM")
X_CM[1, 1, :] = [2/3, 1/4, 1/12]
X_CM[1, 2, :] = [1/2, 1/3, 1/6]
X_CM[1, 3, :] = [1/3, 1/3, 1/3]
X_CM[2, 1, :] = [1/2, 1/3, 1/6]
X_CM[2, 2, :] = [1/3, 1/3, 1/3]
X_CM[2, 3, :] = [1/6, 1/3, 1/2]
X_CM[3, 1, :] = [1/3, 1/3, 1/3]
X_CM[3, 2, :] = [1/6, 1/3, 1/2]
X_CM[3, 3, :] = [1/12, 1/4, 2/3]
add_probabilities!(diagram, "CM", X_CM)

@info("Creating the decision model.")
model, z, μ_s = generate_model(diagram, model_type="RJT")

@info("Creating problem specific constraints and expressions")

function model_variables(model::Model, dims::AbstractVector{Int}; binary::Bool=false)
    v = Array{VariableRef}(undef, dims...)
    for i in eachindex(v)
        v[i] = @variable(model, binary=binary)
    end
    return v
end

# Number of states in each node
n_DP = num_states(diagram, "DP")
n_CT = num_states(diagram, "CT")
n_DA = num_states(diagram, "DA")
n_CM = num_states(diagram, "CM")

n_T = 5                     # number of technology projects
n_A = 5                     # number of application projects
I_t = rand(n_T)*0.1         # costs of technology projects
O_t = rand(1:3,n_T)         # number of patents for each tech project
I_a = rand(n_T)*2           # costs of application projects
O_a = rand(2:4,n_T)         # number of applications for each appl. project

V_A = rand(n_CM, n_A).+0.5 # Value of an application
V_A[1, :] .+= -0.5          # Low market share: less value
V_A[3, :] .+= 0.5           # High market share: more value

x_T = model_variables(model, [n_DP, n_T]; binary=true)
x_A = model_variables(model, [n_DP, n_CT, n_DA, n_A]; binary=true)

M = 20                      # a large constant
ε = 0.5*minimum([O_t O_a])  # a helper variable, allows using ≤ instead of < in constraints (28b) and (29b)

q_P = [0, 3, 6, 9]          # limits of the technology intervals
q_A = [0, 5, 10, 15]        # limits of the application intervals
z_dP = z["DP"].z
z_dA = z["DA"].z

@constraint(model, [i=1:n_DP],
    sum(x_T[i,t] for t in 1:n_T) <= z_dP[i]*n_T)            #(25)
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA],
    sum(x_A[i,j,k,a] for a in 1:n_A) <= z_dP[i]*n_A)        #(26)
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA],
    sum(x_A[i,j,k,a] for a in 1:n_A) <= z_dA[i,j,k]*n_A)    #(27)

#(28a)
@constraint(model, [i=1:n_DP],
    q_P[i] - (1 - z_dP[i])*M <= sum(x_T[i,t]*O_t[t] for t in 1:n_T))
#(28b)
@constraint(model, [i=1:n_DP],
    sum(x_T[i,t]*O_t[t] for t in 1:n_T) <= q_P[i+1] + (1 - z_dP[i])*M - ε)
#(29a)
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA],
    q_A[k] - (1 - z_dA[i,j,k])*M <= sum(x_A[i,j,k,a]*O_a[a] for a in 1:n_A))
#(29b)
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA],
    sum(x_A[i,j,k,a]*O_a[a] for a in 1:n_A) <= q_A[k+1] + (1 - z_dA[i,j,k])*M - ε)

@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA], x_A[i,j,k,1] <= x_T[i,1])
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA], x_A[i,j,k,2] <= x_T[i,1])
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA], x_A[i,j,k,2] <= x_T[i,2])

@info("Creating model objective.")
patent_investment_cost = @expression(model, [i=1:n_DP], sum(x_T[i, t] * I_t[t] for t in 1:n_T))
application_investment_cost = @expression(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA], sum(x_A[i, j, k, a] * I_a[a] for a in 1:n_A))
application_value = @expression(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA, l=1:n_CM], sum(x_A[i, j, k, a] * V_A[l, a] for a in 1:n_A))
@objective(model, Max, sum( sum( diagram.P(convert.(State, (i,j,k,l))) * (application_value[i,j,k,l] - application_investment_cost[i,j,k]) for j in 1:n_CT, k in 1:n_DA, l in 1:n_CM ) - patent_investment_cost[i] for i in 1:n_DP ))


@info("Starting the optimization process.")

optimizer = optimizer_with_attributes(
    () -> HiGHS.Optimizer()
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(diagram, z)
S_probabilities = StateProbabilities(diagram, Z)

@info("Printing decision strategy:")
print_decision_strategy(diagram, Z, S_probabilities)


@info("Extracting path utilities")
struct PathUtility <: AbstractPathUtility
    data::Array{AffExpr}
end
Base.getindex(U::PathUtility, i::State) = getindex(U.data, i)
Base.getindex(U::PathUtility, I::Vararg{State,N}) where N = getindex(U.data, I...)
(U::PathUtility)(s::Path) = value.(U[s...])

path_utility = [@expression(model,
    sum(x_A[s[diagram.Nodes["DP"].index], s[diagram.Nodes["CT"].index], s[diagram.Nodes["DA"].index], a] * (V_A[s[diagram.Nodes["CM"].index], a] - I_a[a]) for a in 1:n_A) -
    sum(x_T[s[diagram.Nodes["DP"].index], t] * I_t[t] for t in 1:n_T)) for s in paths(get_values(diagram.S))]
diagram.U = PathUtility(path_utility)

@info("Computing utility distribution.")
U_distribution = UtilityDistribution(diagram, Z)

@info("Printing utility distribution.")
print_utility_distribution(U_distribution)

@info("Printing statistics")
print_statistics(U_distribution)
