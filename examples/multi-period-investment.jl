using Printf, Random, Parameters, Logging
using JuMP, Gurobi
using DecisionProgramming

Random.seed!(11)

# Number of projects
n_T = if isempty(ARGS) 2 else parse(Int, ARGS[1]) end
n_A = if isempty(ARGS) 2 else parse(Int, ARGS[2]) end
D_P = [i for i in 1:n_T]
C_T = [n_T + i for i in 1:n_T]
D_A = [2*n_T + k for k in 1:n_A]
C_M = [2*n_T + n_A + k for k in 1:n_A]
V = [2*n_T + 2*n_A + k for k in 1:n_A]

# TODO: number of states

r_T = 0.25 .* rand(n_T)
p_T = rand(1:3, n_T)
r_A = 0.25 * rand(n_A)
a_A = rand(1:3, n_A)

@info("Defining influence diagram parameters.")
C = C_T ∪ C_M
D = D_P ∪ D_A
A = Vector{Pair{Int, Int}}()
S_j = Vector{Int}(undef, length(C)+length(D))

@info("Defining arcs.")
add_arcs(from, to) = append!(A, (i => j for (i, j) in zip(from, to)))
add_arcs(D_P, C_T)
add_arcs(C_T, D_A) # FIXME: all combinations
add_arcs(C_T, C_M) # FIXME: all combinations
add_arcs(D_A, C_M)
add_arcs(C_M, V)

@info("Defining states.")
S_j[D_P] = fill(2, length(D_P))
S_j[D_A] = fill(2, length(D_A))
S_j[C_T] = fill(2, length(C_T))
S_j[C_M] = fill(2, length(C_M))

@info("Defining InfluenceDiagram")
@time diagram = InfluenceDiagram(C, D, V, A, S_j)

@info("Defining Params")
@time params = random_params(diagram)

@info("Defining DecisionModel")
@time model = DecisionModel(diagram, params)

# TODO: problem specific constraints and expressions

# TODO: path utility, modified objective
