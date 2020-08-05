using Printf, Random, Parameters, Logging
using Base.Iterators: product
using JuMP, Gurobi
using DecisionProgramming

Random.seed!(11)

n_T = 2
n_A = 3
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
one_to_one(from, to) = append!(A, (i => j for (i, j) in zip(from, to)))
one_to_many(from, to) = append!(A, (i => j for (i, j) in product(from, to)))
one_to_one(D_P, C_T)
one_to_many(C_T, D_A)
one_to_many(C_T, C_M)
one_to_one(D_A, C_M)
one_to_one(C_M, V)

@info("Defining states.")
S_j[D_P] = fill(2, length(D_P))
S_j[D_A] = fill(2, length(D_A))
S_j[C_T] = fill(2, length(C_T))
S_j[C_M] = fill(2, length(C_M))

@info("Defining InfluenceDiagram")
G = InfluenceDiagram(C, D, V, A, S_j)
X = random_probabilities(G)
Y = random_consequences(G)

@info("Defining Model")
model = DecisionModel(G, X)

# Problem specific constraints and expressions
x_T = variables(model, [n_T]; binary=true)
x_A = variables(model, [S_j[C_T]..., n_A]; binary=true)

# Path utility and objective function
