using Printf
using JuMP, Gurobi
using DecisionProgramming

N = 2 # Number of monitors
L = [1]
R_i = [i + 1 for i in 1:N]
A_i = [(N + 1) + i for i in 1:N]
F = [2*N + 2]
T = [2*N + 3]

C = L ∪ R_i ∪ F
D = A_i
V = T
A = Vector{Pair{Int, Int}}()
S_j = Vector{Int}()

# Arcs
add_arcs(from, to) = append!(A, (i => j for (i, j) in zip(from, to)))
add_arcs(repeat(L, N), R_i)
add_arcs(L, F)
add_arcs(R_i, A_i)
add_arcs(A_i, repeat(F, N))
add_arcs(F, T)
add_arcs(A_i, repeat(T, N))

# States
S_j = zeros(Int, length(C)+length(D))
S_j[L] = fill(2, length(L))
S_j[R_i] = fill(2, length(R_i))
S_j[A_i] = fill(2, length(A_i))
S_j[F] = fill(2, length(F))

# Model
diagram = InfluenceDiagram(C, D, V, A, S_j)
