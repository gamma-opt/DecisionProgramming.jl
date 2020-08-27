# N-Monitoring
## Description
The $N$-monitoring problem is described in [^1], sections 4.1 and 6.1.


## Influence Diagram
![](figures/n-monitoring.svg)

The influence diagram of generalized $N$-monitoring problem where $N≥1$ and indices $k=1,...,N.$ The nodes are associated with states as follows. **Load state** $L=\{high, low\}$ denotes the load on a structure, **report states** $R_k=\{high, low\}$ report the load state to the **action states** $A_k=\{yes, no\}$ which represent different decisions to fortify the structure. The **failure state** $F=\{failure, success\}$ represents whether or not the (fortified) structure fails under the load $L$. Finally, the utility at target $T$ depends on the whether $F$ fails and the fortification costs.

We draw the cost of fortification $c_k∼U(0,1)$ from a uniform distribution, and the magnitude of fortification is directly proportional to the cost. Fortification is defined as

$$f(A_k=yes) = c_k$$

$$f(A_k=no) = 0$$

```julia
using Printf, Random, Logging, Parameters, JuMP, Gurobi
using DecisionProgramming

Random.seed!(13)

const N = 4
const L = [1]
const R_k = [k + 1 for k in 1:N]
const A_k = [(N + 1) + k for k in 1:N]
const F = [2*N + 2]
const T = [2*N + 3]
const L_states = ["high", "low"]
const R_k_states = ["high", "low"]
const A_k_states = ["yes", "no"]
const F_states = ["failure", "success"]
const c_k = rand(N)
const b = 0.03
fortification(k, a) = [c_k[k], 0][a]

S = States([
    (length(L_states), L),
    (length(R_k_states), R_k),
    (length(A_k_states), A_k),
    (length(F_states), F)
])
C = Vector{ChanceNode}()
D = Vector{DecisionNode}()
V = Vector{ValueNode}()
X = Vector{Probabilities}()
Y = Vector{Consequences}()
```

### Load State Probability
The probability that the load is high, $ℙ(L=high)$, is drawn from a uniform distribution.

$$ℙ(L=high)∼U(0,1)$$

```julia
for j in L
    I_j = Vector{Node}()
    X_j = zeros(S[I_j]..., S[j])
    X_j[1] = rand()
    X_j[2] = 1.0 - X_j[1]
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(X_j))
end
```

### Reporting Probability
The probabilities of the report states correspond to the load state. We draw the values $x∼U(0,1)$ and $y∼U(0,1)$ from uniform distribution.

$$ℙ(R_k=high∣L=high)=\max\{x,x-1\}$$

$$ℙ(R_k=low∣L=low)=\max\{y,y-1\}$$

The probability of a correct report is thus in the range [0.5,1]. (This reflects the fact that a probability under 50% would not even make sense, since we would notice that if the test suggests a high load, the load is more likely to be low, resulting in that a low report "turns into" a high report and vice versa.)

```julia
for j in R_k
    I_j = L
    x, y = rand(2)
    X_j = zeros(S[I_j]..., S[j])
    X_j[1, 1] = max(x, 1-x)
    X_j[1, 2] = 1.0 - X_j[1, 1]
    X_j[2, 2] = max(y, 1-y)
    X_j[2, 1] = 1.0 - X_j[2, 2]
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(X_j))
end
```

### Decision to Fortify

Only the corresponding load report is known when making the fortification decision, thus $I(A_k)=R_k$.

```julia
for (i, j) in zip(R_k, A_k)
    I_j = [i]
    push!(D, DecisionNode(j, I_j))
end
```

### Probability of Failure
The probabilities of failure which are decresead by fortifications. We draw the values $x∼U(0,1)$ and $y∼U(0,1)$ from uniform distribution.

$$ℙ(F=failure∣A_N,...,A_1,L=high)=\frac{\max{\{x, 1-x\}}}{e^{b(∑_{k=1,...,N} f(A_k))}}$$

$$ℙ(F=failure∣A_N,...,A_1,L=low)=\frac{\min{\{y, 1-y\}}}{e^{b(∑_{k=1,...,N} f(A_k))}}$$

```julia
for j in F
    I_j = L ∪ A_k
    x, y = rand(2)
    X_j = zeros(S[I_j]..., S[j])
    for s in paths(S[A_k])
        d = exp(b * sum(fortification(k, a) for (k, a) in enumerate(s)))
        X_j[1, s..., 1] = max(x, 1-x) / d
        X_j[1, s..., 2] = 1.0 - X_j[1, s..., 1]
        X_j[2, s..., 1] = min(y, 1-y) / d
        X_j[2, s..., 2] = 1.0 - X_j[2, s..., 1]
    end
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(X_j))
end
```

### Consequences
Utility from consequences at target $T$ from failure state $F$

$$g(F=failure) = 0$$

$$g(F=success) = 100$$

Utility from consequences at target $T$ from action states $A_k$ is

$$f(A_k=yes) = c_k$$

$$f(A_k=no) = 0$$

Total cost

$$Y(F, A_N, ..., A_1) = g(F) + (-f(A_N)) + ... + (-f(A_1))$$

```julia
for j in T
    I_j = A_k ∪ F
    Y_j = zeros(S[I_j]...)
    for s in paths(S[A_k])
        cost = sum(-fortification(k, a) for (k, a) in enumerate(s))
        Y_j[s..., 1] = cost + 0
        Y_j[s..., 2] = cost + 100
    end
    push!(V, ValueNode(j, I_j))
    push!(Y, Consequences(Y_j))
end
```

### Validating Influence Diagram

Finally, we need to validate the influence diagram and sort the nodes, probabilities and consequences in increasing order by the node indices.

```julia
validate_influence_diagram(S, C, D, V)
s_c = sortperm([c.j for c in C])
s_d = sortperm([d.j for d in D])
s_v = sortperm([v.j for v in V])
C = C[s_c]
D = D[s_d]
V = V[s_v]
X = X[s_c]
Y = Y[s_v]
```

We define the path probability.
```julia
P = DefaultPathProbability(C, X)
```

As the path utility, we use the default, which is the sum of the consequences given the path.
```julia
U = DefaultPathUtility(V, Y)
```


## Decision Model

An affine transformation is applied to the path utility, making all utilities positive. See [section](../decision-programming/decision-model.md) on positive path utilities for details.

```julia
U⁺ = PositivePathUtility(S, U)
model = DecisionModel(S, D, P; positive_path_utility=true)
```

Two [lazy constraints](../decision-programming/decision-model.md) are also used to speed up the solution process.

```julia
probability_cut(model, S, P)
active_paths_cut(model, S, P)
```

The expected utility is used as the objective and the problem is solved using Gurobi.

```julia
EV = expected_value(model, S, U⁺)
@objective(model, Max, EV)

optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)
```


## Analyzing Results

The decision strategy shows us that the optimal strategy is to make all four fortifications regardless of the reports (state 1 in fortification nodes corresponds to the option "yes").

```julia
Z = DecisionStrategy(model, D)
```

```julia-repl
julia> print_decision_strategy(S, Z)
┌────────┬──────┬───┐
│  Nodes │ (2,) │ 6 │
├────────┼──────┼───┤
│ States │ (1,) │ 1 │
│ States │ (2,) │ 1 │
└────────┴──────┴───┘
┌────────┬──────┬───┐
│  Nodes │ (3,) │ 7 │
├────────┼──────┼───┤
│ States │ (1,) │ 1 │
│ States │ (2,) │ 1 │
└────────┴──────┴───┘
┌────────┬──────┬───┐
│  Nodes │ (4,) │ 8 │
├────────┼──────┼───┤
│ States │ (1,) │ 1 │
│ States │ (2,) │ 1 │
└────────┴──────┴───┘
┌────────┬──────┬───┐
│  Nodes │ (5,) │ 9 │
├────────┼──────┼───┤
│ States │ (1,) │ 1 │
│ States │ (2,) │ 1 │
└────────┴──────┴───┘
```


The state probabilities for the strategy $Z$ can also be obtained. These tell the probability of each state in each node, given the strategy $Z$.


```julia
sprobs = StateProbabilities(S, P, Z)
```

```julia-repl
julia> print_state_probabilities(sprobs, L)
┌───────┬──────────┬──────────┬─────────────┐
│  Node │  State 1 │  State 2 │ Fixed state │
│ Int64 │  Float64 │  Float64 │      String │
├───────┼──────────┼──────────┼─────────────┤
│     1 │ 0.564449 │ 0.435551 │             │
└───────┴──────────┴──────────┴─────────────┘
julia> print_state_probabilities(sprobs, R_k)
┌───────┬──────────┬──────────┬─────────────┐
│  Node │  State 1 │  State 2 │ Fixed state │
│ Int64 │  Float64 │  Float64 │      String │
├───────┼──────────┼──────────┼─────────────┤
│     2 │ 0.515575 │ 0.484425 │             │
│     3 │ 0.442444 │ 0.557556 │             │
│     4 │ 0.543724 │ 0.456276 │             │
│     5 │ 0.552515 │ 0.447485 │             │
└───────┴──────────┴──────────┴─────────────┘
julia> print_state_probabilities(sprobs, A_k)
┌───────┬──────────┬──────────┬─────────────┐
│  Node │  State 1 │  State 2 │ Fixed state │
│ Int64 │  Float64 │  Float64 │      String │
├───────┼──────────┼──────────┼─────────────┤
│     6 │ 1.000000 │ 0.000000 │             │
│     7 │ 1.000000 │ 0.000000 │             │
│     8 │ 1.000000 │ 0.000000 │             │
│     9 │ 1.000000 │ 0.000000 │             │
└───────┴──────────┴──────────┴─────────────┘
julia> print_state_probabilities(sprobs, F)
┌───────┬──────────┬──────────┬─────────────┐
│  Node │  State 1 │  State 2 │ Fixed state │
│ Int64 │  Float64 │  Float64 │      String │
├───────┼──────────┼──────────┼─────────────┤
│    10 │ 0.038697 │ 0.961303 │             │
└───────┴──────────┴──────────┴─────────────┘
```

We can also print the utility distribution for the optimal strategy and some basic statistics for the distribution.

```julia
udist = UtilityDistribution(S, P, U, Z)
```

```julia-repl
julia> print_utility_distribution(udist)
┌───────────┬─────────────┐
│   Utility │ Probability │
│   Float64 │     Float64 │
├───────────┼─────────────┤
│ -2.881344 │    0.038697 │
│ 97.118656 │    0.961303 │
└───────────┴─────────────┘
```

```julia-repl
julia> print_statistics(udist)
┌──────────┬────────────┐
│     Name │ Statistics │
│   String │    Float64 │
├──────────┼────────────┤
│     Mean │  93.248950 │
│      Std │  19.287197 │
│ Skewness │  -4.783515 │
│ Kurtosis │  20.882012 │
└──────────┴────────────┘
```


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)
