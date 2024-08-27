# Contingent Portfolio Programming

!!! warning
    This example discusses adding constraints and decision variables to the Decision Programming formulation, as well as custom path utility calculation. Because of this, it is quite advanced compared to the earlier ones. 

## Description
[^1], section 4.2

> For instance, assume that the first-stage decisions specify which technology development projects will be started to generate patent-based intellectual property ( P ) for a platform. This intellectual property contributes subject to some uncertainties to the technical competitiveness ( T ) of the platform. In the second stage, it is possible to carry out application ( A ) development projects which, when completed, yield cash flows that depend on the market share of the platform. This market share ( M ) depends on the competitiveness of the platform and the number of developed applications. The aim is to maximize the cash flows from application projects less the cost of technology and application development projects.

## Influence Diagram: Projects
![](figures/contingent-portfolio-programming.svg)

The influence diagram of the contingent portfolio programming (CPP) problem.

There are $n_T$ technology development projects and $n_A$ application development projects.

Decision states to develop patents

$$d_i^P∈D_i^P=\{[q_1^P, q_2^P), [q_2^P, q_3^P), ..., [q_{|D^P|}^P, q_{|D^P|+1}^P)\}$$

Chance states of technical competitiveness $c_j^T∈C_j^T$

Decision states to develop applications

$$d_k^A∈D^A=\{[q_1^A, q_2^A), [q_2^A, q_3^A), ..., [q_{|D^A|}^A, q_{|D^A|+1}^A)\}$$

Chance states of market size $c_l^M∈C_l^M$

```julia
using Random
using JuMP, HiGHS
using DecisionProgramming

Random.seed!(42)

diagram = InfluenceDiagram()

add_node!(diagram, DecisionNode("DP", [], ["0-3 patents", "3-6 patents", "6-9 patents"]))
add_node!(diagram, ChanceNode("CT", ["DP"], ["low", "medium", "high"]))
add_node!(diagram, DecisionNode("DA", ["DP", "CT"], ["0-5 applications", "5-10 applications", "10-15 applications"]))
add_node!(diagram, ChanceNode("CM", ["CT", "DA"], ["low", "medium", "high"]))

generate_arcs!(diagram)
```

### Technical competitiveness probability

Probability of technical competitiveness $c_j^T$ given the range $d_i^P$: $ℙ(c_j^T∣d_i^P)∈[0,1]$. A high number of patents increases probability of high competitiveness and a low number correspondingly increases the probability of low competitiveness.

```julia
X_CT = ProbabilityMatrix(diagram, "CT")
X_CT[1, :] = [1/2, 1/3, 1/6]
X_CT[2, :] = [1/3, 1/3, 1/3]
X_CT[3, :] = [1/6, 1/3, 1/2]
add_probabilities!(diagram, "CT", X_CT)
```

### Market share probability

Probability of market share $c_l^M$ given the technical competitiveness $c_j^T$ and range $d_k^A$: $ℙ(c_l^M∣c_j^T,d_k^A)∈[0,1]$. Higher competitiveness and number of application projects both increase the probability of high market share.

```julia
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
```

### Generating the Influence Diagram

We are going to be using a custom objective function, and don't need the default path utilities for that.
```julia
generate_diagram!(diagram, default_utility=false)
```

## Decision Model: Portfolio Selection

We create the decision variables $z(s_j|s_{I(j)})$ and notice that the activation of paths that are compatible with the decision strategy is handled by the problem specific variables and constraints together with the custom objective function, eliminating the need for separate variables representing path activation.
```julia
model = Model()
z = DecisionVariables(model, diagram)
```

### Creating problem specific variables

We recommend reading Section 4.2. in [^1] for motivation and details of the formulation.


Technology project $t$ costs $I_t∈ℝ^+$ and generates $O_t∈ℕ$ patents.

Application project $a$ costs $I_a∈ℝ^+$ and generates $O_a∈ℕ$ applications. If completed, provides cash flow $V(a|c_l^M)∈ℝ^+.$

```julia

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
```

Decision variables $x^T(t)∈\{0, 1\}$ indicate which technologies are selected.

Decision variables $x^A(a∣d_i^P,c_j^T)∈\{0, 1\}$ indicate which applications are selected.

```julia
function variables(model::Model, dims::AbstractVector{Int}; binary::Bool=false)
    v = Array{VariableRef}(undef, dims...)
    for i in eachindex(v)
        v[i] = @variable(model, binary=binary)
    end
    return v
end

x_T = variables(model, [n_DP, n_T]; binary=true)
x_A = variables(model, [n_DP, n_CT, n_DA, n_A]; binary=true)
```

Number of patents $x^T(t) = ∑_i x_i^T(t) z(d_i^P)$

Number of applications $x^A(a∣d_i^P,c_j^T) = ∑_k x_k^A(a∣d_i^P,c_j^T) z(d_k^A|d_i^P,c_j^T)$

Helpful variables:

Large constant $M$ (e.g. $\frac{3}{2}\text{max}\{\sum_t O_t,\sum_a O_a\}$)

Small constant $\varepsilon$ = $\frac{1}{2}\text{min}\{O_t, O_a\}$

```julia
M = 20                      # a large constant
ε = 0.5*minimum([O_t O_a])  # a helper variable, allows using ≤ instead of < in constraints (28b) and (29b)
```

Limits $q_i^P$ and $q_k^A$ of the intervals

```julia
q_P = [0, 3, 6, 9]          # limits of the technology intervals
q_A = [0, 5, 10, 15]        # limits of the application intervals
```

Shorthand for the decision variables $z$

```julia
z_dP = z["DP"].z
z_dA = z["DA"].z
```


### Creating problem specific constraints

$$∑_t x_i^T(t) \le z(d_i^P)n_T, \quad \forall i$$
```julia
@constraint(model, [i=1:n_DP],
    sum(x_T[i,t] for t in 1:n_T) <= z_dP[i]*n_T)
```

$$∑_a x_k^A(a∣d_i^P,c_j^T) \le z(d_i^P)n_A, \quad \forall i,j,k$$
```julia
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA],
    sum(x_A[i,j,k,a] for a in 1:n_A) <= z_dP[i]*n_A)
```

$$∑_a x_k^A(a∣d_i^P,c_j^T) \le z(d_k^A|d_i^P,c_j^T)n_A, \quad \forall i,j,k$$
```julia
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA],
    sum(x_A[i,j,k,a] for a in 1:n_A) <= z_dA[i,j,k]*n_A)
```

$$q_i^P - (1-z(d_i^P))M \le \sum_t x_i^T(t)O_t \le q_{i+1}^P + (1-z(d_i^P))M - \varepsilon, \quad \forall i$$
```julia
@constraint(model, [i=1:n_DP],
    q_P[i] - (1 - z_dP[i])*M <= sum(x_T[i,t]*O_t[t] for t in 1:n_T))
@constraint(model, [i=1:n_DP],
    sum(x_T[i,t]*O_t[t] for t in 1:n_T) <= q_P[i+1] + (1 - z_dP[i])*M - ε)

```

$$q_k^A - (1-z(d_k^A|d_i^P,c_j^T))M \le \sum_a x_k^A(a∣d_i^P,c_j^T)O_a \le q_{k+1}^A + (1-z(d_k^A|d_i^P,c_j^T))M - \varepsilon, \quad \forall i,j,k$$
```julia
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA],
    q_A[k] - (1 - z_dA[i,j,k])*M <= sum(x_A[i,j,k,a]*O_a[a] for a in 1:n_A))
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA],
    sum(x_A[i,j,k,a]*O_a[a] for a in 1:n_A) <= q_A[k+1] + (1 - z_dA[i,j,k])*M - ε)
```

We can also model dependencies between the technology and application projects, e.g. application project $a$ can be completed only if technology project $t$ has been completed. This is done by adding constraints

$$x_k^A(a∣d_i^P,c_j^T) \le x_i^T(t), \quad \forall i,j,k$$

As an example, we state that application projects 1 and 2 require technology project 1, and application project 2 also requires technology project 2.

```julia
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA], x_A[i,j,k,1] <= x_T[i,1])
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA], x_A[i,j,k,2] <= x_T[i,1])
@constraint(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA], x_A[i,j,k,2] <= x_T[i,2])
```

$$x_i^T(t)∈\{0, 1\}, \quad \forall i$$

$$x_k^A(a∣d_i^P,c_j^T)∈\{0, 1\}, \quad \forall i,j,k$$

### Objective function

The path utility can be calculated as
$$\mathcal{U}(s) = \sum_a x_k^A(a∣d_i^P,c_j^T) (V(a|c_l^M) - I_a) - ∑_t x_i^T(t) I_t$$

However, using the expected value objective would lead to a quadratic objective function as the path utility formulation now contains decision variables. In order to keep the problem completely linear, we can use the objective formulation presented in [^1]:

$$\sum_i \left\{ \sum_{j,k,l} p(c_j^T \mid d_i^P) p(c_l^M \mid c_j^T, d_k^A) \left[\sum_a x_k^A(a \mid d_i^P,c_j^T) (V(a \mid c_l^M) - I_a)\right] - \sum_t x_i^T(t) I_t \right\}$$

```julia
patent_investment_cost = @expression(model, [i=1:n_DP], sum(x_T[i, t] * I_t[t] for t in 1:n_T))
application_investment_cost = @expression(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA], sum(x_A[i, j, k, a] * I_a[a] for a in 1:n_A))
application_value = @expression(model, [i=1:n_DP, j=1:n_CT, k=1:n_DA, l=1:n_CM], sum(x_A[i, j, k, a] * V_A[l, a] for a in 1:n_A))
@objective(model, Max, sum( sum( diagram.P(convert.(State, (i,j,k,l))) * (application_value[i,j,k,l] - application_investment_cost[i,j,k]) for j in 1:n_CT, k in 1:n_DA, l in 1:n_CM ) - patent_investment_cost[i] for i in 1:n_DP ))


```

### Solving the Model

```julia
optimizer = optimizer_with_attributes(
    () -> HiGHS.Optimizer()
)
set_optimizer(model, optimizer)
optimize!(model)
```

## Analyzing results

The optimal decision strategy and the utility distribution are printed. The strategy is to make 6-9 patents (state 3 in node 1) and 10-15 applications. The expected utility for this strategy is 1.08. Julia version 1.10.3 was used in random number generation (the version used might affect the results).

```julia
Z = DecisionStrategy(diagram, z)
S_probabilities = StateProbabilities(diagram, Z)
```

```julia-repl
julia> print_decision_strategy(diagram, Z, S_probabilities)
┌────────────────┐
│ Decision in DP │
├────────────────┤
│ 6-9 patents    │
└────────────────┘
┌─────────────────────┬────────────────────┐
│ State(s) of DP, CT  │ Decision in DA     │
├─────────────────────┼────────────────────┤
│ 6-9 patents, low    │ 10-15 applications │
│ 6-9 patents, medium │ 10-15 applications │
│ 6-9 patents, high   │ 10-15 applications │
└─────────────────────┴────────────────────┘
```

We use a custom path utility function to obtain the utility distribution.

```julia
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
```

```julia
U_distribution = UtilityDistribution(diagram, Z)
```

```julia-repl
julia> print_utility_distribution(U_distribution)
┌───────────┬─────────────┐
│   Utility │ Probability │
│   Float64 │     Float64 │
├───────────┼─────────────┤
│ -2.338179 │    0.152778 │
│ -0.130143 │    0.291667 │
│  2.650091 │    0.555556 │
└───────────┴─────────────┘
```

```julia-repl
julia> print_statistics(U_distribution)
┌──────────┬────────────┐
│     Name │ Statistics │
│   String │    Float64 │
├──────────┼────────────┤
│     Mean │   1.077093 │
│      Std │   1.892543 │
│ Skewness │  -0.654557 │
│ Kurtosis │  -1.066341 │
└──────────┴────────────┘
```

## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)
