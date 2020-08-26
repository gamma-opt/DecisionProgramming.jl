# Used Car Buyer
## Description
To illustrate the basic functionality of Decision Programming, we implement a version of the used car buyer problem in [^1]. In this problem, Joe is buying a used car. The car's price is 1000 USD (US dollars), and its value is 1100 USD. Joe's base profit on the car is thus 100 USD. However, Joe knows that the car is a "lemon", meaning that it has defects in 6 major systems, with a 20% probability. With the remaining 80% probability, the car is a "peach", and it has a defect in only one of the systems.

The repair costs for a peach are only 40 USD, decreasing Joe's profit to 60  USD. However, the costs for a lemon are 200 USD, resulting in a total loss of 100 USD. We can now formulate an influence diagram of Joe's initial problem. We present the influence diagram in the figure below. In an influence diagram, circle nodes such as $O$ are called **chance nodes**, representing uncertainty. Node $O$ is a chance node representing the state of the car, lemon or peach. Square nodes such as $A$ are **decision nodes**, representing decisions. Node $A$ represents the decision to buy or not to buy the car. The diamond-shaped **value node** $V$ denotes the utility calculation in the problem. For Joe, the utility function is the expected monetary value. The arrows or **arcs** show connections between nodes. The two arcs in this diagram point to the value node, meaning that the monetary value depends on the state of the car and the purchase decision.

![\label{used-car-buyer-1}](figures/used-car-buyer-1.svg)

We can easily determine the optimal strategy for this problem. If Joe decides not to buy the car, his profit is zero. If he buys the car, with 20% probability he loses 100 USD and with an 80% probability he profits 60 USD. Therefore, the expected profit for buying the car is 28 USD, which is higher than the zero profit of not buying. Thus, Joe should buy the car.

We now add two new features to the problem. A stranger approaches Joe and offers to tell Joe whether the car is a lemon or a peach for 25 USD. Additionally, the car dealer offers a guarantee plan which costs 60 USD and covers 50% of the repair costs. Joe notes that this is not a very good deal, and the dealer includes an anti-lemon feature: if the total repair cost exceeds 100 USD, the quarantee will fully cover the repairs.

## Influence diagram
![\label{used-car-buyer-2}](figures/used-car-buyer-2.svg)

We present the new influence diagram above. The decision node $T$ denotes the decision to accept or decline the stranger's offer, and $R$ is the outcome of the test. We introduce new value nodes $V_1$ and $V_2$ to represent the testing costs and the base profit from purchasing the car. Additionally, the decision node $A$ now can choose to buy with a guarantee.

```julia
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
```

We start by defining the influence diagram structure. The decision and chance nodes, as well as their states, are defined in the first block. Next, the influence diagram parameters consisting of the node sets and the state spaces of the nodes are defined.

### Car's State

The chance node $O$ is defined by its information set $I(O)$ and probability distribution $X_O$. As seen in the influence diagram, the information set is empty and the node is a root node. The probability distribution is thus simply defined over the two states of $O$.

```julia
I_O = Vector{Node}()
X_O = [0.2, 0.8]
push!(C, ChanceNode(O, I_O))
push!(X, Probabilities(X_O))
```

### Stranger's Offer Decision

A decision node is simply defined by its information state.

```julia
I_T = Vector{Node}()
push!(D, DecisionNode(T, I_T))
```

### Test's Outcome

The second chance node, $R$, has nodes $O$ and $T$ in its information set, and the probabilities $ℙ(s_j∣𝐬_{I(j)})$ must thus be defined for all combinations of states in $O$, $T$ and $R$.

```julia
I_R = [O, T]
X_R = zeros(S[O], S[T], S[R])
X_R[1, 1, :] = [1,0,0]
X_R[1, 2, :] = [0,1,0]
X_R[2, 1, :] = [1,0,0]
X_R[2, 2, :] = [0,0,1]
push!(C, ChanceNode(R, I_R))
push!(X, Probabilities(X_R))
```

### Purchace Decision
```julia
I_A = [R]
push!(D, DecisionNode(A, I_A))
```

### Testing Cost

The value nodes are defined similarly as the chance nodes, except that instead of probabilities, we define consequences $Y_j(𝐬_{I(j)})$.

```julia
I_V1 = [T]
Y_V1 = [0.0, -25.0]
push!(V, ValueNode(5, I_V1))
push!(Y, Consequences(Y_V1))
```

### Base Profit of Purchase
```julia
I_V2 = [A]
Y_V2 = [100.0, 40.0, 0.0]
push!(V, ValueNode(6, I_V2))
push!(Y, Consequences(Y_V2))
```

### Repairing Cost
```julia
I_V3 = [O, A]
Y_V3 = [-200.0 0.0 0.0;
        -40.0 -20.0 0.0]
push!(V, ValueNode(7, I_V3))
push!(Y, Consequences(Y_V3))
```

### Validating Influence Diagram
Validate influence diagram and sort nodes, probabilities and consequences

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

We continue by defining the probabilities associated with chance nodes and utilities (consequences) associated with value nodes. The rows of the consequence matrix Y_V3 correspond to the state of the car, while the columns correspond to the decision made in node $A$.

```julia
P = DefaultPathProbability(C, X)
U = DefaultPathUtility(V, Y)
```


## Decision Model
We then construct the decision model using the DecisionProgramming.jl package.

```julia
model = DecisionModel(S, D, P)
EV = expected_value(model, S, U)
@objective(model, Max, EV)
```

We can perform the optimization using an optimizer such as Gurobi.

```julia
optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)
```


## Analyzing Results
### Decision Strategy
Once the model is solved, we obtain the following decision strategy:

```julia
Z = DecisionStrategy(model, D)
```

```julia-repl
julia> print_decision_strategy(S, Z)
┌────────┬────┬───┐
│  Nodes │ () │ 2 │
├────────┼────┼───┤
│ States │ () │ 2 │
└────────┴────┴───┘
┌────────┬──────┬───┐
│  Nodes │ (3,) │ 4 │
├────────┼──────┼───┤
│ States │ (1,) │ 3 │
│ States │ (2,) │ 2 │
│ States │ (3,) │ 1 │
└────────┴──────┴───┘
```

To start explaining this output, let's take a look at the top table. On the right, we have the decision node 2. We defined earlier that the node $T$ is node number 2. On the left, we have the information set of that decision node, which is empty. The strategy in the first decision node is to choose alternative 2, which we defined to be testing the car.

In the bottom table, we have node number 4 (node $A$) and its predecessor, node number 3 (node $R$). The first row, where we obtain no test result, is invalid for this strategy since we tested the car. If the car is a lemon, Joe should buy the car with a guarantee (alternative 2), and if it is a peach, buy the car without guarantee (alternative 1).

### Utility Distribution
```julia
udist = UtilityDistribution(S, P, U, Z)
```

```julia-repl
julia> print_utility_distribution(udist)
┌───────────┬─────────────┐
│   Utility │ Probability │
│   Float64 │     Float64 │
├───────────┼─────────────┤
│ 15.000000 │    0.200000 │
│ 35.000000 │    0.800000 │
└───────────┴─────────────┘
```

From the utility distribution, we can see that Joe's profit with this strategy is 15 USD, with a 20% probability (the car is a lemon) and 35 USD with an 80% probability (the car is a peach).

```julia-repl
julia> print_statistics(udist)
┌──────────┬────────────┐
│     Name │ Statistics │
│   String │    Float64 │
├──────────┼────────────┤
│     Mean │  31.000000 │
│      Std │   8.000000 │
│ Skewness │  -1.500000 │
│ Kurtosis │   0.250000 │
└──────────┴────────────┘
```

The expected profit is thus 31 USD.


## References
[^1]: Howard, R. A. (1977). The used car buyer. Reading in Decision Analysis, 2nd Ed. Stanford Research Institute, Menlo Park, CA.
