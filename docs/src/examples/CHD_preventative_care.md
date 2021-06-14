# CHD preventative care allocation
## Description
 The goal in this optimisation problem is to determine an optimal decision strategy for the testing and treatment decisions involved in providing preventative care for coronary heart disease (CHD). The optimality is evaluated from the perspective of the national health care system and is measured through net monetary benefit (NMB). The tests available in this model are the traditional risk score (TRS) and the genetic risk score (GRS) and the form of preventative care is statin treatment. The description of the CHD preventative care allocation problem is below. This description is from [^1] from section 3.2.

> The problem setting is such that the patient is assumed to have a prior risk estimate. A risk estimate is a prediction of the patient’s chance of having a CHD event in the next ten years. The risk estimates are grouped into risk levels, which range from 0% to 100%. The first testing decision is made based on the prior risk estimate. The first testing decision entails deciding whether TRS or GRS should be performed or if no testing is needed. If a test is conducted, the risk estimate is updated and based on the new information, the second testing decision is made. The second testing decision entails deciding whether further testing should be conducted or not. The second testing decision is constrained so that the same test which was conducted in the first stage cannot be repeated. If a second test is conducted, the risk estimate is updated again. The treatment decision – dictating whether the patient receives statin therapy or not – is made based on the resulting risk estimate of this testing process. Note that if no tests are conducted, the treatment decision is made based on the prior risk estimate.

In this example, we will showcase the subproblem, which optimises the decision strategy given a single prior risk level. The chosen risk level in this example is 12%. The solution to the main problem is found in [^1].

## Influence Diagram
![](figures/CHD_preventative_care.svg)

The influence diagram representation of the problem is seen above. The chance nodes $R$ represent the patient's risk estimate – the prior risk estimate being $R0$. The risk estimate nodes $R0$, $R1$ and $R2$ have 101 states $R = \{0\%, 1\%, ..., 100\%\}$, which are the discretised risk levels for the risk estimates.

The risk estimate is updated according to the first and second test decisions, which are represented by decision nodes $T1$ and $T2$. These nodes have states $T = \{\text{TRS, GRS, no test}\}$. The health of the patient represented by chance node $H$ also affects the update of the risk estimate. In this model, the health of the patient indicates whether they will have a CHD event in the next ten years or not. Thus, the node has states $H = \{\text{CHD, no CHD}\}$. The treatment decision is represented by node $TD$ and it has states $TD = \{\text{treatment, no treatment}\}$.


The prior risk estimate represented by node $R0$ influences the health node $H$, because in the model we make the assumption that the prior risk estimate accurately describes the probability of having a CHD event.

The value nodes in the model are $TC$ and $HB$. Node $TC$ represents the testing costs incurred due to the testing decisions $T1$ and $T2$. Node $HB$ represents the health benefits achieved. The test costs and health benefits are measured in quality-adjusted life-years. These parameter values were evaluated in the study [^2].

We begin by declaring the chosen prior risk level and reading the conditional probability data for the tests. The risk level 12% is referred to as 13 because indexing begins from 1 and the first risk level is 0\%. Note also that the sample data in this repository is dummy data due to distribution restrictions on the real data. We also define functions ```update_risk_distribution ```, ```state_probabilities``` and ```analysing_results ```. These functions will be discussed in the following sections.

```julia
using Logging
using JuMP, Gurobi
using DecisionProgramming
using CSV, DataFrames, PrettyTables


const chosen_risk_level = 13
const data = CSV.read("risk_prediction_data.csv", DataFrame)

function update_risk_distribution(prior::Int64, t::Int64)...
end

function state_probabilities(risk_p::Array{Float64}, t::Int64, h::Int64, prior::Int64)...
end

function analysing_results(Z::DecisionStrategy, sprobs::StateProbabilities)...
end
```


 Then we begin defining the decision programming model. First, we define the node indices and states:

```julia
const R0 = 1
const H  = 2
const T1 = 3
const R1 = 4
const T2 = 5
const R2 = 6
const TD = 7
const TC = 8
const HB = 9


const H_states = ["CHD", "no CHD"]
const T_states = ["TRS", "GRS", "no test"]
const TD_states = ["treatment", "no treatment"]
const R_states = map( x -> string(x) * "%", [0:1:100;])
const TC_states = ["TRS", "GRS", "TRS & GRS", "no tests"]
const HB_states = ["CHD & treatment", "CHD & no treatment", "no CHD & treatment", "no CHD & no treatment"]

@info("Creating the influence diagram.")
S = States([
    (length(R_states), [R0, R1, R2]),
    (length(H_states), [H]),
    (length(T_states), [T1, T2]),
    (length(TD_states), [TD])
])

C = Vector{ChanceNode}()
D = Vector{DecisionNode}()
V = Vector{ValueNode}()
X = Vector{Probabilities}()
Y = Vector{Consequences}()
```


Next, we define the nodes with their information sets and corresponding probabilities for chance nodes and consequences for value nodes.

### Prior risk estimate and health of the patient

In this subproblem, the prior risk estimate is given and therefore the node $R0$ is in effect a deterministic node. In decision programming a deterministic node is added as a chance node for which the probability of one state is set to one and the probabilities of the rest of the states are set to zero. In this case

$$ℙ(R0 = 12\%)=1$$
and
$$ℙ(R0 \neq 12\%)= 0. $$

Notice also that node $R0$ is the first node in the influence diagram, meaning that its information set $I(R0)$ is empty. In decision programming we add node $R0$ and its state probabilities as follows:
```julia
I_R0 = Vector{Node}()
X_R0 = zeros(S[R0])
X_R0[chosen_risk_level] = 1
push!(C, ChanceNode(R0, I_R0))
push!(X, Probabilities(R0, X_R0))
```

Next we add node $H$ and its state probabilities. For modeling purposes, we define the information set of node $H$ to include the prior risk node $R0$. We set the probability of the patient experiencing a CHD event in the next ten years according to the prior risk level such that

$$ℙ(H = \text{CHD} | R0 = \alpha) = \alpha.$$

We define the probability of the patient not experiencing a CHD event in the next ten years as the complement event.

$$ℙ(H = \text{no CHD} | R0 = \alpha) = 1 - \alpha$$

Since node $R0$ is actually deterministic, defining the health node $H$ in this way means that in the our model the patient has 12% probability of experiencing a CHD event and 88% chance of not suffering one.

Node $H$ and its probabilities are added in the following way.

```julia
I_H = [R0]
X_H = zeros(S[R0], S[H])
X_H[:, 1] = data.risk_levels
X_H[:, 2] = 1 .- X_H[:, 1]
push!(C, ChanceNode(H, I_H))
push!(X, Probabilities(H, X_H))
```

### Test decisions and updating the risk estimate

The node representing the first test decision is added to the model in the following way.

```julia
I_T1 = [R0]
push!(D, DecisionNode(T1, I_T1))
```

The probabilities of the states of node $R1$ are determined by calculating the updated risk estimates after a test is performed and aggregating these probabilities into the risk levels. The update of the risk estimate is calculated using the function ```update_risk_distribution```, which calculates the posterior probability distribution for a given health state, test and prior risk estimate.

$$ \textit{risk estimate} = P(\text{CHD} \mid \text{test result}) = \frac{P(\text{test result} \mid \text{CHD})P(\text{CHD})}{P(\text{test result})}$$

The probabilities $P(\text{test result} \mid \text{CHD})$ are test specific and these are read from the CSV data file. The updated risk estimates are aggregated according to the risk levels. These aggregated probabilities are then the state probabilities of node $R1$. The aggregating is done using function ```state_probabilities```.

The node $R1$ and its probabilities are added in the following way.

```julia
I_R1 = [R0, H, T1]
X_R1 = zeros(S[I_R1]..., S[R1])
for s_R0 = 1:101, s_H = 1:2, s_T1 = 1:3
    X_R1[s_R0, s_H, s_T1, :] =  state_probabilities(update_risk_distribution(s_R0, s_T1), s_T1, s_H, s_R0)
end
push!(C, ChanceNode(R1, I_R1))
push!(X, Probabilities(R1, X_R1))
```

Now we add node $T2$ and node $R2$ in a similar fashion as we added $T1$ and $R1$ above.
```julia
I_T2 = [R1]
push!(D, DecisionNode(T2, I_T2))


I_R2 = [H, R1, T2]
X_R2 = zeros(S[I_R2]..., S[R2])
for s_R1 = 1:101, s_H = 1:2, s_T2 = 1:3
    X_R2[s_H, s_R1, s_T2, :] =  state_probabilities(update_risk_distribution(s_R1, s_T2), s_T2, s_H, s_R1)
end
push!(C, ChanceNode(R2, I_R2))
push!(X, Probabilities(R2, X_R2))
```

We also add the treatment decision node $TD$. The treatment decision is made based on the resulting risk estimate of the testing process.

```julia
I_TD = [R2]
push!(D, DecisionNode(TD, I_TD))
```

### Test costs and health benefits

To add the value node $TC$, which represents testing costs, we need to define the consequences of its different information states. The node and the consequences are added in the following way.

```julia
I_TC = [T1, T2]
Y_TC = zeros(S[I_TC]...)
cost_TRS = -0.0034645
cost_GRS = -0.004
cost_forbidden = 0     #the cost of forbidden test combinations is negligible
Y_TC[1 , 1] = cost_forbidden
Y_TC[1 , 2] = cost_TRS + cost_GRS
Y_TC[1, 3] = cost_TRS
Y_TC[2, 1] =  cost_GRS + cost_TRS
Y_TC[2, 2] = cost_forbidden
Y_TC[2, 3] = cost_GRS
Y_TC[3, 1] = cost_TRS
Y_TC[3, 2] = cost_GRS
Y_TC[3, 3] = 0
push!(V, ValueNode(TC, I_TC))
push!(Y, Consequences(TC, Y_TC))
```

The health benefits achieved by the strategy are dictated by whether treatment is administered and by the health of the patient. We add the final node to the model.

```julia
I_HB = [H, TD]
Y_HB = zeros(S[I_HB]...)
Y_HB[1 , 1] = 6.89713671259061  # sick & treat
Y_HB[1 , 2] = 6.65436854256236  # sick & don't treat
Y_HB[2, 1] = 7.64528451705134   # healthy & treat
Y_HB[2, 2] =  7.70088349200034  # healthy & don't treat
push!(V, ValueNode(HB, I_HB))
push!(Y, Consequences(HB, Y_HB))
```

### Validating Influence Diagram
Before creating the decision model, we need to validate the influence diagram and sort the nodes, probabilities and consequences in increasing order by the node indices.

```julia
validate_influence_diagram(S, C, D, V)
sort!.((C, D, V, X, Y), by = x -> x.j)
```

We also define the path probability and the path utility. We use the default path utility, which is the sum of the consequences of the path.
```julia
P = DefaultPathProbability(C, X)
U = DefaultPathUtility(V, Y)
```


## Decision Model
We begin by defining our model and declaring the decision variables.
```julia
model = Model()
z = DecisionVariables(model, S, D)
```

In this problem, we want to forbid the model from choosing paths where the same test is repeated twice and the paths where the first testing decision is to not perform a test but then the second testing decision is to conduct a test. We forbid the paths by declaring these combinations of states as forbidden paths and then giving them to the function that declares the path probability variables.

We also choose a scale factor of 1000, which will be used to scale the path probabilities. The probabilities need to be scaled because in this specific problem they are very small since the $R$ nodes have many states. Scaling the probabilities helps the solver find an optimal solution.

We declare the path probability variables and give the forbidden testing strategies and the scale factor to the function as additional information.

```julia
forbidden_tests = ForbiddenPath[([T1,T2], Set([(1,1),(2,2),(3,1),(3,2)]))]
scale_factor = 1000.0
π_s = PathProbabilityVariables(model, z, S, P; hard_lower_bound = true, forbidden_paths = forbidden_tests, probability_scale_factor = scale_factor)
```

We define the objective function as the expected value.
```julia
EV = expected_value(model, π_s, U)
@objective(model, Max, EV)
```

We set up the solver for the problem and optimise it. 
```julia
optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
    "TimeLimit" => 100,
    "IntFeasTol"=> 1e-9,
    "MIPGap" => 1e-6
)
set_optimizer(model, optimizer)
optimize!(model)
```



## Analyzing Results

### Decision Strategy
We obtain the optimal decision strategy from the z variable values.
```julia
Z = DecisionStrategy(z)
```

We use the function ```analysing_results``` to visualise the results in order to inspect the decision strategy. We use this tailor made function merely for convinience. From the printout, we can see that when the prior risk level is 12% the optimal decision strategy is to first perform TRS testing. At the second decision stage, GRS should be conducted if the updated risk estimate is between 16% and 28% and otherwise no further testing should be conducted. Treatment should be provided to those who have a final risk estimate greater than 18%. Notice that the blank spaces in the table are states which have a probability of zero, which means that given this data it is impossible for the patient to have their risk estimate updated to those risk levels.

```julia
sprobs = StateProbabilities(S, P, Z)
```
```julia
julia> println(analysing_results(Z, sprobs))
┌─────────────────┬────────┬────────┬────────┐
│ Information_set │     T1 │     T2 │     TD │
│          String │ String │ String │ String │
├─────────────────┼────────┼────────┼────────┤
│              0% │        │      3 │      2 │
│              1% │        │      3 │      2 │
│              2% │        │        │      2 │
│              3% │        │      3 │      2 │
│              4% │        │        │        │
│              5% │        │        │        │
│              6% │        │      3 │      2 │
│              7% │        │      3 │      2 │
│              8% │        │        │      2 │
│              9% │        │        │      2 │
│             10% │        │      3 │      2 │
│             11% │        │      3 │      2 │
│             12% │      1 │        │      2 │
│             13% │        │      3 │      2 │
│             14% │        │      3 │      2 │
│             15% │        │        │      2 │
│             16% │        │      2 │      2 │
│             17% │        │      2 │      2 │
│             18% │        │      2 │      1 │
│             19% │        │        │      1 │
│             20% │        │        │      1 │
│             21% │        │      2 │      1 │
│             22% │        │      2 │      1 │
│             23% │        │      2 │      1 │
│             24% │        │        │      1 │
│             25% │        │        │      1 │
│             26% │        │        │      1 │
│             27% │        │        │      1 │
│             28% │        │      3 │      1 │
│             29% │        │      3 │      1 │
│             30% │        │        │      1 │
│        ⋮        │   ⋮     │   ⋮    │   ⋮    │
└─────────────────┴────────┴────────┴────────┘
                               70 rows omitted
```

### State Probabilities
We can inspect the state probabilities for the optimal decision strategy $Z$. These describe the probability of each state in each node, given the decision strategy $Z$.

```julia
julia> print_state_probabilities(sprobs, [R0, R1, R2])
┌───────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────
│  Node │  State 1 │  State 2 │  State 3 │  State 4 │  State 5 │  State 6 │
│ Int64 │  Float64 │  Float64 │  Float64 │  Float64 │  Float64 │  Float64 │ ⋯
├───────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────
│     1 │ 0.000000 │ 0.000000 │ 0.000000 │ 0.000000 │ 0.000000 │ 0.000000 │ ⋯
│     4 │ 0.001030 │ 0.355590 │ 0.000000 │ 0.130765 │ 0.000000 │ 0.000000 │ ⋯
│     6 │ 0.001420 │ 0.355590 │ 0.000966 │ 0.132009 │ 0.000000 │ 0.000000 │ ⋯
└───────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────
                                                            96 columns omitted
```

### Utility Distribution

We can also print the utility distribution for the optimal strategy and some basic statistics for the distribution.

```julia
udist = UtilityDistribution(S, P, U, Z)
```

```julia
julia> print_utility_distribution(udist)
┌──────────┬─────────────┐
│  Utility │ Probability │
│  Float64 │     Float64 │
├──────────┼─────────────┤
│ 6.646904 │    0.005318 │
│ 6.650904 │    0.038707 │
│ 6.889672 │    0.011602 │
│ 6.893672 │    0.064374 │
│ 7.637820 │    0.034188 │
│ 7.641820 │    0.073974 │
│ 7.693419 │    0.035266 │
│ 7.697419 │    0.736573 │
└──────────┴─────────────┘
```
```julia
julia> print_statistics(udist)
┌──────────┬────────────┐
│     Name │ Statistics │
│   String │    Float64 │
├──────────┼────────────┤
│     Mean │   7.583923 │
│      Std │   0.291350 │
│ Skewness │  -2.414877 │
│ Kurtosis │   4.059711 │
└──────────┴────────────┘
```


## References
[^1]: Hankimaa H. (2021). Optimising the use of genetic testing in prevention of CHD using Decision Programming. http://urn.fi/URN:NBN:fi:aalto-202103302644

[^2]: Hynninen Y. (2019). Value of genetic testing in the prevention of coronary heart disease events. PLOS ONE, 14(1):1–16. https://doi.org/10.1371/journal.pone.0210010
