# API Reference
`DecisionProgramming.jl` API reference.

## `influence_diagram.jl`
### Nodes
```@docs
Node
ChanceNode
DecisionNode
ValueNode
State
States
```

### Paths
```@docs
Path
paths
```

### Probabilities
```@docs
Probabilities
getindex(::Probabilities, ::Path)
```

### Consequences
```@docs
Consequences
getindex(::Consequences, ::Path)
```

### Path Probability
```@docs
PathProbability
PathProbability(::Path)
```

### Path Utility
```@docs
AbstractPathUtility
DefaultPathUtility
DefaultPathUtility(::Path)
```


## `decision_model.jl`
### Decision Model
```@docs
PositivePathUtility
PositivePathUtility(::Path)
variables
DecisionModel
DecisionModel(::States, ::Vector{DecisionNode}, ::PathProbability; ::Bool)
probability_sum_cut(::DecisionModel, ::States, ::PathProbability)
number_of_paths_cut(::DecisionModel, ::States, ::PathProbability; ::Float64)
```

### Objective Functions
```@docs
expected_value(::DecisionModel, ::States, ::AbstractPathUtility)
conditional_value_at_risk(::DecisionModel, ::States, ::AbstractPathUtility, ::Float64)
```

### Decision Strategy
```@docs
DecisionStrategy
DecisionStrategy(::Vector{VariableRef})
DecisionStrategy(::Path)
GlobalDecisionStrategy
GlobalDecisionStrategy(::DecisionModel, ::Vector{DecisionNode})
```

## `analysis.jl`
```@docs
ActivePaths
UtilityDistribution
UtilityDistribution(::States, ::PathProbability, ::AbstractPathUtility, ::GlobalDecisionStrategy)
StateProbabilities
StateProbabilities(::States, ::PathProbability, ::GlobalDecisionStrategy)
StateProbabilities(::States, ::PathProbability, ::GlobalDecisionStrategy, ::Node, ::State, ::StateProbabilities)
```

## `printing.jl`
```@docs
print_decision_strategy
print_utility_distribution
print_state_probabilities
print_statistics
print_risk_measures
value_at_risk(::Vector{Float64}, ::Vector{Float64}, ::Float64)
conditional_value_at_risk(::Vector{Float64}, ::Vector{Float64}, ::Float64)
```

## `random.jl`
```@docs
random_diagram(::AbstractRNG, ::Int, ::Int, ::Int, ::Int)
States(::AbstractRNG, ::Vector{State}, ::Int)
Probabilities(::AbstractRNG, ::ChanceNode, ::States)
Consequences(::AbstractRNG, ::ValueNode, ::States; ::Float64, ::Float64)
DecisionStrategy(::AbstractRNG, ::DecisionNode, ::States)
```
