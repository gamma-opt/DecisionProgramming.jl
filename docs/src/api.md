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
States(::Vector{Tuple{State, Vector{Node}}})
validate_influence_diagram
```

### Paths
```@docs
Path
paths
```

### Probabilities
```@docs
Probabilities
Probabilities(::Path)
```

### Path Probability
```@docs
AbstractPathProbability
DefaultPathProbability
DefaultPathProbability(::Path)
```

### Consequences
```@docs
Consequences
Consequences(::Path)
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
DecisionModel(::States, ::Vector{DecisionNode}, ::AbstractPathProbability; ::Bool)
probability_cut(::DecisionModel, ::States, ::AbstractPathProbability)
active_paths_cut(::DecisionModel, ::States, ::AbstractPathProbability; ::Float64)
```

### Objective Functions
```@docs
expected_value(::DecisionModel, ::States, ::AbstractPathUtility)
conditional_value_at_risk(::DecisionModel, ::States, ::AbstractPathUtility, ::Float64)
```

### Decision Strategy
```@docs
LocalDecisionStrategy
LocalDecisionStrategy(::Vector{VariableRef})
LocalDecisionStrategy(::Path)
DecisionStrategy
DecisionStrategy(::DecisionModel, ::Vector{DecisionNode})
```

## `analysis.jl`
```@docs
ActivePaths
UtilityDistribution
UtilityDistribution(::States, ::AbstractPathProbability, ::AbstractPathUtility, ::DecisionStrategy)
StateProbabilities
StateProbabilities(::States, ::AbstractPathProbability, ::DecisionStrategy)
StateProbabilities(::States, ::AbstractPathProbability, ::DecisionStrategy, ::Node, ::State, ::StateProbabilities)
value_at_risk(::Vector{Float64}, ::Vector{Float64}, ::Float64)
conditional_value_at_risk(::Vector{Float64}, ::Vector{Float64}, ::Float64)
```

## `printing.jl`
```@docs
print_decision_strategy
print_utility_distribution
print_state_probabilities
print_statistics
print_risk_measures
```

## `random.jl`
```@docs
random_diagram(::AbstractRNG, ::Int, ::Int, ::Int, ::Int)
States(::AbstractRNG, ::Vector{State}, ::Int)
Probabilities(::AbstractRNG, ::ChanceNode, ::States)
Consequences(::AbstractRNG, ::ValueNode, ::States; ::Float64, ::Float64)
LocalDecisionStrategy(::AbstractRNG, ::DecisionNode, ::States)
```
