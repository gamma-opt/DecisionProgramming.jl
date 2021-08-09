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
paths(::AbstractVector{State})
paths(::AbstractVector{State}, ::Dict{Node, State})
ForbiddenPath
```

### Probabilities
```@docs
Probabilities
```

### Path Probability
```@docs
AbstractPathProbability
DefaultPathProbability
```

### Consequences
```@docs
Consequences
```

### Path Utility
```@docs
AbstractPathUtility
DefaultPathUtility
```

### Decision Strategy
```@docs
LocalDecisionStrategy
DecisionStrategy
```


## `decision_model.jl`
### Decision Model
```@docs
DecisionVariables
PathCompatibilityVariables
lazy_probability_cut(::Model, ::PathCompatibilityVariables, ::AbstractPathProbability)
```

### Objective Functions
```@docs
PositivePathUtility
NegativePathUtility
expected_value(::Model, ::PathCompatibilityVariables, ::AbstractPathUtility, ::AbstractPathProbability; ::Float64)
conditional_value_at_risk(::Model, ::PathCompatibilityVariables{N}, ::AbstractPathUtility, ::AbstractPathProbability, ::Float64; ::Float64) where N
```

### Decision Strategy from Variables
```@docs
LocalDecisionStrategy(::Node, ::Vector{VariableRef})
DecisionStrategy(::DecisionVariables)
```

## `analysis.jl`
```@docs
CompatiblePaths
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
random_diagram(::AbstractRNG, ::Int, ::Int, ::Int, ::Int, ::Int)
States(::AbstractRNG, ::Vector{State}, ::Int)
Probabilities(::AbstractRNG, ::ChanceNode, ::States; ::Int)
Consequences(::AbstractRNG, ::ValueNode, ::States; ::Float64, ::Float64)
LocalDecisionStrategy(::AbstractRNG, ::DecisionNode, ::States)
```
