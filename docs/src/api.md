# API Reference
`DecisionProgramming.jl` API reference.


## `influence_diagram.jl`
### Nodes
```@docs
Node
Name
AbstractNode
ChanceNode
DecisionNode
ValueNode
State
States
```

### Paths
```@docs
Path
ForbiddenPath
FixedPath
paths(::AbstractVector{State})
paths(::AbstractVector{State}, FixedPath)
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

### Utilities
```@docs
Utility
Utilities
```

### Path Utility
```@docs
AbstractPathUtility
DefaultPathUtility
```

### InfluenceDiagram
InfluenceDiagram
generate_arcs!
generate_diagram!
add_node!
ProbabilityMatrix
set_probability!
add_probabilities!
UtilityMatrix
set_utility!
add_utilities!

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
expected_value(::Model, ::InfluenceDiagram, ::PathCompatibilityVariables; ::Float64)
conditional_value_at_risk(::Model, ::InfluenceDiagram, ::PathCompatibilityVariables{N}, ::Float64; ::Float64) where N
```

### Decision Strategy from Variables
```@docs
LocalDecisionStrategy(::Node, ::Vector{VariableRef})
DecisionStrategy(::DecisionVariables)
```

## `analysis.jl`
```@docs
CompatiblePaths
CompatiblePaths(::InfluenceDiagram, ::DecisionStrategy, ::FixedPath)
UtilityDistribution
UtilityDistribution(::InfluenceDiagram, ::DecisionStrategy)
StateProbabilities
StateProbabilities(::InfluenceDiagram, ::DecisionStrategy)
StateProbabilities(::InfluenceDiagram, ::DecisionStrategy, ::Name, ::Name, ::StateProbabilities)
value_at_risk(::UtilityDistribution, ::Float64)
conditional_value_at_risk(::UtilityDistribution, ::Float64)
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
