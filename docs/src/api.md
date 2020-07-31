# API Reference
`DecisionProgramming.jl` API reference.

## `model.jl`
### Influence Diagram
```@docs
Node
State
InfluenceDiagram
InfluenceDiagram(::Vector{Node}, ::Vector{Node}, ::Vector{Node}, ::Vector{Pair{Node, Node}}, ::Vector{State})
```

### Paths
```@docs
Path
paths
```

### Probabilities
```@docs
Probability
Probabilities
Probabilities(::InfluenceDiagram, ::Dict{Node, Probability})
```

### Consequences
```@docs
Consequence
Consequences
Consequences(::InfluenceDiagram, ::Dict{Node, Probability})
```

### Path Probability
```@docs
PathProbability
PathProbability(::Path)
```

### Path Utility
```@docs
PathUtility
PathUtility(::Path)
positive_affine(::PathUtility, ::Path)
```

### Decision Model
```@docs
variables
DecisionModel
DecisionModel(::InfluenceDiagram, ::PathProbability; ::Bool)
probability_sum_cut(::DecisionModel, ::PathProbability)
number_of_paths_cut(::DecisionModel, ::InfluenceDiagram, ::PathProbability; ::Float64)
```

### Objective Functions
```@docs
expected_value(::DecisionModel, ::InfluenceDiagram, ::PathUtility)
conditional_value_at_risk(::DecisionModel, ::InfluenceDiagram, ::PathUtility, ::Float64)
```

### Decision Strategy
```@docs
DecisionStrategy
DecisionStrategy(::DecisionModel)
```

## `analysis.jl`
```@docs
ActivePaths
UtilityDistribution
UtilityDistribution(::InfluenceDiagram, ::PathProbability, ::PathUtility, ::DecisionStrategy)
StateProbabilities
StateProbabilities(::InfluenceDiagram, ::PathProbability, ::DecisionStrategy)
StateProbabilities(::InfluenceDiagram, ::PathProbability, ::DecisionStrategy, ::Node, ::State, ::StateProbabilities)
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
InfluenceDiagram(::AbstractRNG, ::Int, ::Int, ::Int, ::Int, ::Vector{Int})
Probabilities(::AbstractRNG, ::InfluenceDiagram)
Consequences(::AbstractRNG, ::InfluenceDiagram; ::Float64, ::Float64)
```
