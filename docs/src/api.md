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
paths(::AbstractVector{State}, ::FixedPath)
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
```@docs
InfluenceDiagram
add_node!
ProbabilityMatrix
ProbabilityMatrix(::InfluenceDiagram, ::Name)
add_probabilities!
UtilityMatrix
UtilityMatrix(::InfluenceDiagram, ::Name)
add_utilities!
generate_arcs!
generate_diagram!
RJT
indices
I_j_indices
indices_in_vector
get_values
get_keys
num_states
```

### ForbiddenPath and FixedPath outer construction functions
```@docs
ForbiddenPath(::InfluenceDiagram, ::Vector{Name}, ::Vector{NTuple{N, Name}}) where N
FixedPath(::InfluenceDiagram, ::Dict{Name, Name})
```

### Decision Strategy
```@docs
LocalDecisionStrategy
LocalDecisionStrategy(::AbstractRNG, ::InfluenceDiagram, ::Name)
DecisionStrategy
```

## `decision_model.jl`
### Decision Model
```@docs
DecisionVariables
PathCompatibilityVariables
lazy_probability_cut
```

### Objective Functions
```@docs
expected_value(::Model, ::InfluenceDiagram, ::PathCompatibilityVariables)
conditional_value_at_risk(::Model, ::InfluenceDiagram, ::PathCompatibilityVariables{N}, ::Float64; ::Float64) where N
```

### Decision Strategy from Variables
```@docs
LocalDecisionStrategy(::Node, ::Array{VariableRef})
DecisionStrategy(::InfluenceDiagram, ::OrderedDict{Name, DecisionProgramming.DecisionVariable})
```

### RJT model
```@docs
RJTVariables
expected_value(::Model, ::InfluenceDiagram, ::DecisionProgramming.RJTVariables)
conditional_value_at_risk(::Model, ::InfluenceDiagram, ::DecisionProgramming.RJTVariables, ::Float64)
generate_model
```

## `heuristics.jl`
### Single policy update
```@docs
randomStrategy
singlePolicyUpdate
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
print_node
print_diagram
mermaid
```
