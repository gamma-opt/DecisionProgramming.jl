# API Reference
## model.jl
```@docs
InfluenceDiagram
Probabilities
Consequences
UtilityFunction
DecisionModel
DecisionStrategy
```

```@docs
paths
path_probability
```

```@docs
InfluenceDiagram(::Vector{Int}, ::Vector{Int}, ::Vector{Int}, ::Vector{Pair{Int, Int}}, ::Vector{Int})
validate_probabilities(::InfluenceDiagram, ::Dict{Int, Array{Float64}})
validate_consequences(::InfluenceDiagram, ::Dict{Int, Array{Float64}})
```

```@docs
variables
DecisionModel(::InfluenceDiagram, ::Probabilities; ::Bool)
probability_sum_cut
number_of_paths_cut
```

```@docs
transform_affine_positive
expected_value
conditional_value_at_risk
```

```@docs
DecisionStrategy(::DecisionModel)
```

## analysis.jl
```@docs
ActivePaths
utility_distribution
state_probabilities
```

## printing.jl
```@docs
print_decision_strategy
print_state_probabilities
```

## random.jl
```@docs
random_influence_diagram
random_probabilities
random_consequences
```
