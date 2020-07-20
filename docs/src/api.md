# API Reference
## model.jl
```@docs
paths
path_probability
```

```@docs
InfluenceDiagram
InfluenceDiagram(::Vector{Int}, ::Vector{Int}, ::Vector{Int}, ::Vector{Pair{Int, Int}}, ::Vector{Int})
Params
Params(::InfluenceDiagram, ::Dict{Int, Array{Float64}}, ::Dict{Int, Array{Float64}})
```

```@docs
DecisionModel
DecisionModel(::InfluenceDiagram, ::Params)
probability_sum_cut
number_of_paths_cut
```

```@docs
transform_affine_positive
expected_value
value_at_risk
```

```@docs
DecisionStrategy
DecisionStrategy(::DecisionModel)
```

## analysis.jl
```@docs
is_compatible
active_paths
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
random_params
```
