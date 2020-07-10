# API Reference
## Model
```@docs
paths
Specs
InfluenceDiagram
InfluenceDiagram(::Vector{Int}, ::Vector{Int}, ::Vector{Int}, ::Vector{Pair{Int, Int}}, ::Vector{Int})
Params
Params(::InfluenceDiagram, ::Dict{Int, Array{Float64}}, ::Dict{Int, Array{Float64}})
DecisionModel
DecisionModel(::Specs, ::InfluenceDiagram, ::Params)
probability_sum_cut
number_of_paths_cut
```

## Analysis
```@docs
active_paths
is_compatible
state_probabilities
utility_distribution
print_results
print_decision_strategy
print_state_probabilities
```

## Random
```@docs
random_influence_diagram
random_params
```
