using Test

@testset "model.jl" begin
    include("influence_diagram.jl")
    include("decision_model.jl")
    include("heuristics.jl")
end
