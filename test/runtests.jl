using Test

@testset "model.jl" begin
    include("influence_diagram.jl")
    include("random.jl")
    include("decision_model.jl")
end
