using Documenter
using DecisionProgramming

makedocs(
    sitename = "DecisionProgramming.jl",
    format = Documenter.HTML(),
    modules = [DecisionProgramming],
    authors = "Jaan Tollander de Balsch",
    pages = [
        "Home" => "index.md",
        "model.md",
        "Examples" => Any[
            "examples/pig-breeding.md"
        ],
        "api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/gamma-opt/DecisionProgramming.jl.git"
)
