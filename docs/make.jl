using Documenter
using DecisionProgramming

makedocs(
    sitename = "DecisionProgramming",
    format = Documenter.HTML(),
    modules = [DecisionProgramming],
    authors = "Jaan Tollander de Balsch",
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/gamma-opt/DecisionProgramming.jl.git"
)
