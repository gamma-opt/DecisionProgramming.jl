using Documenter
using DecisionProgramming

makedocs(
    sitename = "DecisionProgramming.jl",
    format = Documenter.HTML(
        # assets = ["assets/favicon.ico"]
    ),
    modules = [DecisionProgramming],
    authors = "Jaan Tollander de Balsch",
    pages = [
        "Home" => "index.md",
        "Decision Programming" => Any[
            "decision-programming/influence-diagram.md",
            "decision-programming/decision-model.md",
            "decision-programming/complexity.md",
        ],
        "analysis.md",
        "Examples" => Any[
            "examples/pig-breeding.md",
            "examples/n-monitoring.md",
            "examples/contingent-portfolio-programming.md",
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
