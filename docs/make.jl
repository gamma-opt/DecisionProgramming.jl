using Documenter
using DecisionProgramming, DataStructures

makedocs(
    sitename = "DecisionProgramming.jl",
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"]
    ),
    modules = [DecisionProgramming],
    authors = "Jaan Tollander de Balsch",
    pages = [
        "Home" => "index.md",
        "Decision Programming" => Any[
            "decision-programming/influence-diagram.md",
            "decision-programming/paths.md",
            "decision-programming/decision-model.md",
            "decision-programming/analyzing-decision-strategies.md",
            "decision-programming/computational-complexity.md",
        ],
        "usage.md",
        "Examples" => Any[
            "examples/used-car-buyer.md",
            "examples/pig-breeding.md",
            "examples/n-monitoring.md",
            "examples/contingent-portfolio-programming.md",
            "examples/CHD_preventative_care.md",
        ],
        "api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/gamma-opt/DecisionProgramming.jl.git",
    push_preview = true
)
