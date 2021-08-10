# DecisionProgramming.jl
[![Docs Image](https://img.shields.io/badge/docs-latest-blue.svg)](https://gamma-opt.github.io/DecisionProgramming.jl/dev/)
![Runtests](https://github.com/gamma-opt/DecisionProgramming.jl/workflows/Runtests/badge.svg)


## Description
`DecisionProgramming.jl` is a [Julia](https://julialang.org/) package for solving multi-stage decision problems under uncertainty, modeled using influence diagrams. Internally, it relies on mathematical optimization. Decision models can be embedded within other optimization models. We designed the package as [JuMP](https://jump.dev/) extension.


## Syntax
![](examples/figures/simple-id.svg)

We can create an influence diagram as follows:

```julia
using DecisionProgramming
S = States([2, 2, 2, 2])
C = [ChanceNode(2, [1]), ChanceNode(3, [1])]
D = [DecisionNode(1, Node[]), DecisionNode(4, [2, 3])]
V = [ValueNode(5, [4])]
X = [Probabilities(2, [0.4 0.6; 0.6 0.4]), Probabilities(3, [0.7 0.3; 0.3 0.7])]
Y = [Consequences(5, [1.5, 1.7])]
validate_influence_diagram(S, C, D, V)
sort!.((C, D, V, X, Y), by = x -> x.j)
P = DefaultPathProbability(C, X)
U = DefaultPathUtility(V, Y)
```

Using the influence diagram, we create the decision model as follow:

```julia
using JuMP
model = Model()
z = DecisionVariables(model, S, D)
x_s = PathCompatibilityVariables(model, z, S, P)
EV = expected_value(model, x_s, U, P)
@objective(model, Max, EV)
```

Finally, we can optimize the model using MILP solver.

```julia
using Gurobi
optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
    "IntFeasTol"      => 1e-9,
)
set_optimizer(model, optimizer)
optimize!(model)
```

Finally, we extract the decision strategy from the decision variables.

```julia
Z = DecisionStrategy(z)
```

See the [documentation](https://gamma-opt.github.io/DecisionProgramming.jl/dev/) for more detailed examples.


## Installation
Currently `DecisionProgramming.jl` is unregistered. You can add it using directly from GitHub using the command:

```julia-repl
pkg> add https://github.com/gamma-opt/DecisionProgramming.jl
```

To run examples and develop and solve decision models, you have to install JuMP and a solver capable of solving mixed-integer linear programs (MILP). JuMP documentation contains a list of available solvers.

```julia-repl
pkg> add JuMP
```

We recommend using the [Gurobi](https://www.gurobi.com/) solver, which is an efficient commercial solver. Academics use Gurobi for free with an academic license. You also need to install the Julia Gurobi package.

```julia-repl
pkg> add Gurobi
```

Now you are ready to use decision programming.


## Development
Using the package manager, add `DecisionProgramming.jl` package for development using the command:

```julia-repl
pkg> develop https://github.com/gamma-opt/DecisionProgramming.jl
```

If you have already cloned `DecisionProgramming` from GitHub, you can use the command:

```julia-repl
pkg> develop .
```

Inside `DecisionProgramming` directory, run tests using the commands:

```julia-repl
pkg> activate .
(DecisionProgramming) pkg> test
```

You can find more instruction on how to install packages for development at Julia's [Pkg documentation.](https://docs.julialang.org/en/v1/stdlib/Pkg/)
