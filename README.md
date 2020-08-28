# DecisionProgramming.jl
[![Docs Image](https://img.shields.io/badge/docs-latest-blue.svg)](https://gamma-opt.github.io/DecisionProgramming.jl/dev/)
[![Build Status](https://travis-ci.org/gamma-opt/DecisionProgramming.jl.svg?branch=master)](https://travis-ci.org/gamma-opt/DecisionProgramming.jl)
[![Coverage Status](https://coveralls.io/repos/github/gamma-opt/DecisionProgramming.jl/badge.svg?branch=master)](https://coveralls.io/github/gamma-opt/DecisionProgramming.jl?branch=master)


## Description
`DecisionProgramming.jl` is a [Julia](https://julialang.org/) package for solving multi-stage decision problems under uncertainty, modeled using influence diagrams, and formulated using mixed-integer linear programming. We designed the package as [JuMP](https://jump.dev/) extension.


## Syntax
We can create an influence diagram as follows:

```julia
# TODO
```

Using the influence diagram, we create decision models as follow:

```julia
# TODO
```

See the documentation for more detailed examples.


## Installation
Currently `DecisionProgramming.jl` is unregistered. You can add it using directly from GitHub using the command:

```julia-repl
pkg> add https://github.com/gamma-opt/DecisionProgramming.jl.git
```

To solve the decision model, users have to install a solver capable of solving mixed-integer linear programs (MILP). JuMP documentation contains a list of available solvers.


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
