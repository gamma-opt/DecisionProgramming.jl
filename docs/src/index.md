# DecisionProgramming.jl
DecisionProgramming.jl is a Julia package for solving *multi-stage decision problems under uncertainty*, modeled using influence diagrams, and leveraging the power of mixed-integer linear programming. Solving multi-stage decision problems under uncertainty consists of the following three steps.

In the first step, we model the decision problem using an influence diagram with associated probabilities, consequences, and path utility function.

In the second step, we create a decision model with an objective for the influence diagram. We solve the model to obtain an optimal decision strategy. We can create and solve multiple models with different objectives for the same influence diagram to receive various optimal decision strategies.

In the third step, we analyze the resulting decision strategies for the influence diagram. In particular, we are interested in utility distribution and its associated statistics and risk measures.

DecisionProgramming.jl provides the necessary functionality for expressing and solving decision problems but does not explain how to design influence diagrams. The rest of this documentation will describe the mathematical and programmatic details, touch on the computational challenges, and provide concrete examples of solving decision problems.

DecisionProgramming.jl is developed in the [Systems Analysis Laboratory](https://sal.aalto.fi/en/) at Aalto University by *Ahti Salo*,  *Fabricio Oliveira*, *Juho Andelmin*, *Olli Herala*, and *Jaan Tollander de Balsch*.
