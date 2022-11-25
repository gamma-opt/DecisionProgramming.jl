using Dates

"""
    randomStrategy(diagram::InfluenceDiagram)

Generates a random decision strategy for the problem and also returns the expected utility of the strategy
and the paths that are compatible with the strategy.

# Arguments
- `diagram::InfluenceDiagram`: Influence diagram structure.

!!! warning
    This function does not exclude forbidden paths: the strategy returned by this function might be forbidden if the diagram has forbidden state combinations.

# Examples
```julia
objval, Z, S_active = randomStrategy(diagram)
```
"""
function randomStrategy(diagram::InfluenceDiagram)

    # Initialize empty vector for local decision strategies
    # Z_d = Vector{LocalDecisionStrategy}[] # Doesn't work for some reason...
    Z_d = []

    # Loop through all decision nodes to obtain their local decision strategies
    for j in diagram.D
        I_j = diagram.I_j[j]

        # Generate a matrix of correct dimensions to represent the strategy
        dims = diagram.S[[I_j; j]]
        data = zeros(Int, Tuple(dims))
        n_states = size(data)[end]

        # For each information state, choose a random decision state 
        for s_Ij in paths(diagram.S[I_j])
            data[s_Ij..., rand(1:n_states)] = 1
        end
        push!(Z_d, LocalDecisionStrategy(j,data))
    end

    # Construct a decision strategy and obtain the compatible paths
    Z = DecisionStrategy(diagram.D, diagram.I_j[diagram.D], Z_d)
    S_active = CompatiblePaths(diagram, Z)

    # Calculate the expected utility corresponding to the strategy
    EU = 0
    for s in S_active
        EU += diagram.P(s)*diagram.U(s)
    end
    return EU, Z, collect(S_active)
end


function findBestStrategy(diagram, j, s_Ij, S_active, model)
    # Check that the model is either a minimization or maximization problem
    if objective_sense(model) == MOI.MIN_SENSE
        bestsofar = (Inf, 0, [])
    elseif objective_sense(model) == MOI.MAX_SENSE
        bestsofar = (-Inf, 0, [])
    else
        throw("The given model is not a maximization or minimization problem.")
    end

    # Loop through all decision states and save the one corresponding to the best expected value
    for s_j in 1:num_states(diagram,diagram.Names[j])
        # Get the expected value corresponding to a strategy where the information state s_Ij maps to s_j 
        # and the strategy stays otherwise the same. Note that the strategy is represented by the active paths.
        EU, S_active = get_value(diagram, S_active, j, s_j, s_Ij)

        # Update the best value so far
        if objective_sense(model) == MOI.MIN_SENSE
            if EU <= bestsofar[1]
                bestsofar = (EU, s_j, S_active)
            end
        else #objective_sense(model) == MOI.MAX_SENSE
            if EU >= bestsofar[1]
                bestsofar = (EU, s_j, S_active)
            end
        end
    end
    return bestsofar[2], bestsofar[1], bestsofar[3]
end


function get_value(diagram, S_active, j, s_j, s_Ij)
    I_j = diagram.I_j[j] # Information set of node j

    # Loop through all compatible paths and update the ones that correspond to the given information state s_Ij
    for (k, s) in enumerate(S_active)
        if s[I_j] == s_Ij
            s_new = [s_j for s_j in s]
            s_new[j] = s_j
            s_new = Tuple(s_new)
            S_active[k] = s_new
        end
    end

    #Calculate the expected utility corresponding to this strategy
    EU = 0
    for s in S_active
        EU += diagram.P(s)*diagram.U(s)
    end

function set_MIP_start(diagram, Z, S_active, z, x_s)
    for (k,j) in enumerate(Z.D)
        for s_Ij in paths(diagram.S[Z.I_d[k]])
                # println(z.z[j][s_Ij..., s_j])
                set_start_value(z.z[k][s_Ij..., Z.Z_d[k](s_Ij)], 1)
                # println(start_value(z.z[j][s_Ij..., s_j]))
        end
end

    for s in S_active
        # println(x_s[s])
        set_start_value(x_s[s], 1)
        # println(start_value(x_s[s]))
    end
end

"""
    singlePolicyUpdate(diagram::InfluenceDiagram, model::Model)

Finds a feasible solution using single policy update and sets the model start values to that solution.
Returns a vector of tuples consisting of the value of each improved solution starting from a random policy, 
time (in milliseconds) since the function call and the decision strategy that gave the improved value.
The purpose of all this output is to allow us to examine how fast the method finds good solutions.

# Arguments
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `model::Model`: The decision model, modelled in JuMP
- `z::DecisionVariables`: The decision variables
- `x_s::PathCompatibilityVariables`: The path compatibility variables

!!! warning
    This function does not exclude forbidden paths: the strategies explored by this function might be forbidden if the diagram has forbidden state combinations.

# Examples
```julia
solutionhistory = singlePolicyUpdate(diagram, model)
```
"""
function singlePolicyUpdate(diagram::InfluenceDiagram, model::Model, z::DecisionVariables, x_s::PathCompatibilityVariables)
    t1 = now() # Start time

    # Initialize empty values
    solutionhistory = [] 
    lastchange = nothing

    # Get an initial (random) solution
    EU, strategy, S_active = randomStrategy(diagram)
    push!(solutionhistory, (EU, Dates.value(now()-t1), deepcopy(strategy)))

    # In principle, this always converges, but we set a maximum number of iterations anyway to avoid very long solution times
    for iter in 1:20
        # Loop through all nodes
        for (idx, j) in enumerate(diagram.D)
            # println("Node $(diagram.Names[j]), iteration $iter")
            I_j = diagram.I_j[j]
            # Loop through all information states
            for s_Ij in paths(diagram.S[I_j])
                # Check if any improvement has happened since the last time this node and information state was visited
                # If not, the algorithm terminates with a locally optimal solution
                if iter >= 2
                    if lastchange == (j, s_Ij)
                        set_MIP_start(diagram, solutionhistory[end][3], S_active, z, x_s)
                        return solutionhistory
                    end
                end

                # Find the best decision alternative s_j for information state s_Ij
                s_j, bestval, S_active = findBestStrategy(diagram, j, s_Ij, S_active, model)
                
                # If the strategy improved, save the new strategy and its expected utility
                if bestval > EU
                    lastchange = (j, s_Ij)
                    localstrategy = strategy.Z_d[idx].data
                    localstrategy[s_Ij..., :] .= 0
                    localstrategy[s_Ij..., s_j] = 1
                    strategy.Z_d[idx] = LocalDecisionStrategy(j, localstrategy)
                    EU = bestval
                    push!(solutionhistory, (EU, Dates.value(now()-t1), deepcopy(strategy)))
                end
            end
        end
    end
    
    # TODO: set start values in the model

    set_MIP_start(diagram, solutionhistory[end][3], S_active, z, x_s)

    return solutionhistory
end