"""
    randomStrategy(diagram::InfluenceDiagram)

Generates a random decision strategy for the problem. Returns the strategy as well as 
the expected utility of the strategy and the paths that are compatible with the strategy.

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
    #I_j_indexed = Vector{Node}.([indices_of(diagram, nodes) for nodes in get_values(diagram.I_j)])
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    #D_keys_indexed = Node.([index_of(diagram, node) for node in get_keys(diagram.D)])
    D_indices = indices(diagram.D)
    
    # Loop through all decision nodes and set local decision strategies
    for j in D_indices
        I_j = I_j_indices_result[j]

        # Generate a matrix of correct dimensions to represent the strategy
        dims = get_values(diagram.S)[[I_j; j]]
        data = zeros(Int, Tuple(dims))
        n_states = size(data)[end]

        # For each information state, choose a random decision state 
        for s_Ij in paths(get_values(diagram.S)[I_j])
            data[s_Ij..., rand(1:n_states)] = 1
        end
        push!(Z_d, LocalDecisionStrategy(j,data))
    end

    # Construct a decision strategy and obtain the compatible paths
    Z = DecisionStrategy(D_indices, I_j_indices_result[D_indices], Z_d)
    S_active = CompatiblePaths(diagram, Z)

    # Calculate the expected utility corresponding to the strategy
    EU = sum(diagram.P(s)*diagram.U(s) for s in S_active)

    return EU, Z, collect(S_active)
end


function findBestStrategy(diagram, j, s_Ij, S_active, model, EU)
    # Check that the model is either a minimization or maximization problem
    if objective_sense(model) == MOI.MIN_SENSE
        bestsofar = (0, Inf, [])
    elseif objective_sense(model) == MOI.MAX_SENSE
        bestsofar = (0, -Inf, [])
    else
        throw("The given model is not a maximization or minimization problem.")
    end

    # Loop through all decision states and save the one corresponding to the best expected value
    for s_j in 1:num_states(diagram,diagram.Names[j])
        # Get the expected value corresponding to a strategy where the information state s_Ij maps to s_j 
        # and the strategy stays otherwise the same. Note that the strategy is represented by the active paths.
        EU_new, S_active_new = get_value(diagram, S_active, j, s_j, s_Ij, EU)

        # Update the best value so far
        if objective_sense(model) == MOI.MIN_SENSE
            if EU_new <= bestsofar[2]
                bestsofar = (s_j, EU_new, S_active_new)
            end
        else #objective_sense(model) == MOI.MAX_SENSE
            if EU_new >= bestsofar[2]
                bestsofar = (s_j, EU_new, S_active_new)
            end
        end
    end
    return bestsofar
end


function get_value(diagram, S_active, j, s_j, s_Ij, EU)
    #I_j_indexed = Vector{Node}.([indices_of(diagram, nodes) for nodes in get_values(diagram.I_j)])
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    I_j = I_j_indices_result[j] # Information set of node j
    # Loop through all compatible paths and update the ones that correspond to the given information state s_Ij
    # and update the expected utility whenever a path is updated
    S_active_new = copy(S_active)
    for (k, s) in enumerate(S_active)
        if s[I_j] == s_Ij
            EU -= diagram.P(s)*diagram.U(s)
            s_new = [s_j for s_j in s]
            s_new[j] = s_j
            s_new = Tuple(s_new)
            S_active_new[k] = s_new
            EU += diagram.P(s_new)*diagram.U(s_new)
        end
    end
    return EU, S_active_new
end

function set_MIP_start(diagram, Z, S_active, z_z; x_s)
    for (k,j) in enumerate(Z.D)
        for s_Ij in paths(get_values(diagram.S)[Z.I_d[k]])
                set_start_value(z_z[k][s_Ij..., Z.Z_d[k](s_Ij)], 1)
        end
    end

    if x_s != nothing
        for s in S_active
            set_start_value(x_s[s], 1)
        end
    end
end

"""
    singlePolicyUpdate(diagram::InfluenceDiagram, model::Model, z::OrderedDict{Name, DecisionVariable}, x_s::PathCompatibilityVariables)

Finds a feasible solution using single policy update and sets the model start values to that solution.
Returns a vector of tuples consisting of the value of each improved solution starting from a random policy, 
time (in milliseconds) since the function call and the decision strategy that gave the improved value.
The purpose of all this output is to allow us to examine how fast the method finds good solutions.

# Arguments
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `model::Model`: The decision model, modelled in JuMP
- `z::OrderedDict{Name, DecisionVariable}`: The decision variables
- `x_s::Union{PathCompatibilityVariables, Nothing}`: The path compatibility variables if used.

!!! warning
    This function does not exclude forbidden paths: the strategies explored by this function might be forbidden if the diagram has forbidden state combinations.

# Examples
```julia
solutionhistory = singlePolicyUpdate(diagram, model, z, x_s)
```
"""
function singlePolicyUpdate(diagram::InfluenceDiagram, model::Model, z::OrderedDict{Name, DecisionVariable}; x_s::Union{PathCompatibilityVariables, Nothing}=nothing)
    t1 = time_ns() # Start time

    # Initialize empty values
    solutionhistory = [] 
    lastchange = nothing

    # Get an initial (random) solution
    EU, strategy, S_active = randomStrategy(diagram)
    push!(solutionhistory, (EU, (time_ns()-t1)/1E6, deepcopy(strategy)))

    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    D_indices = indices(diagram.D)

    z_z = [decision_node.z for decision_node in get_values(z)]

    # In principle, this always converges, but we set a maximum number of iterations anyway to avoid very long solution times
    for iter in 1:20
        # Loop through all nodes
        for (idx, j) in enumerate(D_indices)
            I_j = I_j_indices_result[j]
            # Loop through all information states
            for s_Ij in paths(get_values(diagram.S)[I_j])
                # Check if any improvement has happened since the last time this node and information state was visited
                # If not, the algorithm terminates with a locally optimal solution
                if iter >= 2
                    if lastchange == (j, s_Ij)
                        set_MIP_start(diagram, solutionhistory[end][3], S_active, z_z; x_s)
                        return solutionhistory
                    end
                end

                # Find the best decision alternative s_j for information state s_Ij
                s_j, bestval, S_active = findBestStrategy(diagram, j, s_Ij, S_active, model, EU)
                
                # If the strategy improved, save the new strategy and its expected utility
                if (objective_sense(model) == MOI.MIN_SENSE && bestval < EU-1E-9) || (objective_sense(model) == MOI.MAX_SENSE && bestval > EU+1E-9)
                    lastchange = (j, s_Ij)
                    localstrategy = strategy.Z_d[idx].data
                    localstrategy[s_Ij..., :] .= 0
                    localstrategy[s_Ij..., s_j] = 1
                    strategy.Z_d[idx] = LocalDecisionStrategy(j, localstrategy)
                    EU = bestval
                    push!(solutionhistory, (EU, (time_ns()-t1)/1E6, deepcopy(strategy)))
                end
            end
        end
    end

    # Set the best found solution as the MIP start to the model
    set_MIP_start(diagram, solutionhistory[end][3], S_active, z_z; x_s)
    return solutionhistory
end