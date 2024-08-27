using DataFrames, PrettyTables
using StatsBase, StatsBase.Statistics
using DecisionProgramming
using Base64

"""
    print_decision_strategy(diagram::InfluenceDiagram, Z::DecisionStrategy, state_probabilities::StateProbabilities; show_incompatible_states::Bool = false)

Print decision strategy.

# Arguments
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `Z::DecisionStrategy`: Decision strategy structure with optimal decision strategy.
- `state_probabilities::StateProbabilities`: State probabilities structure corresponding to optimal decision strategy.
- `show_incompatible_states::Bool`: Choice to print rows also for incompatible states.

# Examples
```julia
print_decision_strategy(diagram, Z, S_probabilities)
```
"""
function print_decision_strategy(diagram::InfluenceDiagram, Z::DecisionStrategy, state_probabilities::StateProbabilities; show_incompatible_states::Bool = false)
    probs = state_probabilities.probs

    for (d, I_d, Z_d) in zip(Z.D, Z.I_d, Z.Z_d)
        s_I = vec(collect(paths(get_values(diagram.S)[I_d])))
        s_d = [Z_d(s) for s in s_I]

        if !isempty(I_d)
            informations_states = [join([String(get_values(diagram.States)[i][s_i]) for (i, s_i) in zip(I_d, s)], ", ") for s in s_I]
            decision_probs = [ceil(prod(probs[i][s1] for (i, s1) in zip(I_d, s))) for s in s_I]
            decisions = collect(p == 0 ? "--" : get_values(diagram.States)[d][s] for (s, p) in zip(s_d, decision_probs))
            df = DataFrame(informations_states = informations_states, decisions = decisions)
            if !show_incompatible_states
                 filter!(row -> row.decisions != "--", df)
            end
            pretty_table(df, header = ["State(s) of $(join([diagram.Names[i] for i in I_d], ", "))", "Decision in $(diagram.Names[d])"], alignment=:l)
        else
            df = DataFrame(decisions = get_values(diagram.States)[d][s_d])
            pretty_table(df, header = ["Decision in $(diagram.Names[d])"], alignment=:l)
        end
    end
end

"""
    print_utility_distribution(U_distribution::UtilityDistribution; util_fmt="%f", prob_fmt="%f")

Print utility distribution.

# Examples
```julia
U_distribution = UtilityDistribution(diagram, Z)
print_utility_distribution(U_distribution)
```
"""
function print_utility_distribution(U_distribution::UtilityDistribution; util_fmt="%f", prob_fmt="%f")
    df = DataFrame(Utility = U_distribution.u, Probability = U_distribution.p)
    formatters = (
        ft_printf(util_fmt, [1]),
        ft_printf(prob_fmt, [2]))
    pretty_table(df; formatters = formatters)
end

"""
    print_state_probabilities(diagram::InfluenceDiagram, state_probabilities::StateProbabilities, nodes::Vector{Name}; prob_fmt="%f")

Print state probabilities with fixed states.

# Examples
```julia
S_probabilities = StateProbabilities(diagram, Z)
print_state_probabilities(S_probabilities, ["R"])
print_state_probabilities(S_probabilities, ["A"])
```
"""
function print_state_probabilities(diagram::InfluenceDiagram, state_probabilities::StateProbabilities, nodes::Vector{Name}; prob_fmt="%f")
    node_indices = [findfirst(j -> j==node, diagram.Names) for node in nodes]
    states_list = get_values(diagram.States)[node_indices]
    state_sets = unique(states_list)
    n = length(states_list)

    probs = state_probabilities.probs
    fixed = state_probabilities.fixed

    prob(p, state) = if 1≤state≤length(p) p[state] else NaN end
    fix_state(i) = if i∈keys(fixed) string(get_values(diagram.States)[i][fixed[i]]) else "" end


    for state_set in state_sets
        node_indices2 = filter(i -> get_values(diagram.States)[i] == state_set, node_indices)
        state_names = get_values(diagram.States)[node_indices2[1]]
        states = 1:length(state_names)
        df = DataFrame()
        df[!, :Node] = diagram.Names[node_indices2]
        for state in states
            df[!, Symbol("$(state_names[state])")] = [prob(probs[i], state) for i in node_indices2]
        end
        df[!, Symbol("Fixed state")] = [fix_state(i) for i in node_indices2]
        pretty_table(df; formatters = ft_printf(prob_fmt, (first(states)+1):(last(states)+1)))
    end
end

"""
    print_statistics(U_distribution::UtilityDistribution; fmt = "%f")

Print statistics about utility distribution.
"""
function print_statistics(U_distribution::UtilityDistribution; fmt = "%f")
    u = U_distribution.u
    w = ProbabilityWeights(U_distribution.p)
    names = ["Mean", "Std", "Skewness", "Kurtosis"]
    statistics = [mean(u, w), std(u, w, corrected=false), skewness(u, w), kurtosis(u, w)]
    df = DataFrame(Name = names, Statistics = statistics)
    pretty_table(df, formatters = ft_printf(fmt, [2]))
end

"""
    print_risk_measures(U_distribution::UtilityDistribution, αs::Vector{Float64}; fmt = "%f")

Print risk measures.
"""
function print_risk_measures(U_distribution::UtilityDistribution, αs::Vector{Float64}; fmt = "%f")
    VaR = [value_at_risk(U_distribution, α) for α in αs]
    CVaR = [conditional_value_at_risk(U_distribution, α) for α in αs]
    df = DataFrame(α = αs, VaR = VaR, CVaR = CVaR)
    pretty_table(df, formatters = ft_printf(fmt))
end



function print_node_io(io::IO, node::AbstractNode)
    node_type = "Unknown"
    node_states = "n/a"
    
    if node isa ChanceNode
        node_type = "ChanceNode"
        node_states = node.states
    elseif node isa DecisionNode
        node_type = "DecisionNode"
        node_states = node.states
    elseif node isa ValueNode
        node_type = "ValueNode"
    end

    node_info_set = isempty(node.I_j) ? "empty" : node.I_j

    println(io, "An influence diagram node")
    println(io, "Name: ", node.name)
    println(io, "Index: ", node.index)
    println(io, "Type: ", node_type)
    println(io, "Information Set: ", node_info_set)
    
    if node_type != "ValueNode"
        println(io, "States: ", node_states)
    end
end



"""
    print_node(node_name::String, diagram::InfluenceDiagram; print_tables::Bool=true)

Print node information. print_tables determines whether probability and utility tables for the node are printed.

# Examples
```julia
print_node("H2", diagram)
"""
function print_node(node_name::String, diagram::InfluenceDiagram; print_tables::Bool=true)
    if haskey(diagram.Nodes, node_name)
        node = diagram.Nodes[node_name]
    else
        throw(ErrorException("Node '$node_name' does not exist."))
    end

    node_type = "Unknown"
    node_states = "n/a"
    
    if node isa ChanceNode
        node_type = "ChanceNode"
        node_states = node.states
    elseif node isa DecisionNode
        node_type = "DecisionNode"
        node_states = node.states
    elseif node isa ValueNode
        node_type = "ValueNode"
    end

    node_info_set = isempty(node.I_j) ? "empty" : node.I_j

    println("An influence diagram node")
    println("Name: ", node.name)
    println("Index: ", node.index)
    println("Type: ", node_type)
    println("Information Set: ", node_info_set)
    
    if node_type != "ValueNode"
        println("States: ", node_states)
    end
    println("")

    function print_table(table_type::String)
        table_I_j = []
        for I_j_node_name in node.I_j
            push!(table_I_j, diagram.Nodes[I_j_node_name].states)
        end

        if table_type == "probabilities"
            column_names_I_j = node.I_j
            column_name_node = node_name
            table_node = diagram.Nodes[node_name].states
            tables = vcat(table_I_j, [table_node])
            columns = vcat(column_names_I_j, column_name_node)
            columns = map(x -> x * " state", columns)
        elseif table_type == "utilities"
            columns = node.I_j
            tables = table_I_j
            columns = map(x -> x * " state", columns)
        else
            throw(ArgumentError("Invalid table type. Expected 'probabilities' or 'utilities'"))
        end

        if table_type == "probabilities"
            matrix = diagram.X[node.name].data
        elseif table_type == "utilities"
            matrix = diagram.Y[node.name].data
        else
            throw(ArgumentError("Invalid table type. Expected 'probabilities' or 'utilities'"))
        end
        permuted = permutedims(matrix, reverse(1:ndims(matrix)))
        converted_data = vec(permuted)

        function all_combinations(tables)
            function inner(idx, current)
                if idx > length(tables)
                    return [current]
                end
                results = []
                for value in tables[idx]
                    results = vcat(results, inner(idx + 1, vcat(current, value)))
                end
                return results
            end
            return inner(1, [])
        end
        
        combinations = all_combinations(tables)
        
        df = DataFrame()
        for (i, c) in enumerate(columns)
            df[!, Symbol(c)] = [comb[i] for comb in combinations]
        end
        
        df[!, :Value] = converted_data

        pretty_table(df, title = uppercasefirst(table_type) * ":")
        println("")
    end

    if print_tables == true
        if haskey(diagram.X, node.name)
            print_table("probabilities")
        end
        if haskey(diagram.Y, node.name)
            print_table("utilities")
        end
    end
end

"""
    print_diagram(diagram::InfluenceDiagram; print_tables::Bool=true)

Print diagram. print_tables determines whether probability and utility values for the nodes in the diagram are printed.

# Examples
```julia
print_diagram(diagram)
"""
function print_diagram(diagram::InfluenceDiagram; print_tables::Bool=true)
    println("An influence diagram")
    println("")
    println("Node names:")
    println(diagram.Names)
    println("")

    println("Nodes:")
    println("")
    for node in keys(diagram.Nodes)
        print_node(node, diagram; print_tables)
    end
end

#--- MERMAID GRAPH PRINTING FUNCTIONS ---

function nodes(diagram::InfluenceDiagram, class::String, edge::String; S::Union{Nothing, Vector{State}}=nothing)
    lines = []
    I_j = I_j_indices(diagram, diagram.Nodes)
    if class == "chance"
        N = indices(diagram.C)
        left = "(("
        right = "))"
    elseif class == "decision"
        N = indices(diagram.D)
        left = "["
        right = "]"
    elseif class == "value"
        N = indices(diagram.V)
        left = "{"
        right = "}"
    else
        throw(DomainError("Unknown class $(class)"))
    end
    for n in N
        if S === nothing
            push!(lines, "$(n)$(left)Node: $(diagram.Names[n])$(right)")
        else
            push!(lines, "$(n)$(left)Node: $(diagram.Names[n]) <br> $(S[n]) states$(right)")
        end
        if !isempty(I_j[n])
            I_j_joined = join(I_j[n], " & ")
            push!(lines, "$(I_j_joined) $(edge) $(n)")
        end
    end
    js = join([n for n in N], ",")
    push!(lines, "class $(js) $(class)")
    return join(lines, "\n")
end

function graph(diagram::InfluenceDiagram)
    return """
    graph LR
    %% Chance nodes
    $(nodes(diagram, "chance", "-->"; S=get_values(diagram.S)))

    %% Decision nodes
    $(nodes(diagram, "decision", "-->"; S=get_values(diagram.S)))

    %% Value nodes
    $(nodes(diagram, "value", "-.->"))

    %% Styles
    classDef chance fill:#F5F5F5 ,stroke:#666666,stroke-width:2px;
    classDef decision fill:#D5E8D4 ,stroke:#82B366,stroke-width:2px;
    classDef value fill:#FFE6CC ,stroke:#D79B00,stroke-width:2px;
    """
end

"""
    mermaid(diagram::InfluenceDiagram, filename::String="mermaid_graph.png")

Print mermaid graph. NOTE TO USER: Accesses the url mermaid.ink, which is used for graphing.
"""
function mermaid(diagram::InfluenceDiagram, filename::String="mermaid_graph.png")
    graph_output = graph(diagram)
    graphbytes = codeunits(graph_output)
    base64_bytes = base64encode(graphbytes)
    img_url = "https://mermaid.ink/img/" * base64_bytes

    download(img_url, filename)
end