using Pkg
Pkg.activate((@__DIR__)*"/..")
using Logging, Random
using DelimitedFiles
using JuMP, Gurobi
using DecisionProgramming
using DelimitedFiles

idx = parse(Int64, ARGS[1])
Random.seed!(idx)
N_max = 7
result_arr = zeros(N_max-1,6)
for N in N_max:-1:2
    
    p_ill = rand()*0.2                  # Initial probability of pig being ill

    spec = 0.5 + rand()*0.5             # Specificity of the test
    sens = 0.5 + rand()*0.5             # Sensitivity of the test

    p_vals = sort(rand(4))              # Health state update probabilities
    p_hpi = p_vals[1]                   # Probability of ill untreated pig to become healthy, assumed to be the smallest of the random probabilities
    p_hti = p_vals[2]                   # Probability of ill treated pig to become healthy
    p_hph = p_vals[3]                   # Probability of healthy untreated pig to remain healthy
    p_hth = p_vals[4]                   # Probability of healthy treated pig to remain healthy, assumed to be the largest of the random probabilities

    c_treat = -rand()*100               # Cost of treatment
    u_sell = sort(rand(2).*1000).+100   # Utility of selling a pig, ill or healthy (healthy is more valuable)

    @info("Creating the influence diagram.")
    diagram = InfluenceDiagram()

    add_node!(diagram, ChanceNode("H1", [], ["ill", "healthy"]))
    for i in 1:N-1
        # Testing result
        add_node!(diagram, ChanceNode("T$i", ["H$i"], ["positive", "negative"]))
        # Decision to treat
        add_node!(diagram, DecisionNode("D$i", ["T$i"], ["treat", "pass"]))
        # Cost of treatment
        add_node!(diagram, ValueNode("C$i", ["D$i"]))
        # Health of next period
        add_node!(diagram, ChanceNode("H$(i+1)", ["H$(i)", "D$(i)"], ["ill", "healthy"]))
    end
    add_node!(diagram, ValueNode("MP", ["H$N"]))

    generate_arcs!(diagram)

    # Add probabilities for node H1
    add_probabilities!(diagram, "H1", [p_ill, 1-p_ill])

    # Declare proability matrix for health nodes H_2, ... H_N-1, which have identical information sets and states
    X_H = ProbabilityMatrix(diagram, "H2")
    X_H["healthy", "pass", :] = [1-p_hph, p_hph]
    X_H["healthy", "treat", :] = [1-p_hth, p_hth]
    X_H["ill", "pass", :] = [1-p_hpi, p_hpi]
    X_H["ill", "treat", :] = [1-p_hti, p_hti]

    # Declare proability matrix for test result nodes T_1...T_N
    X_T = ProbabilityMatrix(diagram, "T1")
    X_T["ill", "positive"] = 1-spec
    X_T["ill", "negative"] = spec
    X_T["healthy", "negative"] = 1-sens
    X_T["healthy", "positive"] = sens

    for i in 1:N-1
        add_probabilities!(diagram, "T$i", X_T)
        add_probabilities!(diagram, "H$(i+1)", X_H)
    end

    for i in 1:N-1
        add_utilities!(diagram, "C$i", [c_treat, 0.0])
    end

    add_utilities!(diagram, "MP", u_sell)

    generate_diagram!(diagram, positive_path_utility = true)


    @info("Creating the decision model.")
    model = Model()
    z = DecisionVariables(model, diagram)
    x_s = PathCompatibilityVariables(model, diagram, z)
    EV = expected_value(model, diagram, x_s)
    @objective(model, Max, EV)

    @info("Starting the optimization process.")
    optimizer = optimizer_with_attributes(
        () -> Gurobi.Optimizer(Gurobi.Env()),
        "IntFeasTol"      => 1e-9,
    )
    set_optimizer(model, optimizer)
    undo_relax = relax_integrality(model)
    optimize!(model)
    linrel = objective_value(model)
    
    undo_relax()
    
    optimize!(model)
    t1 = solve_time(model)
    objval = objective_value(model)


    model = Model()
    z = DecisionVariables(model, diagram)
    x_s = PathCompatibilityVariables(model, diagram, z)
    EV = expected_value(model, diagram, x_s)
    @objective(model, Max, EV)

    @info("Starting the optimization process.")
    optimizer = optimizer_with_attributes(
        () -> Gurobi.Optimizer(Gurobi.Env()),
        "IntFeasTol"      => 1e-9,
    )
    set_optimizer(model, optimizer)
    spu = singlePolicyUpdate(diagram, model, z, x_s)
    
    if N == 7
        writedlm((@__DIR__)*"/results/pigfarm_spu_6stages_"*string(idx)*".csv", spu, ',')
    end
    
    optimize!(model)
    t2 = solve_time(model)
    t_spu = spu[end][2]/1000
    spu_val = spu[end][1]
    result_arr[N-1,:] = [t1 t2 t_spu objval-diagram.translation spu_val linrel-diagram.translation]
end

writedlm((@__DIR__)*"/results/pigfarm_"*string(idx)*".csv", result_arr, ',')