using Logging
using JuMP, Gurobi
using DecisionProgramming
using CSV, DataFrames, PrettyTables



# Setting subproblem specific parameters
const chosen_risk_level = "12%"


# Reading tests' technical performance data (dummy data in this case)
data = CSV.read("examples/CHD_preventative_care_data.csv", DataFrame)


# Bayes posterior risk probabilities calculation function
# prior = prior risk level for which the posterior risk distribution is calculated for,
# t = test done
# returns a 100x1 vector with the probabilities of getting CHD given the prior risk level and test result
# for no test done (i.e. t = 3) returns a zero vector
function update_risk_distribution(prior::Int64, t::Int64)
    if t == 1 # the test is TRS
        # P(TRS = result | sick) = P(TRS_if_sick = result) * P(sick) = P(TRS_if_sick = result) * P(prior_risk)
        numerators = data.TRS_if_sick .* data.risk_levels[prior]

        # P(TRS = result) = P(TRS_if_sick = result) * P(sick) + P(TRS_if_healthy = result) * P(healthy)
        denominators = data.TRS_if_sick .* data.risk_levels[prior]  + data.TRS_if_healthy .* (1 - data.risk_levels[prior])

        posterior_risks = numerators./denominators

        # if the denominator is zero, post_risk is NaN, changing those to 0
        for i = 1:101
            if isnan(posterior_risks[i])
                posterior_risks[i] = 0
            end
        end

        return posterior_risks


    elseif t == 2 #the test is GRS
        numerators = (data.GRS_if_sick .* data.risk_levels[prior])
        denominators = data.GRS_if_sick .* data.risk_levels[prior]  + data.GRS_if_healthy .* (1 .- data.risk_levels[prior])

        posterior_risks =  numerators./denominators

        # if the denominator is zero, post_risk is NaN, changing those to 0
        for i = 1:101
            if isnan(posterior_risks[i])
                posterior_risks[i] = 0
            end
        end

        return posterior_risks


    else # no test performed
        risks_unchanged = zeros(100,1)


        return risks_unchanged

    end
end

# State probabilites calculation function
# risk_p = the resulting array from update_risk_distribution
# t = test done
# h = CHD or no CHD
# returns the probability distribution in 101x1 vector for the states of the R node given the prior risk level (must be same as to function update_risk_distribution), test t and health h
function state_probabilities(risk_p::Array{Float64}, t::Int64, h::Int64, prior::Int64)

    #if no test is performed, then the probabilities of moving to states (other than the prior risk level) are 0 and to the prior risk element is 1
    if t == 3
        state_probabilites = zeros(101)
        state_probabilites[prior] = 1.0
        return state_probabilites
    end

    # return vector
    state_probabilites = zeros(101)

    # copying the probabilities of the scores for ease of readability
    if h == 1 && t == 1    # CHD and TRS
        p_scores = data.TRS_if_sick
    elseif t ==1    # no CHD and TRS
        p_scores = data.TRS_if_healthy
    elseif h == 1 && t == 2 # CHD and GRS
        p_scores = data.GRS_if_sick
    else # no CHD and GRS
        p_scores = data.GRS_if_healthy
    end

    for i = 1:101 #iterating through all risk levels 0%, 1%, ..., 99%, 100% in data.risk_levels
        for j = 1:101 #iterates through all risk estimates in risk_p
            #finding all risk estimates risk_p[j] within risk level i
            # risk_level[i] <= risk_p < risk_level[i]
            if i < 101 && data.risk_levels[i] <= risk_p[j] && risk_p[j] < data.risk_levels[i+1]
                state_probabilites[i] += p_scores[j]
            elseif i == 101 && data.risk_levels[i] <= risk_p[j] #special case: the highest risk level[101] = 100%
                state_probabilites[i] += p_scores[j]
            end
        end
    end

    return state_probabilites
end


@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

const H_states = ["CHD", "no CHD"]
const T_states = ["TRS", "GRS", "no test"]
const TD_states = ["treatment", "no treatment"]
const R_states = [string(x) * "%" for x in [0:1:100;]]


add_node!(diagram, ChanceNode("R0", [], R_states))
add_node!(diagram, ChanceNode("R1", ["R0", "H", "T1"], R_states))
add_node!(diagram, ChanceNode("R2", ["R1", "H", "T2"], R_states))
add_node!(diagram, ChanceNode("H", ["R0"], H_states))

add_node!(diagram, DecisionNode("T1", ["R0"], T_states))
add_node!(diagram, DecisionNode("T2", ["R1"], T_states))
add_node!(diagram, DecisionNode("TD", ["R2"], TD_states))

add_node!(diagram, ValueNode("TC", ["T1", "T2"]))
add_node!(diagram, ValueNode("HB", ["H", "TD"]))


generate_arcs!(diagram)

X_R0 = ProbabilityMatrix(diagram, "R0")
set_probability!(X_R0, [chosen_risk_level], 1)
add_probabilities!(diagram, "R0", X_R0)


X_H = ProbabilityMatrix(diagram, "H")
set_probability!(X_H, [:, "CHD"], data.risk_levels)
set_probability!(X_H, [:, "no CHD"], 1 .- data.risk_levels)
add_probabilities!(diagram, "H", X_H)

X_R = ProbabilityMatrix(diagram, "R1")
for s_R0 = 1:101, s_H = 1:2, s_T1 = 1:3
    X_R[s_R0, s_H,  s_T1, :] =  state_probabilities(update_risk_distribution(s_R0, s_T1), s_T1, s_H, s_R0)
end
add_probabilities!(diagram, "R1", X_R)
add_probabilities!(diagram, "R2", X_R)


cost_TRS = -0.0034645
cost_GRS = -0.004
forbidden = 0     #the cost of forbidden test combinations is negligible
Y_TC = UtilityMatrix(diagram, "TC")
set_utility!(Y_TC, ["TRS", "TRS"], forbidden)
set_utility!(Y_TC, ["TRS", "GRS"], cost_TRS + cost_GRS)
set_utility!(Y_TC, ["TRS", "no test"], cost_TRS)
set_utility!(Y_TC, ["GRS", "TRS"], cost_TRS + cost_GRS)
set_utility!(Y_TC, ["GRS", "GRS"], forbidden)
set_utility!(Y_TC, ["GRS", "no test"], cost_GRS)
set_utility!(Y_TC, ["no test", "TRS"], cost_TRS)
set_utility!(Y_TC, ["no test", "GRS"], cost_GRS)
set_utility!(Y_TC, ["no test", "no test"], 0)
add_utilities!(diagram, "TC", Y_TC)

Y_HB = UtilityMatrix(diagram, "HB")
set_utility!(Y_HB, ["CHD", "treatment"], 6.89713671259061)
set_utility!(Y_HB, ["CHD", "no treatment"], 6.65436854256236 )
set_utility!(Y_HB, ["no CHD", "treatment"], 7.64528451705134)
set_utility!(Y_HB, ["no CHD", "no treatment"], 7.70088349200034)
add_utilities!(diagram, "HB", Y_HB)

generate_diagram!(diagram)


@info("Creating the decision model.")
model = Model()
z = DecisionVariables(model, diagram)

# Defining forbidden paths to include all those where a test is repeated twice
forbidden_tests = ForbiddenPath(diagram, ["T1","T2"], [("TRS", "TRS"),("GRS", "GRS"),("no test", "TRS"), ("no test", "GRS")])
fixed_R0 = FixedPath(diagram, Dict("R0" => chosen_risk_level))
scale_factor = 10000.0
x_s = PathCompatibilityVariables(model, diagram, z; fixed = fixed_R0, forbidden_paths = [forbidden_tests], probability_cut=false)

EV = expected_value(model, diagram, x_s, probability_scale_factor = scale_factor)
@objective(model, Max, EV)

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
    "MIPFocus" => 3,
    "MIPGap" => 1e-6,
)
set_optimizer(model, optimizer)
GC.enable(false)
optimize!(model)
GC.enable(true)


@info("Extracting results.")
Z = DecisionStrategy(z)
S_probabilities = StateProbabilities(diagram, Z)
U_distribution = UtilityDistribution(diagram, Z)

@info("Printing decision strategy using tailor made function:")
print_decision_strategy(diagram, Z, S_probabilities)

@info("Printing state probabilities:")
# Here we can see that the probability of having a CHD event is exactly that of the chosen risk level
print_state_probabilities(diagram, S_probabilities, ["R0", "R1", "R2"])

@info("Printing utility distribution.")
print_utility_distribution(U_distribution)

print_statistics(U_distribution)
