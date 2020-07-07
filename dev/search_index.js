var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/#Model","page":"API","title":"Model","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"paths\nSpecs\nInfluenceDiagram\nInfluenceDiagram(::Vector{Int}, ::Vector{Int}, ::Vector{Int}, ::Vector{Pair{Int, Int}}, ::Vector{Int})\nParams\nParams(::InfluenceDiagram, ::Dict{Int, Array{Float64}}, ::Dict{Int, Array{Float64}})\nDecisionModel\nDecisionModel(::Specs, ::InfluenceDiagram, ::Params)\nprobability_sum_cut\nnumber_of_paths_cut","category":"page"},{"location":"api/#DecisionProgramming.paths","page":"API","title":"DecisionProgramming.paths","text":"Iterate over paths.\n\n\n\n\n\nIterate over paths with fixed states.\n\n\n\n\n\n","category":"function"},{"location":"api/#DecisionProgramming.Specs","page":"API","title":"DecisionProgramming.Specs","text":"Specification for different model scenarios. For example, we can specify toggling on and off constraints and objectives.\n\nArguments\n\nprobability_sum_cut::Bool: Toggle probability sum cuts on and off.\nnum_paths::Int: If larger than zero, enables the number of paths cuts using the supplied value.\n\n\n\n\n\n","category":"type"},{"location":"api/#DecisionProgramming.InfluenceDiagram","page":"API","title":"DecisionProgramming.InfluenceDiagram","text":"Influence diagram.\n\n\n\n\n\n","category":"type"},{"location":"api/#DecisionProgramming.InfluenceDiagram-Tuple{Array{Int64,1},Array{Int64,1},Array{Int64,1},Array{Pair{Int64,Int64},1},Array{Int64,1}}","page":"API","title":"DecisionProgramming.InfluenceDiagram","text":"Construct and validate an influence diagram.\n\nArguments\n\nC::Vector{Int}: Change nodes.\nD::Vector{Int}: Decision nodes.\nV::Vector{Int}: Value nodes.\nA::Vector{Pair{Int, Int}}: Arcs between nodes.\nS_j::Vector{Int}: Number of states.\n\n\n\n\n\n","category":"method"},{"location":"api/#DecisionProgramming.Params","page":"API","title":"DecisionProgramming.Params","text":"Decision model parameters.\n\n\n\n\n\n","category":"type"},{"location":"api/#DecisionProgramming.Params-Tuple{InfluenceDiagram,Dict{Int64,Array{Float64,N} where N},Dict{Int64,Array{Float64,N} where N}}","page":"API","title":"DecisionProgramming.Params","text":"Construct and validate decision model parameters.\n\nArguments\n\ndiagram::InfluenceDiagram: The influence diagram associated with the probabilities and consequences.\nX::Dict{Int, Array{Float64}}: Probabilities\nY::Dict{Int, Array{Float64}}: Consequences\n\n\n\n\n\n","category":"method"},{"location":"api/#DecisionProgramming.DecisionModel","page":"API","title":"DecisionProgramming.DecisionModel","text":"Defines the DecisionModel type.\n\n\n\n\n\n","category":"type"},{"location":"api/#DecisionProgramming.DecisionModel-Tuple{Specs,InfluenceDiagram,Params}","page":"API","title":"DecisionProgramming.DecisionModel","text":"Construct a DecisionModel from specification, influence diagram and parameters.\n\nArguments\n\nspecs::Specs\ndiagram::InfluenceDiagram\nparams::Params\n\n\n\n\n\n","category":"method"},{"location":"api/#DecisionProgramming.probability_sum_cut","page":"API","title":"DecisionProgramming.probability_sum_cut","text":"Probability sum lazy cut.\n\n\n\n\n\n","category":"function"},{"location":"api/#DecisionProgramming.number_of_paths_cut","page":"API","title":"DecisionProgramming.number_of_paths_cut","text":"Number of paths lazy cut.\n\n\n\n\n\n","category":"function"},{"location":"api/#Analysis","page":"API","title":"Analysis","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"state_probabilities","category":"page"},{"location":"api/#DecisionProgramming.state_probabilities","page":"API","title":"DecisionProgramming.state_probabilities","text":"State probabilities.\n\n\n\n\n\nState probabilities.\n\n\n\n\n\n","category":"function"},{"location":"analysis/#Analysis","page":"Analysis","title":"Analysis","text":"","category":"section"},{"location":"analysis/#Active-Paths","page":"Analysis","title":"Active Paths","text":"","category":"section"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"An active path is path sS with positive path probability π(s)0 Then, we have the set of all active paths","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"S^+=sSπ(s)0","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"We denote the number of active paths as S^+","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"TODO: upper and lower bound of number of active paths, only one decision per information path, assume 2 state per decision (and chance) node,","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"S^+Sprod_iDS_i=prod_iCS_i","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"The ratio of active paths to all paths is","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"r=S^+S","category":"page"},{"location":"analysis/#State-Probabilities","page":"Analysis","title":"State Probabilities","text":"","category":"section"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"We denote paths with fixed states where ϵ denotes an empty state using a recursive definition.","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"beginaligned\nS_ϵ = S \nS_ϵs_i^ = sS_ϵ  s_i=s_i^ \nS_ϵs_i^s_j^ = sS_ϵs_i^  s_j=s_j^quad ji\nendaligned","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"The probability of all paths sums to one","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"ℙ(ϵ) = sum_sS_ϵ π(s) = 1","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"State probabilities for each node iCD and state s_iS_i denote how likely the state occurs given all path probabilities","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"ℙ(s_iϵ) = sum_sS_ϵs_i π(s)  ℙ(ϵ) = sum_sS_ϵs_i π(s)","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"An active state is a state with positive state probability ℙ(s_ic)0 given conditions c","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"We can generalize the state probabilities as conditional probabilities using a recursive definition. Generalized state probabilities allow us to explore how fixing active states affect the probabilities of other states. First, we choose an active state s_i and fix its value. Fixing an inactive state would make all state probabilities zero. Then, we can compute the conditional state probabilities as follows.","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"ℙ(s_jϵs_i) = sum_sS_ϵs_is_j π(s)  ℙ(s_iϵ)","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"We can then repeat this process by choosing an active state from the new conditional state probabilities s_k that is different from previously chosen states kj","category":"page"},{"location":"analysis/","page":"Analysis","title":"Analysis","text":"A robust recommendation is a decision state s_i where iD and subpath c such the state probability is one ℙ(s_ic)=1","category":"page"},{"location":"analysis/#Cumulative-Distribution-Function","page":"Analysis","title":"Cumulative Distribution Function","text":"","category":"section"},{"location":"model/#Decision-Model","page":"Decision Model","title":"Decision Model","text":"","category":"section"},{"location":"model/#Introduction","page":"Decision Model","title":"Introduction","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The model is based on [1], sections 3 and 5. We highly recommend to read them for motivation, details, and proofs of the formulation explained here. The paper [2] explains details about influence diagrams.","category":"page"},{"location":"model/#Influence-Diagram","page":"Decision Model","title":"Influence Diagram","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"(Image: )","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We define the influence diagram as a directed, acyclic graph such that part of its nodes have a finite number of states associated with them","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"G=(NAS_j)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The set of nodes N=CDV consists of chance nodes C decision nodes D and value nodes V. We index the nodes such that CD=1n and V=n+1n+V where n=C+D The set of arcs consists of pairs of nodes such that","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"A(ij)1ijNiV","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The condition enforces that the graph is directed and acyclic, and there are no arcs from value nodes to other nodes.","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Each chance and decision node jCD is associates with a finite number of states S_j We use integers from one to number of states S_j to encode individual states","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"S_j=1S_j","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We define the information set of node jN to be its predecessor nodes","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"I(j)=i(ij)A","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Practically, the information set is an edge list to reverse direction in the graph.","category":"page"},{"location":"model/#Paths","page":"Decision Model","title":"Paths","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Paths in influence diagrams represent realizations of states for chance and decision nodes. Formally, a path is a sequence of states","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"s=(s_1 s_2 s_n)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"where each state s_iS_i for all chance and decision nodes iCD","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We define a subpath of s is a subsequence","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"(s_i_1 s_i_2  s_i_k)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"where 1i_1i_2i_kn and kn","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The information path of node jN on path s is a subpath defined as","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"s_I(j)=(s_i  iI(j))","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Concatenation of two paths s and s^ is denoted ss^","category":"page"},{"location":"model/#Sets","page":"Decision Model","title":"Sets","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We define the set of all paths as a product set of all states","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"S=_jCD S_j","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The set of information paths of node jN is the product set of the states in its information set","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"S_I(j)=_iI(j) S_i","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We denote elements of the sets using notation s_jS_j, sS, and s_I(j)S_I(j)","category":"page"},{"location":"model/#Probabilities","page":"Decision Model","title":"Probabilities","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"For each chance node jC, we denote the probability of state s_j given information path s_I(j) as","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"ℙ(X_j=s_jX_I(j)=s_I(j))0 1","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We define the upper bound of the path probability s as","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"p(s) = _jC ℙ(X_j=s_jX_I(j)=s_I(j))","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We use it as a constraint in the model formulation.","category":"page"},{"location":"model/#Decisions","page":"Decision Model","title":"Decisions","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"For each decision node jD a local decision strategy maps an information path s_I(j) to a state s_j","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Z_jS_I(j)S_j","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Decision strategy Z contains one local decision strategy for each decision node. Set of all decision strategies is denoted ℤ","category":"page"},{"location":"model/#Consequences","page":"Decision Model","title":"Consequences","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"For each value node jV, we define the consequence given information path s_I(j) as","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Y_jS_I(j)ℂ","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"where ℂ is the set of consequences. In the code, the consequences are implicit, and we map information paths directly to the utility values.","category":"page"},{"location":"model/#Utilities","page":"Decision Model","title":"Utilities","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The utility function maps consequences to real-valued utilities","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Uℂℝ","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The utility of a path is defined as the sum of utilities for consequences of value nodes jV with information paths I(j)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"mathcalU(s) = _jV U(Y_j(s_I(j)))","category":"page"},{"location":"model/#Model-Formulation","page":"Decision Model","title":"Model Formulation","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The mixed-integer linear program maximizes the expected utility (1) over all decision strategies as follows.","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"undersetZℤtextmaximizequad\n_sS π(s) mathcalU(s) tag1","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Subject to","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"z(s_js_I(j))  01quad jD s_jS_j s_I(j)S_I(j) tag2","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_s_jS_j z(s_js_I(j))=1quad jD s_I(j)S_I(j) tag3","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"0π(s)p(s)quad sS tag4","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"π(s)  z(s_js_I(j))quad jD sS tag5","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"π(s)  p(s) + _jD z(s_js_I(j)) - Dquad sS tag6","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Decision variables z are binary variables (2) that model different decision strategies. The condition (3) limits decisions s_j to one per information path s_I(j) Decision strategy Z_j(s_I(j))=s_j is equivalent to z(s_js_I(j))=1","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We denote the probability distribution of paths using π The path probability π(s) is between zero and the upper bound of the path probability (4). The path probability is zero on paths where at least one decision variable is zero (5) and equal to the upper bound on paths if all decision variables on the path are one (6).","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We can omit the constraint (6) from the model if we use a positive utility function U^+ which is an affine transformation of utility function U As an example, we can normalize and add one to the original utility function.","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"U^+(c) = fracU(c) - min_cℂU(c)max_cℂU(c) - min_cℂU(c) + 1","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"There are also alternative objectives and ways to model risk. We discuss extensions to the model on the Extensions page.","category":"page"},{"location":"model/#Lazy-Cuts","page":"Decision Model","title":"Lazy Cuts","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Probability sum cut","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_sSπ(s)=1","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Number of active paths cut","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_sSπ(s)p(s)=n_s","category":"page"},{"location":"model/#Complexity","page":"Decision Model","title":"Complexity","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Number of paths","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"S=_iCD S_i = _iC S_i  _iD S_i","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Probability stages","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_iC (S_I(i)S_i)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Number of probability stages","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_iCS_I(i) S_i","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Decision stages","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_iD (S_I(i)S_i)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Number of decision stages","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_iDS_I(i) S_i","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Utility stages","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_iV S_I(i)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Number of utility stages","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_vVS_I(v)","category":"page"},{"location":"model/#References","page":"Decision Model","title":"References","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"[1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from http://arxiv.org/abs/1910.09196","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"[2]: Bielza, C., Gómez, M., & Shenoy, P. P. (2011). A review of representation issues and modeling challenges with influence diagrams. Omega, 39(3), 227–241. https://doi.org/10.1016/j.omega.2010.07.003","category":"page"},{"location":"examples/n-monitoring/#N-Monitoring-Problem","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"","category":"section"},{"location":"examples/n-monitoring/#Description","page":"N-Monitoring Problem","title":"Description","text":"","category":"section"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"The N-monitoring problem is described in [1], sections 4.1 and 6.1.","category":"page"},{"location":"examples/n-monitoring/#Formulation","page":"N-Monitoring Problem","title":"Formulation","text":"","category":"section"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"(Image: )","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"The 2-monitoring problem.","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"(Image: )","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"The incluence diagram of generalized N-monitoring problem where N1 and indices k=1N The nodes are associated with states as follows. Load state L=high low denotes the load, report states R_k=high low report the load state to the action states A_k=yes no which decide whether to fortificate failure state F=failure success Finally, the utility at target T depends on the whether F fails and the fortification costs.","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"We draw the magnitude and cost of fortification c_kU(01) from a uniform distribution. Fortification is defined","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"f(A_k) =\nbegincases\nc_k  A_k=yes \n0  A_k=no\nendcases","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"The probability that the load is high. We draw xU(01) from uniform distribution.","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"ℙ(L=high)=x","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"The probabilities of the report states correspond to the load state. We draw the values xU(01) and yU(01) from uniform distribution.","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"ℙ(R_k=highL=high)=maxxx-1","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"ℙ(R_k=lowL=low)=maxyy-1","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"The probabilities of failure which are decresead by fortifications. We draw the values zU(01) and wU(01) from uniform distribution.","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"ℙ(F=failureA_NA_1L=high)=fraczexp(_k=1N f(A_k))","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"ℙ(F=failureA_NA_1L=low)=fracwexp(_k=1N f(A_k))","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"Utility from consequences at target T from failure state F","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"U(Y(F)) =\nbegincases\n0  F = failure \n100  F = success\nendcases","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"Utility from consequences at target T from action states A_k","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"U(Y(A_k))=\nbegincases\n-c_k  A_k=yes \n0  A_k=no\nendcases","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"Total utility at target T","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"U(Y(FA_NA_1))=U(Y(F))+_k=1N U(Y(A_k))","category":"page"},{"location":"examples/n-monitoring/#References","page":"N-Monitoring Problem","title":"References","text":"","category":"section"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"[1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from http://arxiv.org/abs/1910.09196","category":"page"},{"location":"#DecisionProgramming.jl","page":"Home","title":"DecisionProgramming.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DecisionProgramming.jl","category":"page"},{"location":"examples/pig-breeding/#Pig-Breeding-Problem","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"","category":"section"},{"location":"examples/pig-breeding/#Description","page":"Pig Breeding Problem","title":"Description","text":"","category":"section"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The pig breeding problem as described in [1].","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"A pig breeder is growing pigs for a period of four months and subsequently selling them. During this period the pig may or may not develop a certain disease. If the pig has the disease at the time it must be sold, the pig must be sold for slaughtering, and its expected market price is then 300 DKK (Danish kroner). If it is disease free, its expected market price as a breeding animal is 1000 DKKOnce a month, a veterinary doctor sees the pig and makes a test for presence of the disease. If the pig is ill, the test will indicate this with probability 0.80, and if the pig is healthy, the test will indicate this with probability 0.90. At each monthly visit, the doctor may or may not treat the pig for the disease by injecting a certain drug. The cost of an injection is 100 DKK.A pig has the disease in the first month with probability 0.10. A healthy pig develops the disease in the subsequent month with probability 0.20 without injection, whereas a healthy and treated pig develops the disease with probability 0.10, so the injection has some preventive effect. An untreated pig that is unhealthy will remain so in the subsequent month with probability 0.90, whereas the similar probability is 0.50 for an unhealthy pig that is treated. Thus spontaneous cure is possible, but treatment is beneficial on average.","category":"page"},{"location":"examples/pig-breeding/#Formulation","page":"Pig Breeding Problem","title":"Formulation","text":"","category":"section"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"(Image: )","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The original 4-month formulation.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"(Image: )","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The influence diagram for the the generalized N-month pig breeding. The nodes are associated with the following states. Health states h_k=illhealthy represents the health of the pig at month k=1N. Test states t_k=positivenegative represents the result from testing the pig at month k=1N-1. Treat state d_k=treat pass represents the decision to treat the pig with an injection at month k=1N-1. The dashed arcs represent the no-forgetting principle and we can toggle them on and off in the formulation.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The probabilities that test indicates pig's health correctly at month k=1N-1.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(t_k = positive  h_k = ill) = 08","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(t_k = negative  h_k = healthy) = 09","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The probability that pig is ill in the first month.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(h_1 = ill)=01","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The probability that the pig is ill in the subsequent months k=2N given the treatment decision in and state of health in the previous month.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(h_k = ill  d_k-1 = pass h_k-1 = healthy)=02","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(h_k = ill  d_k-1 = treat h_k-1 = healthy)=01","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(h_k = ill  d_k-1 = pass h_k-1 = ill)=09","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(h_k = ill  d_k-1 = treat h_k-1 = ill)=05","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The cost of treatment decision for the pig at month k=1N-1","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"U(Y(d_k))=begincases\n-100  d_k = treat \n0  d_k = pass\nendcases","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The price of given the pig health at month N","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"U(Y(h_N))=begincases\n300  h_N = ill \n1000  h_N = healthy\nendcases","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"Total utility","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"U(Y(h_Nd_N-1d_1))=U(Y(h_n))+_k=1N U(Y(d_k))","category":"page"},{"location":"examples/pig-breeding/#References","page":"Pig Breeding Problem","title":"References","text":"","category":"section"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"[1]: Lauritzen, S. L., & Nilsson, D. (2001). Representing and solving decision problems with limited information. Management Science, 47(9), 1235–1251. https://doi.org/10.1287/mnsc.47.9.1235.9779","category":"page"},{"location":"extensions/#Extensions","page":"Extensions","title":"Extensions","text":"","category":"section"}]
}
