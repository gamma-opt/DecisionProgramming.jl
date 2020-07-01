var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"paths\nSpecs\nInfluenceDiagram\nInfluenceDiagram(::Vector{Int}, ::Vector{Int}, ::Vector{Int}, ::Vector{Pair{Int, Int}}, ::Vector{Int})\nParams\nParams(::InfluenceDiagram, ::Dict{Int, Array{Float64}}, ::Dict{Int, Array{Int}}, ::Vector{Float64})\nDecisionModel\nDecisionModel(::Specs, ::InfluenceDiagram, ::Params)","category":"page"},{"location":"api/#DecisionProgramming.paths","page":"API","title":"DecisionProgramming.paths","text":"Iterate over paths.\n\n\n\n\n\n","category":"function"},{"location":"api/#DecisionProgramming.Specs","page":"API","title":"DecisionProgramming.Specs","text":"Specification for different model scenarios. For example, we can specify toggling on and off constraints and objectives.\n\nArguments\n\nprobability_sum_cut::Bool: Toggle probability sum cuts on and off.\nnum_paths::Int: If larger than zero, enables the number of paths cuts using the supplied value.\n\n\n\n\n\n","category":"type"},{"location":"api/#DecisionProgramming.InfluenceDiagram","page":"API","title":"DecisionProgramming.InfluenceDiagram","text":"Influence diagram is a directed, acyclic graph.\n\nArguments\n\nC::Vector{Int}: Change nodes.\nD::Vector{Int}: Decision nodes.\nV::Vector{Int}: Value nodes.\nA::Vector{Pair{Int, Int}}: Arcs between nodes.\nS_j::Vector{Int}: Number of states.\nI_j::Vector{Vector{Int}}: Information set.\n\n\n\n\n\n","category":"type"},{"location":"api/#DecisionProgramming.InfluenceDiagram-Tuple{Array{Int64,1},Array{Int64,1},Array{Int64,1},Array{Pair{Int64,Int64},1},Array{Int64,1}}","page":"API","title":"DecisionProgramming.InfluenceDiagram","text":"Construct and validate an influence diagram.\n\n\n\n\n\n","category":"method"},{"location":"api/#DecisionProgramming.Params","page":"API","title":"DecisionProgramming.Params","text":"Model parameters.\n\nArguments\n\nX: Probabilities, X[j][sI(j);sj], ∀j∈C\nY: Consequences, Y[j][s_I(j)], ∀j∈V\nU: Utilities map consequences to real valued outcomes.\n\n\n\n\n\n","category":"type"},{"location":"api/#DecisionProgramming.Params-Tuple{InfluenceDiagram,Dict{Int64,Array{Float64,N} where N},Dict{Int64,Array{Int64,N} where N},Array{Float64,1}}","page":"API","title":"DecisionProgramming.Params","text":"Construct and validate model parameters.\n\n\n\n\n\n","category":"method"},{"location":"api/#DecisionProgramming.DecisionModel","page":"API","title":"DecisionProgramming.DecisionModel","text":"Defines the DecisionModel type.\n\n\n\n\n\n","category":"type"},{"location":"api/#DecisionProgramming.DecisionModel-Tuple{Specs,InfluenceDiagram,Params}","page":"API","title":"DecisionProgramming.DecisionModel","text":"Initializes the DecisionModel.\n\n\n\n\n\n","category":"method"},{"location":"model/#Decision-Model","page":"Decision Model","title":"Decision Model","text":"","category":"section"},{"location":"model/#Introduction","page":"Decision Model","title":"Introduction","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The model is based on [1], sections 3 and 5. We highly recommend to read them for motivation, details, and proofs of the formulation explained here. We explain how we have implemented the model in the source code.","category":"page"},{"location":"model/#Influence-Diagram","page":"Decision Model","title":"Influence Diagram","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"(Image: )","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Influence diagram is defined as a directed, acyclic graph such that some of its nodes have a finite number of states associated with them","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"G=(NAS_j)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The set of nodes N=CDV consists of chance nodes C decision nodes D and value nodes V. We index the nodes such that CD=1n and V=n+1n+V where n=C+D As a consequence, the value nodes are never the children of chance or decision nodes.","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The set of arcs consists of pairs of nodes such that","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"A(ij)1ijN","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The condition enforces that the graph is directed and acyclic.","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Each chance and decision node jCD is associates with a finite number of states S_j=1S_j We use integers from one to the size of the set of states to represent states. Hence, we use the sizes of the sets of states S_j to represent them.","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We define the information set of node jN to be its predecessor nodes","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"I(j)=i(ij)A","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Practically, the information set is an edge list to reverse direction in the graph.","category":"page"},{"location":"model/#Paths","page":"Decision Model","title":"Paths","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Paths in influence diagrams represent realizations of states for multiple nodes. Formally, a path is a sequence of states","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"s=(s_1 s_2 s_n)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"where each state s_iS_i for all chance and decision nodes iCD","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"A subpath of s is a subsequence","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"(s_i_1 s_i_2  s_i_k)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"where 1i_1i_2i_kn and kn","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The information path of node jN on path s is a subpath defined as","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"s_I(j)=(s_i  iI(j))","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Concatenation of two paths s and s^ is denoted ss^","category":"page"},{"location":"model/#Sets","page":"Decision Model","title":"Sets","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The set of all paths is the product set of all states","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"S=_jCD S_j","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The set of information paths of node jN is the product set of the states in its information set","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"S_I(j)=_iI(j) S_i","category":"page"},{"location":"model/#Probabilities","page":"Decision Model","title":"Probabilities","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"For each chance node jC, the probability of state s_j given information state s_I(j) is denoted as","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"ℙ(X_j=s_jX_I(j)=s_I(j))0 1","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The upper bound of the probability of a path s is defined as","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"p(s) = _jC ℙ(X_j=s_jX_I(j)=s_I(j))","category":"page"},{"location":"model/#Decisions","page":"Decision Model","title":"Decisions","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"For each decision node jD a local decision strategy maps an information path to a state","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Z_jS_I(j)S_j","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Decision strategy Z contains one local decision strategy for each decision node. Set of all decision strategies is denoted ℤ","category":"page"},{"location":"model/#Consequences","page":"Decision Model","title":"Consequences","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"For each value node jV, the consequence given information state S_I(j) is defined as","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Y_vS_I(j)ℂ","category":"page"},{"location":"model/#Utilities","page":"Decision Model","title":"Utilities","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The utility function maps consequence to real-valued utilities","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Uℂℝ","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Affine transformation to positive utilities","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"U^(c) = U(c) - min_cℂU(c)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The utility of a path","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"mathcalU(s) = _jV U^(Y_j(s_I(j)))","category":"page"},{"location":"model/#Model-Formulation","page":"Decision Model","title":"Model Formulation","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The probability distribution of paths depends on the decision strategy. We model this distribution as the variable π and denote the probability of path s as π(s)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Decision strategy Z_j(s_I(j))=s_j is equivalent to z(s_js_I(j))=1 and _s_jS_j z(s_js_I(j))=1 for all jD s_I(j)S_I(j)","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"The mixed-integer linear program maximizes the expected utility over all decision strategies as follows.","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"beginaligned\nundersetZℤtextmaximizequad\n _sS π(s) mathcalU(s) \ntextsubject toquad\n _s_jS_j z(s_js_I(j))=1quad jD s_I(j)S_I(j) \n 0π(s)p(s)quad sS \n π(s)  z(s_js_I(j))quad sS \n z(s_js_I(j))  01quad jD s_jS_j s_I(j)S_I(j)\nendaligned","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"We discuss an extension to the model on the Extension page.","category":"page"},{"location":"model/#Lazy-Cuts","page":"Decision Model","title":"Lazy Cuts","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Probability sum cut","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_sSπ(s)=1","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Number of pats cut","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_sSπ(s)p(s)=n_s","category":"page"},{"location":"model/#Results","page":"Decision Model","title":"Results","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Active paths sπ(s)0","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Active states s_iπ(s)0 for each node iCD, robust recommendations?","category":"page"},{"location":"model/#Sizes","page":"Decision Model","title":"Sizes","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"States and paths","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"_iC (S_I(i)S_i) probability stages\n_iD (S_I(i)S_i) decision stages\n_iV S_I(i) utility stages","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"Sizes","category":"page"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"S=_iCD S_i Number of paths\n_iCS_I(i) S_i Number of probability stages\n_iDS_I(i) S_i Number of decision stages\n_vVS_I(v) Number of utility stages","category":"page"},{"location":"model/#References","page":"Decision Model","title":"References","text":"","category":"section"},{"location":"model/","page":"Decision Model","title":"Decision Model","text":"[1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from http://arxiv.org/abs/1910.09196","category":"page"},{"location":"examples/n-monitoring/#N-Monitoring-Problem","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"","category":"section"},{"location":"examples/n-monitoring/#Description","page":"N-Monitoring Problem","title":"Description","text":"","category":"section"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"The N-monitoring problem is described in [1], sections 4.1, 6.1.","category":"page"},{"location":"examples/n-monitoring/#Formulation","page":"N-Monitoring Problem","title":"Formulation","text":"","category":"section"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"(Image: )","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"The 2-monitoring problem.","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"(Image: )","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"The generalized N-monitoring problem.","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"Diagram N1 k=1N","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"States","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"Load L, {low, high}\nActions A_k\nRisk of failure F, {failure, success}\nReports of load R_k, {low, high}\nTarget T","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"Utility","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"Failure 0\nSuccess 100","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"Probabilities","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"xU(01)","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"ℙ(L=high)=x","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"yU(01)","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"ℙ(R_k=highL=high)+ℙ(R_k=lowL=low)=maxyy-1","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"zwU(01)","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"c_kU(01)","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"ℙ(F=failureL=high A_1A_N)=fraczexp(_kA c_k)","category":"page"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"ℙ(F=failureL=low A_1A_N)=fracwexp(_kA c_k)","category":"page"},{"location":"examples/n-monitoring/#References","page":"N-Monitoring Problem","title":"References","text":"","category":"section"},{"location":"examples/n-monitoring/","page":"N-Monitoring Problem","title":"N-Monitoring Problem","text":"[1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from http://arxiv.org/abs/1910.09196","category":"page"},{"location":"#DecisionProgramming.jl","page":"Home","title":"DecisionProgramming.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DecisionProgramming.jl","category":"page"},{"location":"examples/pig-breeding/#Pig-Breeding-Problem","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"","category":"section"},{"location":"examples/pig-breeding/#Description","page":"Pig Breeding Problem","title":"Description","text":"","category":"section"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The pig breeding problem as described in [1].","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"A pig breeder is growing pigs for a period of four months and subsequently selling them. During this period the pig may or may not develop a certain disease. If the pig has the disease at the time it must be sold, the pig must be sold for slaughtering, and its expected market price is then 300 DKK (Danish kroner). If it is disease free, its expected market price as a breeding animal is 1000 DKKOnce a month, a veterinary doctor sees the pig and makes a test for presence of the disease. If the pig is ill, the test will indicate this with probability 0.80, and if the pig is healthy, the test will indicate this with probability 0.90. At each monthly visit, the doctor may or may not treat the pig for the disease by injecting a certain drug. The cost of an injection is 100 DKK.A pig has the disease in the first month with probability 0.10. A healthy pig develops the disease in the subsequent month with probability 0.20 without injection, whereas a healthy and treated pig develops the disease with probability 0.10, so the injection has some preventive effect. An untreated pig that is unhealthy will remain so in the subsequent month with probability 0.90, whereas the similar probability is 0.50 for an unhealthy pig that is treated. Thus spontaneous cure is possible, but treatment is beneficial on average.","category":"page"},{"location":"examples/pig-breeding/#Formulation","page":"Pig Breeding Problem","title":"Formulation","text":"","category":"section"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"(Image: )","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The original 4-month formulation.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"(Image: )","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"Generalized N-month formulation.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"The diagram has the following states","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"Change nodes and states","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"State h_k represents the health of the pig at month k=1N. Two possible states, ill and healthy.\nState t_k represents the result from testing the pig at month k=1N-1. Two possible states, positive, and negative.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"Decision nodes and states","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"State d_k represents the decision to treat the pig with injection at month k=1N-1. Two possible states, treat and pass.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"Value nodes and states","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"Consequences u_k represents the consequences from treating or not treating the pig at month k=1N-1.\nConsequence u_N represents the consequence from health of the pig at month N.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"Influence is represented by the arcs. Dashed arcs represent no forgetting principle.","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"Probabilities","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(t_k = positive  h_k = ill) = 08quad k=1N-1","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(t_k = negative  h_k = healthy) = 09quad k=1N-1","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(h_1 = ill)=01","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(h_k = ill  d_k-1 = pass h_k-1 = healthy)=02quad k=2N","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(h_k = ill  d_k-1 = treat h_k-1 = healthy)=01quad k=2N","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(h_k = ill  d_k-1 = pass h_k-1 = ill)=09quad k=2N","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"ℙ(h_k = ill  d_k-1 = treat h_k-1 = ill)=05quad k=2N","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"Utilities","category":"page"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"Cost of treating the pig U(Y(d_k=treat))=-100 at month k=1N-1\nCost of not treating the pig U(Y(d_k=pass))=0 at month k=1N-1\nPrice of unhealthy pig U(Y(h_N=ill))=300\nPrice of healthy pig U(Y(h_N=healthy))=1000","category":"page"},{"location":"examples/pig-breeding/#Results","page":"Pig Breeding Problem","title":"Results","text":"","category":"section"},{"location":"examples/pig-breeding/#References","page":"Pig Breeding Problem","title":"References","text":"","category":"section"},{"location":"examples/pig-breeding/","page":"Pig Breeding Problem","title":"Pig Breeding Problem","text":"[1]: Lauritzen, S. L., & Nilsson, D. (2001). Representing and solving decision problems with limited information. Management Science, 47(9), 1235–1251. https://doi.org/10.1287/mnsc.47.9.1235.9779","category":"page"}]
}
