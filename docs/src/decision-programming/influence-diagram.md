# Influence Diagram
## Introduction
Based on [^1], sections 3.

The paper [^2] explains details about influence diagrams.


## Definition
![](figures/influence-diagram.svg)

We define the **influence diagram** as a directed, acyclic graph such that part of its nodes have a finite number of states associated with them

$$G=(N,A,S_j).$$

The set of nodes $N=C∪D∪V$ consists of **chance nodes** $C,$ **decision nodes** $D,$ and **value nodes** $V$. We index the nodes such that $C∪D=\{1,...,n\}$ and $V=\{n+1,...,n+|V|\}$ where $n=|C|+|D|.$ The set of **arcs** consists of pairs of nodes such that

$$A⊆\{(i,j)∣1≤i<j≤|N|,i∉V\}.$$

The condition enforces that the graph is directed and acyclic, and there are no arcs from value nodes to other nodes.

Each chance and decision node $j∈C∪D$ is associates with a finite number of **states** $S_j.$ We use integers from one to number of states $|S_j|$ to encode individual states

$$S_j=\{1,...,|S_j|\}.$$

We define the **information set** of node $j∈N$ to be its predecessor nodes

$$I(j)=\{i∣(i,j)∈A\}.$$

Practically, the information set is an edge list to reverse direction in the graph.


## Paths
Paths in influence diagrams represent realizations of states for chance and decision nodes. Formally, a **path** is a sequence of states

$$s=(s_1, s_2, ...,s_n),$$

where each state $s_i∈S_i$ for all chance and decision nodes $i∈C∪D.$

We define a **subpath** of $s$ is a subsequence

$$(s_{i_1}, s_{i_2}, ..., s_{i_{k}}),$$

where $1≤i_1<i_2<...<i_k≤n$ and $k≤n.$

The **information path** of node $j∈N$ on path $s$ is a subpath defined as

$$s_{I(j)}=(s_i ∣ i∈I(j)).$$

**Concatenation of two paths** $s$ and $s^′$ is denoted $s;s^′.$

We define the set of **all paths** as a product set of all states

$$S=∏_{j∈C∪D} S_j.$$

The set of **information paths** of node $j∈N$ is the product set of the states in its information set

$$S_{I(j)}=∏_{i∈I(j)} S_i.$$

We denote elements of the sets using notation $s_j∈S_j$, $s∈S$, and $s_{I(j)}∈S_{I(j)}.$


## Probabilities
For each chance node $j∈C$, we denote the **probability** of state $s_j$ given information path $s_{I(j)}$ as

$$ℙ(X_j=s_j∣X_{I(j)}=s_{I(j)})∈[0, 1],$$

with

$$∑_{s_j∈S_j} ℙ(X_j=s_j∣X_{I(j)}=s_{I(j)}) = 1.$$

We define the **upper bound of path probability** $s$ as

$$p(s) = ∏_{j∈C} ℙ(X_j=s_j∣X_{I(j)}=s_{I(j)}).$$


## Decisions
For each decision node $j∈D,$ a **local decision strategy** maps an information path $s_{I(j)}$ to a state $s_j$

$$Z_j:S_{I(j)}↦S_j.$$

**Decision strategy** $Z$ contains one local decision strategy for each decision node. Set of **all decision strategies** is denoted $ℤ.$

A decision stategy $Z∈ℤ$ is **compatible** with the path $s∈S$ if and only if $Z_j(s_{I(j)})=s_j$ forall $Z_j∈Z$ and $j∈D.$

The path probability $ℙ(s∣Z)$ equals $p(s)$ if the path $s$ is compatible with the decision strategy $Z$. Otherwise, the path cannot occur and the probability is zero.

An **active path** is path $s∈S$ that is compatible with decision strategy $Z.$ We denote the set of **all active paths** using $S^Z.$ Since each decision strategy $Z_j$ chooses only one state out of all of its states, the **number of active paths** is

$$|S^Z|=|S|/\prod_{j∈D}|S_j|=\prod_{j∈C}|S_j|.$$


## Consequences
For each value node $j∈V$, we define the **consequence** given information path $s_{I(j)}$ as

$$Y_j:S_{I(j)}↦ℂ,$$

where $ℂ$ is the set of consequences. In the code, the consequences are implicit, and we map information paths directly to the utility values.

The **utility function** maps consequences to real-valued utilities

$$U:ℂ↦ℝ.$$

The **path utility** is defined as the sum of utilities for consequences of value nodes $j∈V$ with information paths $I(j)$

$$\mathcal{U}(s) = ∑_{j∈V} U(Y_j(s_{I(j)})).$$


## Path Distribution
A **path distribution** is a pair

$$(ℙ(s∣Z), \mathcal{U}(s))$$

that comprises of path probability function and path utility function over paths $s∈S$ conditional to the decision strategy $Z.$


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)

[^2]: Bielza, C., Gómez, M., & Shenoy, P. P. (2011). A review of representation issues and modeling challenges with influence diagrams. Omega, 39(3), 227–241. https://doi.org/10.1016/j.omega.2010.07.003
