# Influence Diagram
## Introduction
Based on [^1], sections 3.

The paper [^2] explains details about influence diagrams.


## Definition
![](figures/influence-diagram.svg)

We define the **influence diagram** as a directed, acyclic graph

$$G=(C,D,V,I,S).$$

The nodes $N=C∪D∪V$ consists of **chance nodes** $C,$ **decision nodes** $D,$ and **value nodes** $V$. We index the chance and decision nodes such that $C∪D=\{1,...,n\}$ and values nodes such that $V=\{n+1,...,n+|V|\}$ where $n=|C|+|D|.$

We define the **information set** $I$ of node $j∈N$ as

$$I_j⊆\{i∈C∪D∣i<j\}$$

The condition enforces that the graph is directed and acyclic, and there are no arcs from value nodes to other nodes. Practically, the information set is an edge list to reverse direction in the graph. Furthermore, if $I_j=∅$ node $j$ is called a **root** node.

We refer to $S$ as the **state space**. Each chance and decision node $j∈C∪D$ is associates with a finite number of **states** $S_j$ that we encode using integers $\{1,...,|S_j|\}$ from one to number of states $|S_j|.$


## Paths
Paths in influence diagrams represent realizations of states for chance and decision nodes. Formally, a **path** is a sequence of states

$$𝐬=(s_1, s_2, ...,s_n),$$

where each state $s_i∈S_i$ for all chance and decision nodes $i∈C∪D.$

We define a **subpath** of $𝐬$ is a subsequence

$$(𝐬_{i_1}, 𝐬_{i_2}, ..., 𝐬_{i_{k}}),$$

where $1≤i_1<i_2<...<i_k≤n$ and $k≤n.$

The **information path** of node $j∈N$ on path $𝐬$ is a subpath defined as

$$𝐬_{I(j)}=(𝐬_i ∣ i∈I(j)).$$

We define the set of **all paths** as a product set of all states

$$𝐒=∏_{j∈C∪D} S_j.$$

The set of **information paths** of node $j∈N$ is the product set of the states in its information set

$$𝐒_{I(j)}=∏_{i∈I(j)} S_i.$$

We denote elements of the sets using notation $s_j∈S_j$, $𝐬∈𝐒$, and $𝐬_{I(j)}∈𝐒_{I(j)}.$


## Probabilities
For each chance node $j∈C$, we denote the **probability** of state $s_j$ given information path $𝐬_{I(j)}$ as

$$ℙ(X_j=s_j∣X_{I(j)}=𝐬_{I(j)})=ℙ(s_j∣𝐬_{I(j)})∈[0, 1],$$

with

$$∑_{s_j∈S_j} ℙ(s_j∣𝐬_{I(j)}) = 1.$$

Implementation wise, we can think probabilities as functions of information paths concatenated with state $X_j : 𝐒_{I(j)};S_j → [0, 1]$ where $∑_{s_j∈S_j} X_j(𝐬_{I(j)};s_j)=1.$


## Decision Strategy
For each decision node $j∈D,$ a **local decision strategy** maps an information path $𝐬_{I(j)}$ to a state $s_j$

$$Z_j:𝐒_{I(j)}↦S_j.$$

**Decision strategy** $Z$ contains one local decision strategy for each decision node. Set of **all decision strategies** is denoted $ℤ.$

A decision stategy $Z∈ℤ$ is **compatible** with the path $𝐬∈𝐒$ if and only if $Z_j(𝐬_{I(j)})=s_j$ forall $Z_j∈Z$ and $j∈D.$

An **active path** is path $𝐬∈𝐒$ that is compatible with decision strategy $Z.$ We denote the set of **all active paths** using $𝐒^Z.$ Since each decision strategy $Z_j$ chooses only one state out of all of its states, the **number of active paths** is

$$|𝐒^Z|=|𝐒|/\prod_{j∈D}|S_j|=\prod_{j∈C}|S_j|.$$


## Path Probability
We define the **path probability (upper bound)** as

$$p(𝐬) = ∏_{j∈C} ℙ(𝐬_j∣𝐬_{I(j)}).$$

The path probability $ℙ(𝐬∣Z)$ equals $p(𝐬)$ if the path $𝐬$ is compatible with the decision strategy $Z$. Otherwise, the path cannot occur and the probability is zero.


## Consequences
For each value node $j∈V$, we define the **consequence** given information path $𝐬_{I(j)}$ as

$$Y_j:𝐒_{I(j)}↦ℂ,$$

where $ℂ$ is the set of real-valued consequences.


## Path Utility
The **utility function** is a function that maps consequences to real-valued utility

$$U:ℂ^{|V|}↦ℝ.$$

The **path utility** is defined as the utility function acting on the consequences of value nodes given their information paths

$$\mathcal{U}(𝐬) = U(\{Y_j(𝐬_{I(j)}) ∣ j∈V\}).$$

The **default path utility** is the sum of consequences

$$\mathcal{U}(𝐬) = ∑_{j∈V} Y_j(𝐬_{I(j)}).$$


## Path Distribution
A **path distribution** is a pair

$$(ℙ(𝐬∣Z), \mathcal{U}(𝐬))$$

that comprises of path probability function and path utility function over paths $𝐬∈𝐒$ conditional to the decision strategy $Z.$


## Attributes
In this section, we define common attributes for influence diagrams. [^2]

**Discrete** influence diagram refers to countable state space. Otherwise, the influence diagram is **continuous**. We can discretize continuous influence diagrams using bins.

Two nodes are **sequential** if there exists a directed path from one node to the other in the influence diagram. Otherwise, the nodes are **parallel**. Sequential nodes often model time dimension.

**Repeated subdiagram** refers to a recurring pattern within an influence diagram. Often, influence diagrams do no have a unique structure, but they consist of a repeated pattern due to the properties of the underlying problem.

**Limited-memory** influence diagram refers to an influence diagram where an upper bound limits the size of the information set for decision nodes.

**Isolated subdiagrams** refer to an influence diagram that consists of multiple diagrams that are not connected. Therefore, decisions on one isolated subdiagram do not affect decisions on other isolated subdiagrams.


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)

[^2]: Bielza, C., Gómez, M., & Shenoy, P. P. (2011). A review of representation issues and modeling challenges with influence diagrams. Omega, 39(3), 227–241. https://doi.org/10.1016/j.omega.2010.07.003
