# Decision Model
## Introduction
The model is based on [^1], sections 3 and 5. We highly recommend to read them for motivation, details, and proofs of the formulation explained here. We explain how we have implemented the model in the source code.


## Influence Diagram
![](figures/influence-diagram.svg)

**Influence diagram** is defined as a directed, acyclic graph such that some of its nodes have a finite number of states associated with them

$$G=(N,A,S_j).$$

The set of nodes $N=C∪D∪V$ consists of **chance nodes** $C,$ **decision nodes** $D,$ and **value nodes** $V$. We index the nodes such that $C∪D=\{1,...,n\}$ and $V=\{n+1,...,n+|V|\}$ where $n=|C|+|D|.$ As a consequence, the value nodes are never the children of chance or decision nodes.

The set of **arcs** consists of pairs of nodes such that

$$A⊆\{(i,j)∣1≤i<j≤|N|\}.$$

The condition enforces that the graph is directed and acyclic.

Each chance and decision node $j∈C∪D$ is associates with a finite number of **states** $S_j=\{1,...,|S_j|\}.$ We use integers from one to the size of the set of states to represent states. Hence, we use the sizes of the sets of states $|S_j|$ to represent them.

We define the **information set** of node $j∈N$ to be its predecessor nodes

$$I(j)=\{i∣(i,j)∈A\}.$$

Practically, the information set is an edge list to reverse direction in the graph.


## Paths
Paths in influence diagrams represent realizations of states for multiple nodes. Formally, a **path** is a sequence of states

$$s=(s_1, s_2, ...,s_n),$$

where each state $s_i∈S_i$ for all chance and decision nodes $i∈C∪D.$

A **subpath** of $s$ is a subsequence

$$(s_{i_1}, s_{i_2}, ..., s_{i_{k}}),$$

where $1≤i_1<i_2<...<i_k≤n$ and $k≤n.$

The **information path** of node $j∈N$ on path $s$ is a subpath defined as

$$s_{I(j)}=(s_i ∣ i∈I(j)).$$

**Concatenation of two paths** $s$ and $s^′$ is denoted $s;s^′.$


## Sets
The set of **all paths** is the product set of all states

$$S=∏_{j∈C∪D} S_j.$$

The set of **information paths** of node $j∈N$ is the product set of the states in its information set

$$S_{I(j)}=∏_{i∈I(j)} S_i.$$


## Probabilities
For each chance node $j∈C$, the **probability** of state $s_j$ given information state $s_{I(j)}$ is denoted as

$$ℙ(X_j=s_j∣X_{I(j)}=s_{I(j)})∈[0, 1].$$

The **upper bound of the probability of a path** $s$ is defined as

$$p(s) = ∏_{j∈C} ℙ(X_j=s_j∣X_{I(j)}=s_{I(j)}).$$


## Decisions
For each decision node $j∈D,$ a **local decision strategy** maps an information path to a state

$$Z_j:S_{I(j)}↦S_j$$

**Decision strategy** $Z$ contains one local decision strategy for each decision node. Set of **all decision strategies** is denoted $ℤ.$

## Consequences
For each value node $j∈V$, the **consequence** given information state $S_{I(j)}$ is defined as

$$Y_v:S_{I(j)}↦ℂ$$


## Utilities
The **utility function** maps consequence to real-valued utilities

$$U:ℂ↦ℝ$$

Affine transformation to positive utilities

$$U^′(c) = U(c) - \min_{c∈ℂ}U(c)$$

The **utility of a path**

$$\mathcal{U}(s) = ∑_{j∈V} U^′(Y_j(s_{I(j)}))$$


## Model Formulation
The probability distribution of paths depends on the decision strategy. We model this distribution as the variable $π$ and denote the **probability of path** $s$ as $π(s).$

Decision strategy $Z_j(s_I(j))=s_j$ is equivalent to $z(s_j∣s_{I(j)})=1$ and $∑_{s_j∈S_j} z(s_j∣s_{I(j)})=1$ for all $j∈D, s_{I(j)}∈S_{I(j)}.$

The mixed-integer linear program maximizes the expected utility over all decision strategies as follows.

$$\begin{aligned}
\underset{Z∈ℤ}{\text{maximize}}\quad
& ∑_{s∈S} π(s) \mathcal{U}(s) \\
\text{subject to}\quad
& ∑_{s_j∈S_j} z(s_j∣s_{I(j)})=1,\quad ∀j∈D, s_{I(j)}∈S_{I(j)} \\
& 0≤π(s)≤p(s),\quad ∀s∈S \\
& π(s) ≤ z(s_j∣s_{I(j)}),\quad ∀j∈D, s∈S \\
& z(s_j∣s_{I(j)}) ∈ \{0,1\},\quad ∀j∈D, s_j∈S_j, s_{I(j)}∈S_{I(j)}
\end{aligned}$$

We discuss an extension to the model on the Extension page.


## Lazy Cuts
Probability sum cut

$$∑_{s∈S}π(s)=1$$

Number of pats cut

$$∑_{s∈S}π(s)/p(s)=n_{s}$$


## Analyzing Results
### Active Paths
An **active path** is path $s∈S$ with positive path probability $π(s)>0.$ Then, we have the set of all active paths $S^+=\{s∈S∣π(s)>0\}.$ We denote the number of active paths as $|S^+|.$

### State Probabilities
We denote **paths with fixed states** where $ϵ$ denotes an empty state using a recursive definition.

$$\begin{aligned}
S_{ϵ} &= S \\
S_{ϵ,s_i^′} &= \{s∈S_{ϵ} ∣ s_i=s_i^′\} \\
S_{ϵ,s_i^′,s_j^′} &= \{s∈S_{ϵ,s_i^′} ∣ s_j=s_j^′\},\quad j≠i
\end{aligned}$$

The probability of all paths sums to one.

$$ℙ(ϵ) = \sum_{s∈S_ϵ} π(s) = 1.$$

**State probabilities** for each node $i∈C∪D$ and state $s_i∈S_i$ denote how likely the state occurs given all path probabilities

$$ℙ(s_i∣ϵ) = \sum_{s∈S_{ϵ,s_i}} π(s) / ℙ(ϵ) = \sum_{s∈S_{ϵ,s_i}} π(s)$$

An **active state** is a state with positive state probability $ℙ(s_i∣...)>0.$

We can **generalize the state probabilities** as conditional probabilities using a recursive definition. Generalized state probabilities allow us to explore how fixing active states affect the probabilities of other states. First, we choose an active state $s_i$ and fix its value. Fixing an inactive state would make all state probabilities zero. Then, we can compute the conditional state probabilities as follows.

$$ℙ(s_j∣ϵ,s_i) = \sum_{s∈S_{ϵ,s_i,s_j}} π(s) / ℙ(s_i∣ϵ)$$

We can then repeat this process by choosing an active state from the new conditional state probabilities $s_k$ that is different from previously chosen states $k≠j.$

A **robust recommendation** is a set of conditions such that a decision state $s_i$ where $i∈D$ has a state probability of one $ℙ(s_i∣...)=1.$


## Complexity
States and paths

*  $⋃_{i∈C} (S_{I(i)}×S_i)$ probability stages
*  $⋃_{i∈D} (S_{I(i)}×S_i)$ decision stages
*  $⋃_{i∈V} S_{I(i)}$ utility stages

Sizes

*  $|S|=∏_{i∈C∪D} |S_i|$ Number of paths
*  $∑_{i∈C}|S_{I(i)}| |S_i|$ Number of probability stages
*  $∑_{i∈D}|S_{I(i)}| |S_i|$ Number of decision stages
*  $∑_{v∈V}|S_{I(v)}|$ Number of utility stages


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)
