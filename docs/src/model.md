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

## Results
**Active paths** $\{s∣π(s)>0\}$

**Active states** $\{s_i∣π(s)>0\}$ for each node $i∈C∪D$, robust recommendations?

## Sizes
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
