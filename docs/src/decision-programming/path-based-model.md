# [Path-based model](path-based-model.md)
## Introduction

This section introduces path variables and how to structure an optimization problem based on them. Generally solution times are slower for path based formulations than for RJT based formulations and thus using [RJT formulations](RJT-model.md) is recommended.

## Paths
![](figures/paths.svg)

In influence diagrams, paths represent realizations of states for chance and decision nodes. For example, the above tree represents generating all paths with states $S_1=\{1,2\}$ and $S_2=\{1,2,3\}.$

Formally, a **path** is a sequence of states

$$𝐬=(s_1, s_2, ...,s_n)∈𝐒,$$

where a state $s_i∈S_i$ is defined for all chance and decision nodes $i∈C∪D.$ We denote the set of **paths** as

$$𝐒=∏_{j∈C∪D} S_j=S_1×S_2×...×S_n.$$

We define a **subpath** of $𝐬$ with $A⊆C∪D$ as a subsequence

$$𝐬_A=(𝐬_{i}∣i∈A)∈𝐒_A.$$

We denote the set of **subpaths** as

$$𝐒_A=∏_{i∈A} S_i.$$

We define the **number of paths** as

$$|𝐒_A|=∏_{i∈A}|S_i|.$$

As mentioned above, each node $j∈N$ has an information set $I(j)$. A subpath, which is formed by the states of the nodes in the information set, is referred to as an **information state**  $𝐬_{I(j)}$ of node $j$. The set of these subpaths is called the **information states** $𝐒_{I(j)}$ of node $j∈N.$

Also note that $𝐒=𝐒_{C∪D},$ and $𝐒_{i}=S_i$ and $𝐬_i=s_i$ where $i∈C∪D$ is an individual node.


## Probabilities
Each chance node is associated with a set of discrete probability distributions over its states. Each of the probability distributions corresponds to one of the node's information states. Formally, for each chance node $j∈C$, we denote the **probability** of state $s_j$ given information state $𝐬_{I(j)}$ as

$$ℙ(X_j=s_j∣X_{I(j)}=𝐬_{I(j)})∈[0, 1],$$

with

$$∑_{s_j∈S_j} ℙ(X_j=s_j∣X_{I(j)}=𝐬_{I(j)}) = 1.$$

A chance state with a given information state is considered **active** if its probability is nonzero

$$ℙ(X_j=s_j∣X_{I(j)}=𝐬_{I(j)})>0.$$

Otherwise, it is **inactive**.


## Decision Strategies
Each decision strategy models how the decision maker chooses a state $s_j∈S_j$ given an information state $𝐬_{I(j)}$ at decision node $j∈D.$ A decision node can be seen as a special type of chance node, such that the probability of the chosen state given an information state is fixed to one

$$ℙ(X_j=s_j∣X_{I(j)}=𝐬_{I(j)})=1.$$

By definition, the probabilities for other states are zero.

Formally, for each decision node $j∈D,$ a **local decision strategy** is a function that maps an information state $𝐬_{I(j)}$ to a state $s_j$

$$Z_j:𝐒_{I(j)}↦S_j.$$

A **decision strategy** contains one local decision strategy for each decision node

$$Z=\{Z_j∣j∈D\}.$$

The set of **all decision strategies** is denoted with $ℤ.$



## [Path Probability](@id path-probability-doc)
The probability distributions at chance and decision nodes define the probability distribution over all paths $𝐬∈𝐒,$ which depends on the decision strategy $Z∈ℤ.$ We refer to it as the path probability

$$ℙ(X=𝐬∣Z) = ∏_{j∈C∪D} ℙ(X_j=𝐬_j∣X_{I(j)}=𝐬_{I(j)}).$$

We can decompose the path probability into two parts

$$ℙ(X=𝐬∣Z) = p(𝐬) q(𝐬∣Z).$$

The first part consists of the probability contributed by the chance nodes. We refer to it as the **upper bound of path probability**

$$p(𝐬) = ∏_{j∈C} ℙ(X_j=𝐬_j∣X_{I(j)}=𝐬_{I(j)}).$$

The second part consists of the probability contributed by the decision nodes.

$$q(𝐬∣Z) = ∏_{j∈D} ℙ(X_j=𝐬_j∣X_{I(j)}=𝐬_{I(j)}).$$

Because the probabilities of decision nodes are defined as one or zero depending on the decision strategy, we can simplify the second part to an indicator function

$$q(𝐬∣Z)=\begin{cases}
1, & x(𝐬) = 1 \\
0, & \text{otherwise}
\end{cases}.$$

The binary variable $x(𝐬)$ indicates whether a decision stategy is **compatible** with the path $𝐬,$ that is, if each local decision strategy chooses a state on the path. Using the indicator function $I(.)$ whose value is 1 if the expression inside is *true* and 0 otherwise, we have

$$x(𝐬) = \prod_{j∈D} I(Z_j(𝐬_{I(j)})=𝐬_j).$$

Now the **path probability** equals the upper bound if the path is compatible with given decision strategy. Otherwise, the path probability is zero. Formally, we have

$$ℙ(𝐬∣X,Z)=
\begin{cases}
p(𝐬), & x(𝐬) = 1 \\
0, & \text{otherwise}
\end{cases}.$$


## Consequences
For each value node $j∈V$, we define the **consequence** given information state $𝐬_{I(j)}$ as

$$Y_j:𝐒_{I(j)}↦ℂ,$$

where $ℂ$ is the set of real-valued consequences.


## Path Utility
The **utility function** is a function that maps consequences to real-valued utility

$$U:ℂ^{|V|}↦ℝ.$$

The **path utility** is defined as the utility function acting on the consequences of value nodes given their information states

$$\mathcal{U}(𝐬) = U(\{Y_j(𝐬_{I(j)}) ∣ j∈V\}).$$

The **default path utility** is the sum of node utilities $U_j$

$$\mathcal{U}(𝐬) = ∑_{j∈V} U_j(Y_j(𝐬_{I(j)})).$$

The utility function affects the objectives as discussed on the [Decision Model](#Decision Model) page. We can choose the utility function such that the path utility function either returns:

* a numerical value, which leads to a mixed-integer linear programming (MILP) formulation or
* a linear function with real and integer-valued variables, which leads to a mixed-integer quadratic programming (MIQP) formulation.

Different formulations require a solver capable of solving them.


## Path Distribution
A **path distribution** is a pair

$$(ℙ(X=𝐬∣Z), \mathcal{U}(𝐬))$$

that comprises of a path probability function and a path utility function over paths $𝐬∈𝐒$ conditional to the decision strategy $Z.$



## Effective Paths
![](figures/paths_eff.svg)

It is possible for some combinations of chance or decision states to be unrealizable. We refer to such subpaths as ineffective. For example, the above tree represents the generation of paths where subpaths $𝐒_{\{1,2\}}^′=\{(2,2)\}$, $𝐒_{\{1,2,3\}}^′=\{(1,1,2), (1,2,1)\}$ are ineffective.

Formally, the path $𝐬$ is **ineffective** if and only if $𝐬_A∈𝐒_A^′$ given ineffective subpaths $𝐒_A^′⊆𝐒_A$ for nodes $A⊆C∪D.$ Then, **effective paths** is a subset of all paths without ineffective paths

$$𝐒^∗=\{𝐬∈𝐒∣𝐬_{A}∉𝐒_{A}^′\}⊆𝐒.$$

The [Decision Model](#Decision Model) size depends on the number of effective paths, rather than the number of paths or size of the influence diagram directly.

In Decision Programming, one can declare certain subpaths to be ineffective using the *fixed path* and *forbidden paths* sets.

### Fixed Path
**Fixed path** refers to a subpath which must be realized. If the fixed path is $s_Y = S_Y^f$ for all nodes $Y⊆C∪D$, then the effective paths in the model are

$$𝐒^∗=\{𝐬∈𝐒∣s_{Y} = S_{Y}^f \forall \ Y \}.$$


### Forbidden Paths
**Forbidden paths** are a way to declare ineffective subpaths. If $𝐬_X∈𝐒_X^′$ are forbidden subpaths for nodes $X⊆C∪D$, then the effective paths in the model are

$$𝐒^∗=\{𝐬∈𝐒∣𝐬_{X} ∉ 𝐒_{X}^′\}.$$



## Active Paths
If the upper bound of path probability is zero, its probability is zero, and it does not affect the solution. Therefore, we can consider only the paths with a positive upper bound of path probability. We refer to these paths as active paths. Formally, we define an **active path** as a path $𝐬$ if all of its chance states are active

$$\begin{aligned}
X(𝐬)&↔(p(𝐬)>0)\\ &↔ ⋀_{j∈C} (ℙ(X_j=𝐬_j∣X_{I(j)}=𝐬_{I(j)})>0).
\end{aligned}$$

Otherwise, it is an **inactive path**. We denote the set of **active paths** as

$$𝐒(X)=\{𝐬∈𝐒 ∣ X(𝐬)\}.$$

The **number of active paths** is

$$|𝐒(X)|≤|𝐒|.$$

Effective paths are related to active paths, such that, for all $j∈C,$ we have ineffective subpaths

$$𝐒_{I(j)∪j}^′=\{𝐬_{I(j)∪j}∈𝐒_{I(j)∪j} ∣ ℙ(X_j=s_j∣X_{I(j)}=𝐬_{I(j)})=0\}.$$

Generally, the effective paths is a subset of the active paths, that is

$$𝐒^∗ ⊆ 𝐒(X).$$

If there are no other ineffective subpaths, we have

$$𝐒^∗ = 𝐒(X).$$

Notice that, the number of active paths affects the size of the [Decision Model](#Decision Model) because it depends on the number of effective paths.


## Compatible Paths
Each decision strategy $Z∈ℤ$ determines a set of **compatible paths**. We use the shorthand $Z(s) ↔ (q(𝐬 \mid Z) = 1)$, where q is as defined in [Path Probability](@ref path-probability-doc). Formally, we denote the set of compatible paths as

$$𝐒(Z)=\{𝐬∈𝐒 ∣ Z(𝐬)\}.$$

Since each local decision strategy $Z_j∈Z$ is deterministic, it can choose only one state $s_j$ for each information state $𝐬_{I(j)}$. Thus, the **number of compatible paths** is

$$|𝐒(Z)|=|𝐒|/|𝐒_D|=|𝐒_C|.$$

The compatible paths of all distinct pairs of decision strategies are disjoint. Formally, for all $Z,Z^′∈ℤ$ where $Z≠Z^′$, we have

$$𝐒(Z)∩𝐒(Z^′)=\{𝐬∈𝐒∣Z(𝐬)∧Z^′(𝐬)\}=∅.$$


### Locally Compatible Paths
**Locally compatible paths** refers to a subset of paths that include the subpath $(𝐬_{I(j)}, s_j)$ and thus, represent the local decision strategy $Z_j(𝐬_{I(j)}) = s_j$ for decision node $j \in D$. Formally, the locally compatible paths for node $j \in D$, state $s_j \in S_j$ and information state $𝐬_{I(j)} \in 𝐒_{I(j)}$ includes the paths

$$𝐒_{s_j \mid s_{I(j)}} = \{ 𝐬 \in 𝐒 \mid (𝐬_{I(j)}, s_j) ⊂ 𝐬\}.$$


## Symmetry
We define the set of active and compatible paths as

$$𝐒(X)∩𝐒(Z)=\{𝐬∈𝐒∣X(𝐬)∧Z(𝐬)\}.$$

An influence diagram is **symmetric** if the number of active and compatible paths is a constant. Formally, if for all $Z,Z^′∈ℤ,$ where $Z≠Z^′,$ we have

$$|𝐒(X)∩𝐒(Z)|=|𝐒(X)∩𝐒(Z^′)|.$$

For example, if all paths are active, we have $|𝐒(X)∩𝐒(Z)|=|𝐒(Z)|,$ which is a constant. Otherwise, the influence diagram is **asymmetric**. The figures below demonstrate symmetric and asymmetric influence diagrams.

### Example 1

![](figures/id1.svg)

Consider the influence diagram with two nodes. The first is a decision node with two states, and the second is a chance node with three states.

![](figures/paths1.svg)

If there are no inactive chance states, all paths are possible. That is, for all $s∈S,$ we have $p(s)>0.$ In this case, the influence diagram is symmetric.

### Example 2

![](figures/paths2.svg)

However, if there are inactive chance states, such as $ℙ(s_2=2∣s_1=2)=0$, we can remove $(2,2)$ from the paths, visualized by a dashed shape. Therefore, there is a varying number of possible paths depending on whether the decision-maker chooses state $s_1=1$ or $s_1=2$ in the first node, and the influence diagram is asymmetric.

### Example 3
![](figures/id2.svg)

Let us add one chance node with two states to the influence diagram.

![](figures/paths3.svg)

Now, given inactive chance states such that we remove the dashed paths, we have a symmetric influence diagram. Both decisions will have an equal number of possible paths. However, there are only eight possible paths instead of twelve if there were no inactive chance states.



## Decision Model
We aim to find an optimal decision strategy $Z$ among all decision strategies $ℤ$ by maximizing an objective function $f$ on the path distribution of an influence diagram

$$\underset{Z∈ℤ}{\text{maximize}}\quad f(\{(ℙ(X=𝐬∣Z), \mathcal{U}(𝐬)) ∣ 𝐬∈𝐒\}). \tag{1}$$

**Decision model** refers to the mixed-integer linear programming formulation of this optimization problem. This page explains how to express decision strategies, compatible paths, path utilities and the objective of the model as a mixed-integer linear program. We present two standard objective functions, including expected value and conditional value-at-risk. The original decision model formulation was described in [^1], sections 3 and 5. We base the decision model on an improved formulation described in [^2] section 3.3. We recommend reading the references for motivation, details, and proofs of the formulation.


## Decision Variables
Decision variables $z(s_j∣𝐬_{I(j)})$ are equivalent to local decision strategies such that $Z_j(𝐬_{I(j)})=s_j$ if and only if $z(s_j∣𝐬_{I(j)})=1$ and $z(s_{j}^′∣𝐬_{I(j)})=0$ for all $s_{j}^′∈S_j∖s_j.$ Constraint $(2)$ defines the decisions to be binary variables and the constraint $(3)$ states that only one decision alternative $s_{j}$ can be chosen for each information set $s_{I(j)}$.

$$z(s_j∣𝐬_{I(j)}) ∈ \{0,1\},\quad ∀j∈D, s_j∈S_j, 𝐬_{I(j)}∈𝐒_{I(j)} \tag{2}$$

$$∑_{s_j∈S_j} z(s_j∣𝐬_{I(j)})=1,\quad ∀j∈D, 𝐬_{I(j)}∈𝐒_{I(j)} \tag{3}$$


## Path Compatibility Variables
Path compatibility variables $x(𝐬)$ are indicator variables for whether path $𝐬$ is compatible with decision strategy $Z$ defined by the decision variables $z$. These are continous variables but only assume binary values, so that the compatible paths $𝐬 ∈ 𝐒(Z)$ take values $x(𝐬) = 1$ and other paths $𝐬 ∈ 𝐒 \setminus 𝐒(Z)$ take values $x(𝐬) = 0$. Constraint $(4)$ defines the lower and upper bounds for the variables.

$$0≤x(𝐬)≤1,\quad ∀𝐬∈𝐒 \tag{4}$$

Constraint $(5)$ ensures that only the variables associated with locally compatible paths $𝐬 \in 𝐒_{s_j | 𝐬_{I(j)} }$ of the decision strategy can take value $x(𝐬) = 1$. The effective locally compatible paths are denoted with $| 𝐒^*_{s_j | 𝐬_{I(j)}}|$. The upper bound of the constraint uses the minimum of the *feasible paths* upper bound and the *theoretical* upper bound. The motivation of the feasible paths upper bound is below. For proofs and motivation on the theoretical upper bound see reference [^2].

$$∑_{𝐬 \in 𝐒^*_{s_j | 𝐬_{I(j)}} } x(𝐬) \leq \min ( \ | 𝐒^*_{s_j | 𝐬_{I(j)}}|, \ \frac{| 𝐒_{s_j | 𝐬_{I(j)}}| }{\displaystyle  \prod_{d \in D \setminus \{j, I(j)\}} |𝐒_d|} \ ) \ z(s_j∣𝐬_{I(j)}),\quad \forall j \in D, s_j \in S_j, 𝐬_{I(j)} \in 𝐒_{I(j)} \tag{5}$$

Constraint $(6)$ is called the probability cut constraint and it defines that the sum of the path probabilities of the compatible paths must equal one.

$$∑_{𝐬∈𝐒}x(𝐬) p(𝐬) = 1 \tag{6}$$

### Feasible paths upper bound
The *feasible paths upper bound* for the path compatibility variables is

$$∑_{𝐬 \in 𝐒^*_{s_j | 𝐬_{I(j)}} } x(𝐬) \leq  | 𝐒^*_{s_j | 𝐬_{I(j)}}| \ z(s_j∣𝐬_{I(j)}),\quad \forall j \in D, s_j \in S_j, 𝐬_{I(j)} \in 𝐒_{I(j)}$$

where $𝐒^*_{s_j | s_{I(j)}}$ is the set of effective locally compatible paths. This upper bound is motivated by the implementation of the framework in which path compatibility variables $x(𝐬)$ are only generated for effective paths $𝐬 \in 𝐒^∗$. The ineffective paths are not generated because they do not influence the objective function and having less variables reduces the size of the model.

Therefore, if the model has ineffective paths $𝐬 \in 𝐒^′$, then the number of effective paths is less than the number of all paths.

$$|𝐒^*| < |𝐒|$$

Therefore,

$$|𝐒^*_{s_j | 𝐬_{I(j)}} | < | 𝐒_{s_j | 𝐬_{I(j)}}| .$$

The feasible paths upper bound is used in conjunction with the *theoretical upper bound* as follows.

$$∑_{𝐬 \in 𝐒^*_{s_j | 𝐬_{I(j)}} } x(𝐬) \leq \min ( \ | 𝐒^*_{s_j | 𝐬_{I(j)}}|, \ \frac{| 𝐒_{s_j | 𝐬_{I(j)}}| }{\displaystyle  \prod_{d \in D \setminus \{j, I(j)\}} |𝐒_d|} \ ) \ z(s_j∣𝐬_{I(j)}),\quad \forall j \in D, s_j \in S_j, 𝐬_{I(j)} \in 𝐒_{I(j)}$$

The motivation for using the minimum of these bounds is that it depends on the problem structure which one is tighter. The feasible paths upper bound may be tighter if the set of ineffective paths is large compared to the number of all paths.




## Lazy Probability Cut
Constraint $(6)$ is a complicating constraint involving all path compatibility variables $x(s)$ and thus adding it directly to the model may slow down the overall solution process. It may be beneficial to instead add it as a *lazy constraint*. In the solver, a lazy constraint is only generated when an incumbent solution violates it. In some instances, this allows the MILP solver to prune nodes of the branch-and-bound tree more efficiently.

## Single Policy Update
To obtain (hopefully good) starting solutions, the SPU heuristic described in [^3] can be used. The heuristic finds a locally optimal strategy in the sense that the strategy cannot be improved by changing any single local strategy. With large problems, the heuristic can quickly provide a solution that would otherwise take very long to obtain.


## Expected Value
The **expected value** objective is defined using the path compatibility variables $x(𝐬)$ and their associated path probabilities $p(𝐬)$ and path utilities $\mathcal{U}(𝐬)$.

$$\operatorname{E}(Z) = ∑_{𝐬∈𝐒} x(𝐬) \ p(𝐬) \ \mathcal{U}(𝐬). \tag{7}$$

## Positive and Negative Path Utilities
We can omit the probability cut defined in constraint $(6)$ from the model if we are maximising expected value of utility and use a **positive path utility** function $\mathcal{U}^+$. Similarly, we can use a **negative path utility** function $\mathcal{U}^-$ when minimizing expected value. These functions are affine transformations of the path utility function $\mathcal{U}$ which translate all utility values to positive/negative values. As an example of a positive path utility function, we can subtract the minimum of the original utility function and then add one as follows.

$$\mathcal{U}^+(𝐬) = \mathcal{U}(𝐬) - \min_{𝐬∈𝐒} \mathcal{U}(𝐬) + 1. \tag{8}$$

$$\mathcal{U}^-(𝐬) = \mathcal{U}(𝐬) - \max_{𝐬∈𝐒} \mathcal{U}(𝐬) - 1. \tag{9}$$

## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2022). Decision programming for mixed-integer multi-stage optimization under uncertainty. European Journal of Operational Research, 299(2), 550-565.

[^2]: Hölsä, O. (2020). Decision Programming Framework for Evaluating Testing Costs of Disease-Prone Pigs. Retrieved from [http://urn.fi/URN:NBN:fi:aalto-202009295618](http://urn.fi/URN:NBN:fi:aalto-202009295618)

[^3]: Hankimaa, H., Herrala, O., Oliveira, F., Tollander de Balsch, J. (2023). DecisionProgramming.jl -- A framework for modelling decision problems using mathematical programming. Retrieved from [https://arxiv.org/abs/2307.13299](https://arxiv.org/abs/2307.13299)