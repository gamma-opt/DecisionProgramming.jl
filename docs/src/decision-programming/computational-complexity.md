# [Computational Complexity](@id computational-complexity)
## Introduction
Decision programming relies on mixed-integer linear programming, which is known to be an NP-hard problem. This section provides an overview on computational complexity of solving influence diagrams using decision programming.

## RJT model

2-3 magnitudes faster solving times are expected using RJT formulations compared to path-based formulations. [^1] In problems with small treewidths, such as the pig breeding problem, the model size does not grow exponentially with diagram size, which makes computational benefits even larger. More analysis on computational complexity of RJT models can be found from Herrala et al. (2024) [^1].

## Path-based model
### Definitions
We use the following inequalities for sum and product of non-negative elements $A$ to derive the lower and upper bounds for the number of paths and the number of decision variables. Sum inequality:

$$|A| \left(\min_{a∈A} a\right) ≤ ∑_{a∈A} a ≤ |A| \left(\max_{a∈A} a\right).$$

Product inequality:

$$\left(\min_{a∈A} a\right)^{|A|} ≤ ∏_{a∈A} a ≤ \left(\max_{a∈A} a\right)^{|A|}.$$

The following bounds for the number of paths and the number of decision variables show how the number of states, nodes, and arcs affects the size of the model.


### Number of Paths
From the definition of the influence diagram, we know the path length $n=|C∪D|.$ Then, we have the bounds for the number of paths as

$$\left(\min_{i∈C∪D} |S_i|\right)^n ≤ |𝐒| ≤ \left(\max_{i∈C∪D} |S_i|\right)^n.$$

We assume that all nodes $i∈C∪D$ are non-trivial. That is, each decision or chance node has at least two states $|S_i|≥2.$ Then, the number of paths is exponential to the path length $n.$


### Number of Decision Variables
We define the number of decision variables as

$$∑_{i∈D}|𝐒_{I(i)∪\{i\}}| = ∑_{i∈D} ∏_{j∈I(i)∪\{i\}}|S_j|.$$

From the definition of the information set, for all $i∈D$ we have $I(i)∪\{i\}⊆C∪D,$ with size $1≤|I(i)∪\{i\}|=|I(i)|+1≤m≤n$ where $m$ denotes the upper bound of influence other nodes have on any decision node. Also, we have the number of decision nodes $0≤|D|≤n.$ Thus, we have the bounds

$$0 ≤ ∑_{i∈D}|𝐒_{I(i)∪\{i\}}| ≤ |D| \left(\max_{i∈C∪D} |S_j|\right)^{m}.$$

In the worst case, $m=n$, a decision node is influenced by every other chance and decision node. However, in most practical cases, we have $m < n,$ where decision nodes are influenced only by a limited number of other chance and decision nodes, making models easier to solve.

### Numerical challenges

As has become evident above, in Decision Programming the size of the [Decision Model](path-based-model.md) may become large if the influence diagram has a large number of nodes or nodes with a large number of states. In practice, this results in having a large number of path compatibility and decision variables. This may result in numerical challenges.

#### Probability Scaling Factor
If an influence diagram has a large number of nodes or some nodes have a large set of states, the path probabilities $p(𝐬)$ become increasingly small. This may cause numerical issues with the solver, even prevent it from finding a solution. This issue is showcased in the [CHD preventative care example](../examples/CHD_preventative_care.md).

The issue may be helped by multiplying the path probabilities with a scaling factor $\gamma > 0$. For example, the objective function becomes

$$\operatorname{E}(Z) = ∑_{𝐬∈𝐒} x(𝐬) \ p(𝐬) \ \gamma \ \mathcal{U}(𝐬).$$

The path probabilities should also be scaled in other objective functions or constraints, including the conditional value-at-risk function and the probability cut constraint $∑_{𝐬∈𝐒}x(𝐬) p(𝐬) = 1$.

[^1]: Herrala, O., Terho, T., Oliveira, F., 2024. Risk-averse decision strategies for influence diagrams using rooted junction trees. Retrieved from [https://arxiv.org/abs/2401.03734]