# Computational Complexity
## Introduction
Decision programming relies on mixed-integer linear programming, which is known to be an NP-complete problem. In this section, we analyze how the influence diagram affects the size of the mixed-integer linear model, determining whether it is tractable.

We use the following inequalities for sum and product of non-negative elements $A$ to derive the lower and upper bounds for the number of paths and the number of decision stages. Sum inequality:

$$|A| \left(\min_{a∈A} a\right) ≤ ∑_{a∈A} a ≤ |A| \left(\max_{a∈A} a\right).$$

Product inequality:

$$\left(\min_{a∈A} a\right)^{|A|} ≤ ∏_{a∈A} a ≤ \left(\max_{a∈A} a\right)^{|A|}.$$

The following bounds for the number of paths and the number of decision stages show how the number of states, nodes, and arcs affects the size of the model.


## Number of Paths
We define the number of paths as

$$|S|=∏_{i∈C∪D} |S_i|.$$

From the definition of the influence diagram, we have the path length of $n=|C∪D|.$ Then, we have the bounds for the number of paths as

$$\left(\min_{i∈C∪D} |S_i|\right)^n ≤ |S| ≤ \left(\max_{i∈C∪D} |S_i|\right)^n.$$

We assume non-trivial influence diagram such that $|S_i|≥2$ for all $i∈C∪D$. That is, each decision or chance node has at least two states. Therefore, the number of paths is always exponential to the path length of $n.$


## Number of Decision Stages
We define the number of decision stages as

$$∑_{i∈D}|S_{I(i)}| |S_i| = ∑_{i∈D} |S_i| ∏_{j∈I(i)}|S_j| = ∑_{i∈D} ∏_{j∈I(i)∪\{i\}}|S_j|.$$

From the definition of the information set, for all $i∈D$ we have $I(i)∪\{i\}⊆C∪D,$ with size $1≤|I(i)∪\{i\}|=|I(i)|+1≤m≤n$ where $m$ denotes the upper bound of influence other nodes have on any decision node. Also, we have the number of decision nodes $0≤|D|≤n.$ Thus, we have the bounds

$$0 ≤ ∑_{i∈D}|S_{I(i)}| |S_i| ≤ |D| \left(\max_{i∈C∪D} |S_j|\right)^{m}.$$

In the worst case, $m=n$, a decision node is influenced by every other chance and decision node. However, in most practical cases, we have $m < n,$ where decision nodes are influenced only by a limited number of other chance and decision nodes, making models easier to solve.
