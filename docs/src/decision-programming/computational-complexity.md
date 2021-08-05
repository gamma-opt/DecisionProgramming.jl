# [Computational Complexity](@id computational-complexity)
## Introduction
Decision programming relies on mixed-integer linear programming, which is known to be an NP-hard problem. In this section, we analyze how the influence diagram affects the size of the mixed-integer linear model, determining whether it is tractable.

We use the following inequalities for sum and product of non-negative elements $A$ to derive the lower and upper bounds for the number of paths and the number of decision variables. Sum inequality:

$$|A| \left(\min_{a∈A} a\right) ≤ ∑_{a∈A} a ≤ |A| \left(\max_{a∈A} a\right).$$

Product inequality:

$$\left(\min_{a∈A} a\right)^{|A|} ≤ ∏_{a∈A} a ≤ \left(\max_{a∈A} a\right)^{|A|}.$$

The following bounds for the number of paths and the number of decision variables show how the number of states, nodes, and arcs affects the size of the model.


## Number of Paths
From the definition of the influence diagram, we know the path length $n=|C∪D|.$ Then, we have the bounds for the number of paths as

$$\left(\min_{i∈C∪D} |S_i|\right)^n ≤ |𝐒| ≤ \left(\max_{i∈C∪D} |S_i|\right)^n.$$

We assume that all nodes $i∈C∪D$ are non-trivial. That is, each decision or chance node has at least two states $|S_i|≥2.$ Then, the number of paths is exponential to the path length $n.$


## Number of Decision Variables
We define the number of decision variables as

$$∑_{i∈D}|𝐒_{I(i)∪\{i\}}| = ∑_{i∈D} ∏_{j∈I(i)∪\{i\}}|S_j|.$$

From the definition of the information set, for all $i∈D$ we have $I(i)∪\{i\}⊆C∪D,$ with size $1≤|I(i)∪\{i\}|=|I(i)|+1≤m≤n$ where $m$ denotes the upper bound of influence other nodes have on any decision node. Also, we have the number of decision nodes $0≤|D|≤n.$ Thus, we have the bounds

$$0 ≤ ∑_{i∈D}|𝐒_{I(i)∪\{i\}}| ≤ |D| \left(\max_{i∈C∪D} |S_j|\right)^{m}.$$

In the worst case, $m=n$, a decision node is influenced by every other chance and decision node. However, in most practical cases, we have $m < n,$ where decision nodes are influenced only by a limited number of other chance and decision nodes, making models easier to solve.

## Tackling tracktability challenges 

As has become evident above, in Decision Programming the [Decision Model](@ref decision-model) may become large due to a large number of paths and decisions. In practice, this means having a large number of path compatibility and decision variables.

### Feasible paths upper bound
The *feasible paths upper bound* for the path compatibility variables is 

$$∑_{s \in 𝐒^*_{s_j | s_{I(j)}} } x(s) \leq  | 𝐒^*_{s_j | s_{I(j)}}| \ z(s_j∣s_{I(j)}),\quad \forall j \in D, s_j \in S_j, s_{I(j)} \in S_{I(j)} $$

where $𝐒^*_{s_j | s_{I(j)}}$ is the set of effective locally compatible paths. This upper bound is motivated by the implementation of the framework in which path compatibility variables $x(s)$ are only generated for effective paths $s \in 𝐒^∗$. Remeber that effective paths are a subset of the active paths $ 𝐒(X) ⊆ 𝐒$. This is done because ineffective and inactive paths do not influence the objective function, and not generating them reduces the model size.

Therefore if the model has ineffective paths $s \in 𝐒^′$, then the number of effective paths is less than the number of all paths.

$$ |𝐒^*| < |𝐒|$$

Therefore,

$$ |𝐒^*_{s_j | s_{I(j)}} | < | 𝐒_{s_j | s_{I(j)}}| .$$

The feasible paths upper bound is used in conjunction with the [*theoretical upper bound*]() as follows 

$$∑_{s \in S^*_{s_j | s_{I(j)}} } x(s) \leq \min ( \ | S^*_{s_j | s_{I(j)}}|, \ \frac{| S_{s_j | s_{I(j)}}| }{\displaystyle  \prod_{d \in D \setminus \{j, I(j)\}} |S_d|} \ ) \ z(s_j∣s_{I(j)}),\quad \forall j \in D, s_j \in S_j, s_{I(j)} \in S_{I(j)}.$$

The motivation for using the minimum of these bounds is because it depends on the problem structure which one is tighter. The feasible paths upper bound may be tighter if the set of ineffective paths is large compared to the number of all paths.


### Probability Scaling Factor
In an influence diagram a large number of nodes or some nodes having a large set of states, causes the path probabilities $p(s)$ to become increasingly small. This may cause numerical and/or tractability issues with the solver. This issue is showcased in the [CHD preventative care example](../examples/CHD_preventative_care.md).

The issue may be helped by multiplying the path probabilities with a scaling factor $\gamma > 0$ in the objective function and probability cut in the following way.


$$\operatorname{E}(Z) = ∑_{𝐬∈𝐒} x(𝐬) \ p(𝐬) \ \gamma \ \mathcal{U}(𝐬)$$

$$∑_{𝐬∈𝐒}x(s) \ p(s) \ \gamma = \gamma $$