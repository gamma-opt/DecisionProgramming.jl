# Paths and Properties
## Effective Paths
![](figures/paths_eff.svg)

It is possible for some combinations of chance or decision states to be unrealizable. We refer to such subpaths as ineffective. For example, the above tree represents the generation of paths where subpaths $𝐒_{\{1,2\}}^′=\{(2,2)\}$, $𝐒_{\{1,2,3\}}^′=\{(1,1,2), (1,2,1)\}$ are ineffective.

Formally, the path $𝐬$ is **ineffective** if and only if $𝐬_A∈𝐒_A^′$ given ineffective subpaths $𝐒_A^′⊆𝐒_A$ for nodes $A⊆C∪D.$ Then, **effective paths** is a subset of all paths without ineffective paths

$$𝐒^∗=\{𝐬∈𝐒∣𝐬_{A}∉𝐒_{A}^′\}⊆𝐒.$$

The size of the [Decision Model](@ref) depends on the number of effective paths, rather than the number of paths or size of the influence diagram directly. If effective paths is empty, the influence diagram has no solutions.


## Active Paths
If the upper bound of path probability is zero, its probability is zero, and it has no effect on the solution. Therefore, we can only consider paths with positive upper bound of path probability. We refer to these paths as active paths. Formally, we define an **active path** as a path $𝐬$ if all of its chance states are active

$$\begin{aligned}
X(𝐬)&↔(p(𝐬)>0)\\ &↔ ⋀_{j∈C} (ℙ(X_j=𝐬_j∣X_{I(j)}=𝐬_{I(j)})>0).
\end{aligned}$$

Otherwise, it is an **inactive path**. We denote the set of **active paths** as

$$𝐒(X)=\{𝐬∈𝐒 ∣ X(𝐬)\}.$$

The **number of active paths** is

$$|𝐒(X)|≤|𝐒|.$$

Effective paths belong to the active paths

$$𝐒^∗ ⊆ 𝐒(X).$$


## Compatible Paths
Each decision strategy $Z∈ℤ$ chooses a set of paths from all paths, referred to as compatible paths. Formally, we denote the set of **compatible paths** as

$$𝐒(Z)=\{𝐬∈𝐒 ∣ Z(𝐬)\}.$$

Since each local decision strategy $Z_j∈Z$ can choose only one of its states, the **number of compatible paths** is

$$|𝐒(Z)|=|𝐒|/|𝐒_D|=|𝐒_C|.$$

The compatible paths of all distinct pairs of decision strategies are disjoint. Formally, for all $Z_1,Z_2∈ℤ$ where $Z_1≠Z_2$, we have

$$𝐒(Z_1)∩𝐒(Z_2)=\{𝐬∈𝐒∣Z_1(𝐬)∧Z_2(𝐬)\}=\{s∈𝐒∣⊥\}=∅.$$


## Symmetry
We define the set of active and compatible paths as

$$𝐒(X)∩𝐒(Z)=\{𝐬∈𝐒∣X(𝐬)∧Z(𝐬)\}.$$

An influence diagram is **symmetric** if the number of active and compatible paths is a constant. Formally, if for all $Z_1,Z_2∈ℤ,$ where $Z_1≠Z_2,$ we have

$$|𝐒(X)∩𝐒(Z_1)|=|𝐒(X)∩𝐒(Z_2)|.$$

Otherwise, the influence diagram is **asymmetric**. The figures below demonstrate symmetric and asymmetric influence diagrams.

### Example 1

![](figures/id1.svg)

Consider the influence diagram with two nodes. The first is a decision node with two states, and the second is a chance node with three states.

If all paths are active $X(𝐬)↔⊤$ then $𝐒(X)∩𝐒(Z)=𝐒(Z).$

![](figures/paths1.svg)

### Example 2
If there are no inactive chance states, all paths are possible. That is, for all $s∈S,$ we have $p(s)>0.$ In this case, the influence diagram is symmetric.

![](figures/paths2.svg)

However, if there are inactive chance states, such as $ℙ(s_2=2∣s_1=2)=0$, we can remove $(2,2)$ from the paths, visualized by a dashed shape. Therefore, there is a varying number of possible paths depending on whether the decision-maker chooses state $s_1=1$ or $s_1=2$ in the first node, and the influence diagram is asymmetric.

### Example 3
![](figures/id2.svg)

Let us add one chance node with two states to the influence diagram.

![](figures/paths3.svg)

Now, given inactive chance states such that we remove the dashed paths, we have a symmetric influence diagram. Both decisions will have an equal number of possible paths. However, there are only eight possible paths instead of twelve if there were no inactive chance states.


## Other Properties
In this section, we define more properties for influence diagrams.

**Discrete** influence diagram refers to countable state space. Otherwise, the influence diagram is **continuous**. We can discretize continuous influence diagrams using discrete bins.

Two nodes are **sequential** if there exists a directed path from one node to the other in the influence diagram. Otherwise, the nodes are **parallel**. Sequential nodes often model time dimension.

**Repeated subdiagram** refers to a recurring pattern within an influence diagram. Often, influence diagrams do not have a unique structure, but they consist of a repeated pattern due to the underlying problem's properties.

**Limited-memory** influence diagram refers to an influence diagram where an upper bound limits the size of the information set for decision nodes. That is, $I(j)≤m$ for all $j∈D$ where the limit $m$ is less than $|C∪D|.$ Smaller limits of $m$ are desirable because they reduce the decision model size, as discussed in the [Computational Complexity](@ref computational-complexity) page.

**Isolated subdiagrams** refer to an influence diagram that consists of multiple unconnected diagrams. That is, there are no undirected connections between the diagrams. Therefore, one isolated subdiagram's decisions affect decisions on the other isolated subdiagrams only through the utility function.

Chance or decision node is **redundant** if it is a leaf node and not in any value node's information set. Formally, if $j∈C∪D$ is a leaf node and there does not exist a value node $i∈V$ such that $j∈I(i).$
