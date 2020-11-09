# Influence Diagram Advanced
## Effective Paths
![](figures/paths_eff.svg)

It is possible for some combinations of chance or decision states to be unrealizable. We refer to such subpaths as ineffective. For example, the above tree represents the generation of paths where subpaths $𝐒_{\{1,2\}}^′=\{(2,2)\}$, $𝐒_{\{1,2,3\}}^′=\{(1,1,2), (1,2,1)\}$ are ineffective.

Formally, the path $𝐬$ is **ineffective** if and only if $𝐬_A∈𝐒_A^′$ given ineffective subpaths $𝐒_A^′⊆𝐒_A$ for nodes $A⊆C∪D.$ Then, **effective paths** is a subset of all paths without ineffective paths

$$𝐒^∗=\{𝐬∈𝐒∣𝐬_{A}∉𝐒_{A}^′\}⊆𝐒.$$

If effective paths is empty, the influence diagram has no solutions.


## Active Paths
A path is active if all of its subpaths are active

$$X(𝐬)↔(p(𝐬)>0)↔⋀_{j∈C} (ℙ(X_j=𝐬_j∣X_{I(j)}=𝐬_{I(j)})>0).$$

The path probability of **inactive** paths is fixed to zero, irrespective of the decision strategy.

The set of **active paths** is

$$𝐒(X)=\{𝐬∈𝐒 ∣ X(𝐬)\}.$$

The number of active paths is

$$|𝐒(X)|≤|𝐒|.$$


## Compatible Paths
We denote the set of **compatible paths** as

$$𝐒(Z)=\{𝐬∈𝐒 ∣ Z(𝐬)\}.$$

Since each decision strategy $Z_j$ chooses only one of its states, the **number of compatible paths** is a constant

$$|𝐒(Z)|=|𝐒|/\prod_{j∈D}|S_j|=\prod_{j∈C}|S_j|.$$


## Active-Compatible Paths


$$𝐒(X)∩𝐒(Z)=\{𝐬∈𝐒∣X(𝐬)∧Z(𝐬)\}$$

$$|𝐒(X)∩𝐒(Z)|≤|𝐒(Z)|$$

---

If all paths are active $𝐒(X)↔T$ then

$$𝐒(X)∩𝐒(Z)=𝐒(Z)$$

---

We denote the set of **active paths** given a decision strategy $Z$ as

$$𝐒^+(Z)=\{𝐬∈𝐒 ∣ ℙ(𝐬∣Z)>0\}.$$

$$=\{𝐬∈𝐒(Z) ∣ p(𝐬)>0\}$$

By definition, the active paths is a subset of compatible paths. Therefore, the **number of active paths** is bounded by the number of compatible paths

$$|𝐒^+(Z)|≤|𝐒(Z)|.$$

If an influence diagram has **zero inactive chance states** the number of active paths is equal to the number of compatible paths

$$|𝐒^+(Z)|=|𝐒(Z)|.$$

Otherwise, the number of active paths is less than the number of compatible paths.


## Symmetry
An influence diagram is **symmetric** if the number of active paths is a constant, that is, independent of the decision strategy. Otherwise, it is **asymmetric**. With the figures below, we demonstrate both of these properties.

![](figures/id1.svg)

Consider the influence diagram with two nodes. The first is a decision node with two states, and the second is a chance node with three states.

![](figures/paths1.svg)

If there are no inactive chance states, all paths are possible. That is, for all $s∈S,$ we have $p(s)>0.$ In this case, the influence diagram is symmetric.

![](figures/paths2.svg)

However, if there are inactive chance states, such as $ℙ(s_2=2∣s_1=2)=0$, we can remove $(2,2)$ from the paths, visualized by a dashed shape. Therefore, there is a varying number of possible paths depending on whether the decision-maker chooses state $s_1=1$ or $s_1=2$ in the first node, and the influence diagram is asymmetric.

![](figures/id2.svg)

Let us add one chance node with two states to the influence diagram.

![](figures/paths3.svg)

Now, given inactive chance states such that we remove the dashed paths, we have a symmetric influence diagram. Both decisions will have an equal number of possible paths. However, there are only eight possible paths instead of twelve if there were no inactive chance states.


## Properties
In this section, we define more properties for influence diagrams.

**Discrete** influence diagram refers to countable state space. Otherwise, the influence diagram is **continuous**. We can discretize continuous influence diagrams using discrete bins.

Two nodes are **sequential** if there exists a directed path from one node to the other in the influence diagram. Otherwise, the nodes are **parallel**. Sequential nodes often model time dimension.

**Repeated subdiagram** refers to a recurring pattern within an influence diagram. Often, influence diagrams do not have a unique structure, but they consist of a repeated pattern due to the underlying problem's properties.

**Limited-memory** influence diagram refers to an influence diagram where an upper bound limits the size of the information set for decision nodes. That is, $I(j)≤m$ for all $j∈D$ where the limit $m$ is less than $|C∪D|.$ Smaller limits of $m$ are desirable because they reduce the decision model size, as discussed in the [Computational Complexity](@ref computational-complexity) page.

**Isolated subdiagrams** refer to an influence diagram that consists of multiple unconnected diagrams. That is, there are no undirected connections between the diagrams. Therefore, one isolated subdiagram's decisions affect decisions on the other isolated subdiagrams only through the utility function.

Chance or decision node is **redundant** if it is a leaf node and not in any value node's information set. Formally, if $j∈C∪D$ is a leaf node and there does not exist a value node $i∈V$ such that $j∈I(i).$
