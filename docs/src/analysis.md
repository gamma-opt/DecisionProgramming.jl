# Analysis
## Utility Distribution
First, we generate the active paths $S^+$ from the decision strategy $Z$. Then, the unique path utility values are

$$\mathcal{U}^+=\{\mathcal{U}(s)∣s∈S^+\}.$$

The probability mass function of the **utility distribution** associates each unique path utility $u∈\mathcal{U}^+$ to a probability as follows

$$ℙ(X=u)=∑_{s∈S^+∣\mathcal{U}(s)=u} ℙ(s∣Z).$$


## State Probabilities
We denote **paths with fixed states** where $ϵ$ denotes an empty state using a recursive definition.

$$\begin{aligned}
S_{ϵ} &= S \\
S_{ϵ,s_i^′} &= \{s∈S_{ϵ} ∣ s_i=s_i^′\} \\
S_{ϵ,s_i^′,s_j^′} &= \{s∈S_{ϵ,s_i^′} ∣ s_j=s_j^′\},\quad j≠i
\end{aligned}$$

The probability of all paths sums to one

$$ℙ(ϵ) = \sum_{s∈S_ϵ} π(s) = 1.$$

**State probabilities** for each node $i∈C∪D$ and state $s_i∈S_i$ denote how likely the state occurs given all path probabilities

$$ℙ(s_i∣ϵ) = \sum_{s∈S_{ϵ,s_i}} \frac{π(s)}{ℙ(ϵ)} = \sum_{s∈S_{ϵ,s_i}} π(s)$$

An **active state** is a state with positive state probability $ℙ(s_i∣c)>0$ given conditions $c.$

We can **generalize the state probabilities** as conditional probabilities using a recursive definition. Generalized state probabilities allow us to explore how fixing active states affect the probabilities of other states. First, we choose an active state $s_i$ and fix its value. Fixing an inactive state would make all state probabilities zero. Then, we can compute the conditional state probabilities as follows.

$$ℙ(s_j∣ϵ,s_i) = \sum_{s∈S_{ϵ,s_i,s_j}} \frac{π(s)}{ℙ(s_i∣ϵ)}$$

We can then repeat this process by choosing an active state from the new conditional state probabilities $s_k$ that is different from previously chosen states $k≠j.$

A **robust recommendation** is a decision state $s_i$ where $i∈D$ and subpath $c$ such the state probability is one $ℙ(s_i∣c)=1.$
