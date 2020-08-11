# Analyzing Decision Strategies
## Introduction
We can analyze fixed decision strategies $Z$ on an influence diagram $G$, such as ones resulting from the optimization, by generating the active paths $𝐒^Z.$


## Active Paths
We can generate active paths $𝐬∈𝐒^Z$ as follows.

1) Initialize path $𝐬$ of length $n$ with undefined values.
2) Fill path with chance states $𝐬_j∈S_j$ for all $j∈C.$
3) In increasing order of decision nodes $j∈D$, fill decision states by computing decision strategy $𝐬_j=Z_j(𝐬_{I(j)}).$

The path probability for all active paths is equal to the upper bound

$$ℙ(𝐬∣Z)=p(𝐬), \quad ∀𝐬∈𝐒^Z.$$

We exclude inactive paths from the analysis because their path probabilities are zero.


## Utility Distribution
We define unique path utility values as

$$\mathcal{U}^∗=\{\mathcal{U}(𝐬)∣𝐬∈𝐒^Z\}.$$

The probability mass function of the **utility distribution** associates each unique path utility to a probability as follows

$$ℙ(X=u)=∑_{𝐬∈𝐒^Z∣\mathcal{U}(𝐬)=u} p(𝐬),\quad ∀u∈\mathcal{U}^∗.$$

From the utility distribution, we can calculate the cumulative distribution, statistics, and risk measures. The relevant statistics are expected value, standard deviation, skewness and kurtosis. Risk measures focus on the conditional value-at-risk (CVaR), also known as, expected shortfall.


## State Probabilities
We denote **paths with fixed states** where $ϵ$ denotes an empty state using a recursive definition.

$$\begin{aligned}
𝐒_{ϵ} &= 𝐒^Z \\
𝐒_{ϵ,s_i} &= \{𝐬∈𝐒_{ϵ} ∣ 𝐬_i=s_i\} \\
𝐒_{ϵ,s_i,s_j} &= \{𝐬∈𝐒_{ϵ,s_i} ∣ 𝐬_j=s_j\},\quad j≠i
\end{aligned}$$

The probability of all paths sums to one

$$ℙ(ϵ) = \sum_{𝐬∈𝐒_ϵ} p(𝐬) = 1.$$

**State probabilities** for each node $i∈C∪D$ and state $s_i∈S_i$ denote how likely the state occurs given all path probabilities

$$ℙ(s_i∣ϵ) = \sum_{𝐬∈𝐒_{ϵ,s_i}} \frac{p(𝐬)}{ℙ(ϵ)} = \sum_{𝐬∈𝐒_{ϵ,s_i}} p(𝐬)$$

An **active state** is a state with positive state probability $ℙ(s_i∣c)>0$ given conditions $c.$

We can **generalize the state probabilities** as conditional probabilities using a recursive definition. Generalized state probabilities allow us to explore how fixing active states affect the probabilities of other states. First, we choose an active state $s_i$ and fix its value. Fixing an inactive state would make all state probabilities zero. Then, we can compute the conditional state probabilities as follows.

$$ℙ(s_j∣ϵ,s_i) = \sum_{𝐬∈𝐒_{ϵ,s_i,s_j}} \frac{p(𝐬)}{ℙ(s_i∣ϵ)}$$

We can then repeat this process by choosing an active state from the new conditional state probabilities $s_k$ that is different from previously chosen states $k≠j.$

A **robust recommendation** is a decision state $s_i$ where $i∈D$ and subpath $c$ such the state probability is one $ℙ(s_i∣c)=1.$
