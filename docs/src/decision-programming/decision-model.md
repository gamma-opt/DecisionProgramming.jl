# [Decision Model](@id decision-model)
## Introduction
**Decision programming** aims to find an optimal decision strategy $Z$ from all decision strategies $ℤ$ by maximizing an objective function $f$ on the path distribution of an influence diagram

$$\underset{Z∈ℤ}{\text{maximize}}\quad f(\{(ℙ(𝐬∣Z), \mathcal{U}(𝐬)) ∣ 𝐬∈𝐒\}). \tag{1}$$

**Decision model** refers to the mixed-integer linear programming formulation of this optimization problem. This page explains how to express decision strategy, path probability, path utility, and the objective in the mixed-integer linear form. We also present standard objective functions, including expected value and risk measures.  We based the decision model on [^1], sections 3 and 5. We recommend reading the references for motivation, details, and proofs of the formulation.


## Decision Variables
**Decision variables** $z(s_j∣𝐬_{I(j)})$ are equivalent to the decision strategies $Z$ such that $Z_j(𝐬_I(j))=s_j$ if and only if $z(s_j∣𝐬_{I(j)})=1$ and $z(s_{j}^′∣𝐬_{I(j)})=0$ for all $s_{j}^′∈S_j∖s_j.$ Constraint $(2)$ defines the decisions to be binary variables and the constraint $(3)$ limits decisions to one per information path.

$$z(s_j∣𝐬_{I(j)}) ∈ \{0,1\},\quad ∀j∈D, s_j∈S_j, 𝐬_{I(j)}∈𝐒_{I(j)} \tag{2}$$

$$∑_{s_j∈S_j} z(s_j∣𝐬_{I(j)})=1,\quad ∀j∈D, 𝐬_{I(j)}∈𝐒_{I(j)} \tag{3}$$


## Path Probability Variables
**Path probability variables** $π(𝐬)$ are equivalent to the path probabilities $ℙ(𝐬∣Z)$ where decision variables $z$ define the decision strategy $Z$. The constraint $(4)$ defines the lower and upper bound to the probability, constraint $(5)$ defines that the probability equals zero if path is not compatible with the decision strategy, and constraint $(6)$ defines that probability equals path probability if the path is compatible with the decision strategy.

$$0≤π(𝐬)≤p(𝐬),\quad ∀𝐬∈𝐒 \tag{4}$$

$$π(𝐬) ≤ z(𝐬_j∣𝐬_{I(j)}),\quad ∀j∈D, 𝐬∈𝐒 \tag{5}$$

$$π(𝐬) ≥ p(𝐬) + ∑_{j∈D} z(𝐬_j∣𝐬_{I(j)}) - |D|,\quad ∀𝐬∈𝐒 \tag{6}$$


## Positive Path Utility
We can omit the constraint $(6)$ from the model if we use a **positive path utility** function $\mathcal{U}^+$ which is an affine transformation of path utility function $\mathcal{U}.$ As an example, we can subtract the minimum of the original utility function and then add one as follows.

$$\mathcal{U}^+(𝐬) = \mathcal{U}(𝐬) - \min_{𝐬∈𝐒} \mathcal{U}(𝐬) + 1.$$


## Lazy Constraints
Valid equalities are equalities that can be be derived from the problem structure. They can help in computing the optimal decision strategies, but adding them directly may slow down the overall solution process. By adding valid equalities during the solution process as *lazy constraints*, the MILP solver can prune nodes of the branch-and-bound tree more efficiently. We have the following valid equalities.

### Probability Cut
We can exploit the fact that the path probabilities sum to one by using the **probability cut** defined as

$$∑_{𝐬∈𝐒}π(𝐬)=1. \tag{7}$$

### Active Paths Cut
For problems where the number of active paths is known, we can exploit it by using the **active paths cut** defined as

$$∑_{𝐬∈𝐒} \frac{π(𝐬)}{p(𝐬)}=|𝐒^+(Z)|. \tag{8}$$


## Expected Value
We define the **expected value** objective as

$$\operatorname{E}(Z) = ∑_{𝐬∈𝐒} π(𝐬) \mathcal{U}(𝐬). \tag{?}$$


## Conditional Value-at-Risk
The section [Measuring Risk](@ref) explains and visualizes the relationships between the formulation of expected value, value-at-risk and conditional value-at-risk for discrete probability distribution.

Given decision strategy $Z,$ we define the cumulative distribution of path probability variables as

$$F_Z(t) = ∑_{𝐬∈𝐒∣\mathcal{U}(𝐬)≤t} π(𝐬).$$

Given a **probability level** $α∈(0, 1],$ we define the **value-at-risk** as

$$\operatorname{VaR}_α(Z)=u_α=\sup \{\mathcal{U}(𝐬)∣𝐬∈𝐒, F_Z(\mathcal{U}(𝐬))<α\}.$$

Then, we have the paths that have path utility less than and equal to the value-at-risk as

$$𝐒_{α}^{<}=\{𝐬∈𝐒∣\mathcal{U}(𝐬)<u_α\},$$

$$𝐒_{α}^{=}=\{𝐬∈𝐒∣\mathcal{U}(𝐬)=u_α\}.$$

We define **conditional value-at-risk** as

$$\operatorname{CVaR}_α(Z)=\frac{1}{α}\left(∑_{𝐬∈𝐒_α^{<}} π(𝐬) \mathcal{U}(𝐬) + ∑_{𝐬∈𝐒_α^{=}} \left(α - ∑_{𝐬∈𝐒_α^{<}} π(𝐬) \right) \mathcal{U}(𝐬) \right).$$

We can form the conditional value-at-risk as an optimization problem. We have the following pre-computed parameters.

Lower and upper bound of the value-at-risk

$$\operatorname{VaR}_0(Z)=u^-=\min\{\mathcal{U}(𝐬)∣𝐬∈𝐒\},$$

$$\operatorname{VaR}_1(Z)=u^+=\max\{\mathcal{U}(𝐬)∣𝐬∈𝐒\}.$$

Largest difference between path utilities

$$M=u^+-u^-.$$

Half of the smallest positive difference between path utilities

$$ϵ=\frac{1}{2} \min\{|\mathcal{U}(𝐬)-\mathcal{U}(𝐬^′)| ∣ |\mathcal{U}(𝐬)-\mathcal{U}(𝐬^′)| > 0, 𝐬, 𝐬^′∈𝐒\}.$$

The objective is to minimize the variable $η$ whose optimal value is equal to the value-at-risk, that is, $\operatorname{VaR}_α(Z)=η^∗.$

$$\min η$$

We define the constraints as follows:

$$η-\mathcal{U}(𝐬)≤M λ(𝐬),\quad ∀𝐬∈𝐒 \tag{?}$$

$$η-\mathcal{U}(𝐬)≥(M+ϵ) λ(𝐬) - M,\quad ∀𝐬∈𝐒 \tag{?}$$

$$η-\mathcal{U}(𝐬)≤(M+ϵ) \bar{λ}(𝐬) - ϵ,\quad ∀𝐬∈𝐒 \tag{?}$$

$$η-\mathcal{U}(𝐬)≥M (\bar{λ}(𝐬) - 1),\quad ∀𝐬∈𝐒 \tag{?}$$

$$\bar{ρ}(𝐬) ≤ \bar{λ}(𝐬),\quad ∀𝐬∈𝐒 \tag{?}$$

$$π(𝐬) - (1 - λ(𝐬)) ≤ ρ(𝐬) ≤ λ(𝐬),\quad ∀𝐬∈𝐒 \tag{?}$$

$$ρ(𝐬) ≤ \bar{ρ}(𝐬) ≤ π(𝐬),\quad ∀𝐬∈𝐒 \tag{?}$$

$$∑_{𝐬∈𝐒}\bar{ρ}(𝐬) = α \tag{?}$$

$$\bar{λ}(𝐬), λ(𝐬)∈\{0, 1\},\quad ∀𝐬∈𝐒 \tag{?}$$

$$\bar{ρ}(𝐬),ρ(𝐬)∈[0, 1],\quad ∀𝐬∈𝐒 \tag{?}$$

$$η∈[u^-, u^+] \tag{?}$$

We can express the conditional value-at-risk objective as

$$\operatorname{CVaR}_α(Z)=\frac{1}{α}∑_{𝐬∈𝐒}\bar{ρ}(𝐬) \mathcal{U}(𝐬)\tag{?}.$$

The values of conditional value-at-risk are limited to the interval between the lower bound of value-at-risk and the expected value

$$\operatorname{VaR}_0(Z)<\operatorname{CVaR}_α(Z)≤E(Z).$$


## Mixed Objective
We can combine expected value and conditional value-at-risk using a convex combination at a fixed probability level $α$ as follows

$$w \operatorname{E}(Z) + (1-w) \operatorname{CVaR}_α(Z), \tag{?}$$

where the parameter $w∈(0, 1)$ expresses the decision maker's risk tolerance.


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)
