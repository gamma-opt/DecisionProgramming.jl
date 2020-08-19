# Decision Model
## Introduction
**Decision programming** aims to find a decision strategy $Z$ which optimizes some metric of the path distribution on an influence diagram such as expected value or risk. The **decision model** is a mixed-integer linear programming formulation of this optimization problem. The model that is presented here, is based on [^1], sections 3 and 5. We recommend reading it for motivation, details, and proofs of the formulation.


## Objective
The mixed-integer linear program optimizes the objective function $f$, that is a measure of the path distribution, over all decision strategies as follows

$$\underset{Z∈ℤ}{\text{maximize}}\quad
f(\{(ℙ(𝐬∣Z), \mathcal{U}(𝐬)) ∣ 𝐬∈𝐒\}). \tag{1}$$

Common measures include expected value and risk metrics. The main consideration regarding the measures is that we can linearize them, and thus solve the model efficiently.


## Variables
**Decision variables** $z(s_j∣𝐬_{I(j)})$ are equivalent to the decision strategies $Z$ such that $Z_j(𝐬_I(j))=s_j$ if and only if $z(s_j∣𝐬_{I(j)})=1$ and $z(s_{j}^′∣𝐬_{I(j)})=0$ for all $s_{j}^′∈S_j∖s_j.$ Constraint $(2)$ defines the decisions to be binary variables and the constraint $(3)$ limits decisions to one per information path.

$$z(s_j∣𝐬_{I(j)}) ∈ \{0,1\},\quad ∀j∈D, s_j∈S_j, 𝐬_{I(j)}∈𝐒_{I(j)} \tag{2}$$

$$∑_{s_j∈S_j} z(s_j∣𝐬_{I(j)})=1,\quad ∀j∈D, 𝐬_{I(j)}∈𝐒_{I(j)} \tag{3}$$

**Path probability variables** $π(𝐬)$ are equivalent to the path probabilities $ℙ(𝐬∣Z)$ where decision variables $z$ define the decision strategy $Z$. The constraint $(4)$ defines the lower and upper bound to the probability, constraint $(5)$ defines that the probability equals zero if path is not compatible with the decision strategy, and constraint $(6)$ defines that probability equals path probability if the path is compatible with the decision strategy.

$$0≤π(𝐬)≤p(𝐬),\quad ∀𝐬∈𝐒 \tag{4}$$

$$π(𝐬) ≤ z(𝐬_j∣𝐬_{I(j)}),\quad ∀j∈D, 𝐬∈𝐒 \tag{5}$$

$$π(𝐬) ≥ p(𝐬) + ∑_{j∈D} z(𝐬_j∣𝐬_{I(j)}) - |D|,\quad ∀𝐬∈𝐒 \tag{6}$$


## Positive Path Utility
We can omit the constraint $(6)$ from the model if we use a **positive path utility** function $\mathcal{U}^+$ which is an affine transformation of path utility function $\mathcal{U}.$ As an example, we can subtract the minimum of the original utility function and then add one as follows.

$$\mathcal{U}^+(𝐬) = \mathcal{U}(𝐬) - \min_{𝐬∈𝐒} \mathcal{U}(𝐬) + 1.$$


## Lazy Constraints
Valid equalities are equalities that can be be derived from the problem structure. They can help in computing the optimal decision strategies, but adding them directly may slow down the overall solution process. By adding valid equalities during the solution process as *lazy constraints*, the MILP solver can prune nodes of the branch-and-bound tree more efficiently. We have the following valid equalities.

We can exploit the fact that the path probabilities sum to one by using the **probability cut** defined as

$$∑_{𝐬∈𝐒}π(𝐬)=1. \tag{7}$$

For problems where the number of active paths is known, we can exploit it by using the **active paths cut** defined as

$$∑_{𝐬∈𝐒} \frac{π(𝐬)}{p(𝐬)}=|𝐒^+(Z)|. \tag{8}$$


## Expected Value
We define the **expected value** as

$$\operatorname{E}(Z) = ∑_{𝐬∈𝐒} π(𝐬) \mathcal{U}(𝐬). \tag{?}$$

However, the expected value objective does not account for risk caused by the variablity in the path distribution.


## Conditional Value-at-Risk
Given a **probability level** $α∈(0, 1]$ and decision strategy $Z$ we denote **value-at-risk** $\operatorname{VaR}_α(Z)$ and **conditional value-at-risk** $\operatorname{CVaR}_α(Z).$

Pre-computed parameters

$$u^+=\max\{\mathcal{U}(𝐬)∣𝐬∈𝐒\}$$

$$u^-=\min\{\mathcal{U}(𝐬)∣𝐬∈𝐒\}$$

$$M=u^+-u^-$$

$$ϵ=\frac{1}{2} \min\{|\mathcal{U}(𝐬)-\mathcal{U}(𝐬^′)| ∣ |\mathcal{U}(𝐬)-\mathcal{U}(𝐬^′)| > 0, 𝐬, 𝐬^′∈𝐒\}$$

Objective

$$\min η$$

Constraints

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

Solution

$$\operatorname{VaR}_α(Z)=η \tag{?}$$

$$\operatorname{CVaR}_α(Z)=\frac{1}{α}∑_{𝐬∈𝐒}\bar{ρ}(𝐬) \mathcal{U}(𝐬)\tag{?}$$


## Mixed Objective
We can formulate

$$w \operatorname{E}(Z) + (1-w) \operatorname{CVaR}_α(Z) \tag{?}$$

where $w∈(0, 1)$ is the **trade-off** between maximization of


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)
