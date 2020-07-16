# Decision Model
## Introduction
**Decision programming** aims to find decision strategies which optimizes characteristics of the path distribution on an influence diagram. The **decision model** is a mixed-integer linear programming formulation of this optimization problem. The model that is presented here, is based on [^1], sections 3 and 5. We recommend reading it for motivation, details, and proofs of the formulation.


## Formulation
The mixed-integer linear program maximizes the linear objective function that depends on the path distribution over all decision strategies as follows.

$$\underset{Z∈ℤ}{\text{maximize}}\quad
f(π, \mathcal{U}) \tag{1}$$

**Decision variables** $z(s_j∣s_{I(j)})$ are equivalent to the decision strategies $Z$ such that $Z_j(s_I(j))=s_j$ if and only if $z(s_j∣s_{I(j)})=1$, otherwise $z(s_j∣s_{I(j)})=0.$ Constraint $(2)$ defines the decisions to be binary variables and the constraint $(3)$ limits decisions to one per information path.

$$z(s_j∣s_{I(j)}) ∈ \{0,1\},\quad ∀j∈D, s_j∈S_j, s_{I(j)}∈S_{I(j)} \tag{2}$$

$$∑_{s_j∈S_j} z(s_j∣s_{I(j)})=1,\quad ∀j∈D, s_{I(j)}∈S_{I(j)} \tag{3}$$

**Path probability variables** $π(s)$ are equivalent to the path probabilities $ℙ(s∣Z)$ where decision variables $z$ define the decision strategy $Z$. The constraint $(4)$ defines the lower and upper bound to the probability, constraint $(5)$ defines that the probability equals zero if path is not compatible with the decision strategy, and constraint $(6)$ defines that probability equals path probability if the path is compatible with the decision strategy.

$$0≤π(s)≤p(s),\quad ∀s∈S \tag{4}$$

$$π(s) ≤ z(s_j∣s_{I(j)}),\quad ∀j∈D, s∈S \tag{5}$$

$$π(s) ≥ p(s) + ∑_{j∈D} z(s_j∣s_{I(j)}) - |D|,\quad ∀s∈S \tag{6}$$

We can omit the constraint $(6)$ from the model if we use a positive path utility function $\mathcal{U}^+$ which is an affine transformation of path utility function $\mathcal{U}.$ As an example, we can normalize the original utility function and then add one as follows.

$$\mathcal{U}^+(s) = \frac{\mathcal{U}(s) - \min_{s∈S} \mathcal{U}(s)}{\max_{s∈S} \mathcal{U}(s) - \min_{s∈S} \mathcal{U}(s)} + 1.$$

Next we discuss lazy constraint and concrete objective functions below.


## Lazy Constraints
Valid equalities are equalities that can be be derived from the problem structure. They can help in computing the optimal decision strategies, but adding them directly may slow down the overall solution process. By adding valid equalities during the solution process as *lazy constraints*, the MILP solver can prune nodes of the branch-and-bound tree more efficiently. We have the following valid equalities.

We can exploit the fact that the path probabilities sum to one by using the **probability sum cut**

$$∑_{s∈S}π(s)=1. \tag{7}$$

For problems where the number of active paths $|S^+|$ is known, we can exploit it by using the **number of active paths cut**

$$∑_{s∈S} \frac{π(s)}{p(s)}=|S^+|. \tag{8}$$


## Expected Value
We define the **expected value** as

$$\operatorname{E}(Z) = ∑_{s∈S} π(s) \mathcal{U}(s). \tag{?}$$

However, the expected value objective does not account for risk caused by the variablity in the path distribution.


## Value-at-Risk
Given a **probability level** $α∈(0, 1]$ and decision strategy $Z$ we denote **value-at-Risk** $\operatorname{VaR}_α(Z)$ and **conditional Value-at-Risk** $\operatorname{CVaR}_α(Z).$

Pre-computed parameters

$$c^∗=\max\{\mathcal{U}(s)∣s∈S\}$$

$$c^∘=\min\{\mathcal{U}(s)∣s∈S\}$$

$$M=c^∗-c^∘$$

$$ϵ=\frac{1}{2} \min\{|\mathcal{U}(s)-\mathcal{U}(s^′)| ∣ |\mathcal{U}(s)-\mathcal{U}(s^′)| > 0, s, s^′∈S\}$$

Objective

$$\min η$$

Constraints

$$η-\mathcal{U}(s)≤M λ(s),\quad ∀s∈S \tag{?}$$

$$η-\mathcal{U}(s)≥(M+ϵ) λ(s) - M,\quad ∀s∈S \tag{?}$$

$$η-\mathcal{U}(s)≤(M+ϵ) \bar{λ}(s) - ϵ,\quad ∀s∈S \tag{?}$$

$$η-\mathcal{U}(s)≥M (\bar{λ}(s) - 1),\quad ∀s∈S \tag{?}$$

$$\bar{ρ}≤\bar{λ}(s),\quad ∀s∈S \tag{?}$$

$$π(s) - (1 - λ(s)) ≤ ρ(s) ≤ λ(s),\quad ∀s∈S \tag{?}$$

$$ρ(s) ≤ \bar{ρ}(s) ≤ π(s),\quad ∀s∈S \tag{?}$$

$$∑_{s∈S}\bar{ρ}(s) = α \tag{?}$$

$$\bar{λ}(s), λ(s)∈\{0, 1\},\quad ∀s∈S \tag{?}$$

$$\bar{ρ}(s),ρ(s)∈[0, 1],\quad ∀s∈S \tag{?}$$

$$η∈[c^∘, c^∗] \tag{?}$$

Solution

$$\operatorname{VaR}_α(Z)=η^∗ \tag{?}$$

$$\operatorname{CVaR}_α(Z)=\frac{1}{α}∑_{s∈S}\bar{ρ}(s) \mathcal{U}(s)\tag{?}$$


## Expected Value and Value-at-Risk
We can formulate

$$w \operatorname{E}(Z) + (1-w) \operatorname{CVaR}_α(Z) \tag{?}$$

where $w∈(0, 1)$ is the **trade-off** between maximization of


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)
