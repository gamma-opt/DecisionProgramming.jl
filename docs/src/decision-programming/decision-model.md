# [Decision Model](@id decision-model)
## Introduction
**Decision programming** aims to find an optimal decision strategy $Z$ from all decision strategies $ℤ$ by maximizing an objective function $f$ on the path distribution of an influence diagram

$$\underset{Z∈ℤ}{\text{maximize}}\quad f(\{(ℙ(X=𝐬∣Z), \mathcal{U}(𝐬)) ∣ 𝐬∈𝐒\}). \tag{1}$$

**Decision model** refers to the mixed-integer linear programming formulation of this optimization problem. This page explains how to express decision strategy, path probability, path utility, and the objective in the mixed-integer linear form.%% grammar? We also present standard objective functions, including expected value and risk measures.  The original decision model formulation was described in [^1], sections 3 and 5. We base the decision model on an improved formulation described in [^2] section 3.3. We recommend reading the references for motivation, details, and proofs of the formulation.


## Decision Variables
**Decision variables** $z(s_j∣𝐬_{I(j)})$ are equivalent to local decision strategies such that $Z_j(𝐬_I(j))=s_j$ if and only if $z(s_j∣𝐬_{I(j)})=1$ and $z(s_{j}^′∣𝐬_{I(j)})=0$ for all $s_{j}^′∈S_j∖s_j.$ Constraint $(2)$ defines the decisions to be binary variables and the constraint $(3)$ limits decisions to one per information path.

$$z(s_j∣𝐬_{I(j)}) ∈ \{0,1\},\quad ∀j∈D, s_j∈S_j, 𝐬_{I(j)}∈𝐒_{I(j)} \tag{2}$$

$$∑_{s_j∈S_j} z(s_j∣𝐬_{I(j)})=1,\quad ∀j∈D, 𝐬_{I(j)}∈𝐒_{I(j)} \tag{3}$$


## Path Compatibility Variables
**Path compatibility variables** $x(s)$ are indicator variables for whether the path is compatible with decision strategy $Z$ that is defined by the decision variables $z$. These are continous variables but only assume binary values $\{0, 1\}$, with the compatible paths $s ∈ S$ taking values $x(s) = 1$. Constraint $(4)$ defines the lower and upper bounds for the variables. 

Constraint $(5)$ ensures that only the variables associated with locally compatible paths $s \in S_{s_j | s_{I(j)} }$ of the decision strategy can take value $x(s) = 1$. The upperbound of the constraint uses the minimum of the *feasible paths* upperbound and the *theoretical* upperbound. For motivation on of the feasible paths upper bound see the [Computational Complexity](@ref computational-complexity) page. For proofs and motivation on the theoretical upperbound see reference [^2].

Constraint $(6)$ is called the probability cut constraint and it defines that the sum of the path probabilities of the compatible paths must equal one.

$$0≤x(s)≤1,\quad ∀s∈𝐒 \tag{4}$$

$$∑_{s \in S'_{s_j | s_{I(j)}} } x(s) \leq \min ( \ | S'_{s_j | s_{I(j)}}|, \ \frac{| S_{s_j | s_{I(j)}}| }{\displaystyle  \prod_{d \in D \setminus \{j, I(j)\}} |S_d|} \ ) \ z(s_j∣s_{I(j)}),\quad \forall j \in D, s_j \in S_j, s_{I(j)} \in S_{I(j)} \tag{5}$$

$$∑_{𝐬∈𝐒}x(s) p(s) = 1 \tag{6}$$



## Lazy Probability Cut
Constraint $(6)$ is a complicating constraint and thus adding it directly to the model may slow down the overall solution process. It may be beneficial to instead add it as a *lazy constraint*. In the solver, a lazy constraint is only generated when an incumbent solution violates it. In some instances, this allows the MILP solver to prune nodes of the branch-and-bound tree more efficiently. 


## Expected Value
The **expected value** objective is defined using the compatible paths $\{s \in S \mid x(s) = 1 \}$ and their path probabilities $p(s)$ and path utilities $\mathcal{U}(s)$. 

$$\operatorname{E}(Z) = ∑_{𝐬∈𝐒} x(𝐬) \ p(𝐬) \ \mathcal{U}(𝐬). \tag{7}$$

## Positive Path Utility
We can omit the probability cut defined in constraint $(6)$ from the model if we are maximising expected value of utility and use a **positive path utility** function $\mathcal{U}^+$. The positive path utility function $\mathcal{U}^+$ is an affine transformation of path utility function $\mathcal{U}$ which translates all utility values to positive values. As an example, we can subtract the minimum of the original utility function and then add one as follows.

$$\mathcal{U}^+(𝐬) = \mathcal{U}(𝐬) - \min_{𝐬∈𝐒} \mathcal{U}(𝐬) + 1. \tag{8}$$

## Negative Path Utility
We can omit the probability cut defined in constraint $(6)$ from the model if we are minimising expected value of utility and use a **negative path utility** function $\mathcal{U}^-$. This affine transformation of the path utility function $\mathcal{U}$ translates all utility values to negative values. As an example, we can subtract the maximum of the original utility function and then subtract one as follows.

$$\mathcal{U}^+(𝐬) = \mathcal{U}(𝐬) - \max_{𝐬∈𝐒} \mathcal{U}(𝐬) - 1. \tag{9}$$


## Conditional Value-at-Risk
The section [Measuring Risk](@ref) explains and visualizes the relationships between the formulation of expected value, value-at-risk and conditional value-at-risk for discrete probability distribution.

Given decision strategy $Z,$ we define the cumulative distribution of effective paths' probabilities as

$$F_Z(t) = ∑_{𝐬∈𝐒∣\mathcal{U}(𝐬)≤t} x(𝐬) p(𝐬).$$

Given a **probability level** $α∈(0, 1],$ we define the **value-at-risk** as

$$\operatorname{VaR}_α(Z)=u_α=\sup \{\mathcal{U}(𝐬)∣𝐬∈𝐒, F_Z(\mathcal{U}(𝐬))<α\}.$$

Then, we have the paths that have path utility less than and equal to the value-at-risk as

$$𝐒_{α}^{<}=\{𝐬∈𝐒∣\mathcal{U}(𝐬)<u_α\},$$

$$𝐒_{α}^{=}=\{𝐬∈𝐒∣\mathcal{U}(𝐬)=u_α\}.$$

We define **conditional value-at-risk** as

$$\operatorname{CVaR}_α(Z)=\frac{1}{α}\left(∑_{𝐬∈𝐒_α^{<}} x(𝐬) \ p(𝐬) \ \mathcal{U}(𝐬) + ∑_{𝐬∈𝐒_α^{=}} \left(α - ∑_{𝐬∈𝐒_α^{<}} x(𝐬) \ p(𝐬) \right) \mathcal{U}(𝐬) \right).$$

We can form the conditional value-at-risk as an optimization problem. We have the following pre-computed parameters.

Lower and upper bound of the value-at-risk

$$\operatorname{VaR}_0(Z)=u^-=\min\{\mathcal{U}(𝐬)∣𝐬∈𝐒\}, \tag{11}$$

$$\operatorname{VaR}_1(Z)=u^+=\max\{\mathcal{U}(𝐬)∣𝐬∈𝐒\}. \tag{12}$$

Largest difference between path utilities

$$M=u^+-u^-. \tag{13}$$

Half of the smallest positive difference between path utilities

$$ϵ=\frac{1}{2} \min\{|\mathcal{U}(𝐬)-\mathcal{U}(𝐬^′)| ∣ |\mathcal{U}(𝐬)-\mathcal{U}(𝐬^′)| > 0, 𝐬, 𝐬^′∈𝐒\}. \tag{14}$$

The objective is to minimize the variable $η$ whose optimal value is equal to the value-at-risk, that is, $\operatorname{VaR}_α(Z)=\min η.$

We define the constraints as follows:

$$η-\mathcal{U}(𝐬)≤M λ(𝐬),\quad ∀𝐬∈𝐒 \tag{14}$$

$$η-\mathcal{U}(𝐬)≥(M+ϵ) λ(𝐬) - M,\quad ∀𝐬∈𝐒 \tag{15}$$

$$η-\mathcal{U}(𝐬)≤(M+ϵ) \bar{λ}(𝐬) - ϵ,\quad ∀𝐬∈𝐒 \tag{16}$$

$$η-\mathcal{U}(𝐬)≥M (\bar{λ}(𝐬) - 1),\quad ∀𝐬∈𝐒 \tag{17}$$

$$\bar{ρ}(𝐬) ≤ \bar{λ}(𝐬),\quad ∀𝐬∈𝐒 \tag{18}$$

$$x(𝐬) \ p(𝐬) - (1 - λ(𝐬)) ≤ ρ(𝐬) ≤ λ(𝐬),\quad ∀𝐬∈𝐒 \tag{19}$$

$$ρ(𝐬) ≤ \bar{ρ}(𝐬) ≤ x(𝐬) \ p(𝐬),\quad ∀𝐬∈𝐒 \tag{20}$$

$$∑_{𝐬∈𝐒}\bar{ρ}(𝐬) = α \tag{21}$$

$$\bar{λ}(𝐬), λ(𝐬)∈\{0, 1\},\quad ∀𝐬∈𝐒 \tag{22}$$

$$\bar{ρ}(𝐬),ρ(𝐬)∈[0, 1],\quad ∀𝐬∈𝐒 \tag{23}$$

$$η∈[u^-, u^+] \tag{24}$$

We can express the conditional value-at-risk objective as

$$\operatorname{CVaR}_α(Z)=\frac{1}{α}∑_{𝐬∈𝐒}\bar{ρ}(𝐬) \mathcal{U}(𝐬)\tag{25}.$$

The values of conditional value-at-risk are limited to the interval between the lower bound of value-at-risk and the expected value

$$\operatorname{VaR}_0(Z)<\operatorname{CVaR}_α(Z)≤E(Z).$$


## Convex Combination
We can combine expected value and conditional value-at-risk using a convex combination at a fixed probability level $α∈(0, 1]$ as follows

$$w \operatorname{E}(Z) + (1-w) \operatorname{CVaR}_α(Z), \tag{26}$$

where the parameter $w∈[0, 1]$ expresses the decision maker's **risk tolerance**.


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)

[^2]: Hölsä, O. (2020). Decision Programming Framework for Evaluating Testing Costs of Disease-Prone Pigs. Retrieved from [http://urn.fi/URN:NBN:fi:aalto-202009295618](http://urn.fi/URN:NBN:fi:aalto-202009295618)
