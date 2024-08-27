# [Decision Model](@id decision-model)
## Introduction
**Decision Programming** aims to find an optimal decision strategy $Z$ among all decision strategies $ℤ$ by maximizing an objective function $f$ on the path distribution of an influence diagram

$$\underset{Z∈ℤ}{\text{maximize}}\quad f(\{(ℙ(X=𝐬∣Z), \mathcal{U}(𝐬)) ∣ 𝐬∈𝐒\}). \tag{1}$$

**Decision model** refers to the mixed-integer linear programming formulation of this optimization problem. This page explains how to express decision strategies, compatible paths, path utilities and the objective of the model as a mixed-integer linear program. We present two standard objective functions, including expected value and conditional value-at-risk. The original decision model formulation was described in [^1], sections 3 and 5. We base the decision model on an improved formulation described in [^2] section 3.3. We recommend reading the references for motivation, details, and proofs of the formulation.


## Decision Variables
**Decision variables** $z(s_j∣𝐬_{I(j)})$ are equivalent to local decision strategies such that $Z_j(𝐬_{I(j)})=s_j$ if and only if $z(s_j∣𝐬_{I(j)})=1$ and $z(s_{j}^′∣𝐬_{I(j)})=0$ for all $s_{j}^′∈S_j∖s_j.$ Constraint $(2)$ defines the decisions to be binary variables and the constraint $(3)$ states that only one decision alternative $s_{j}$ can be chosen for each information set $s_{I(j)}$.

$$z(s_j∣𝐬_{I(j)}) ∈ \{0,1\},\quad ∀j∈D, s_j∈S_j, 𝐬_{I(j)}∈𝐒_{I(j)} \tag{2}$$

$$∑_{s_j∈S_j} z(s_j∣𝐬_{I(j)})=1,\quad ∀j∈D, 𝐬_{I(j)}∈𝐒_{I(j)} \tag{3}$$


## Path Compatibility Variables
**Path compatibility variables** $x(𝐬)$ are indicator variables for whether path $𝐬$ is compatible with decision strategy $Z$ defined by the decision variables $z$. These are continous variables but only assume binary values, so that the compatible paths $𝐬 ∈ 𝐒(Z)$ take values $x(𝐬) = 1$ and other paths $𝐬 ∈ 𝐒 \setminus 𝐒(Z)$ take values $x(𝐬) = 0$. Constraint $(4)$ defines the lower and upper bounds for the variables.

$$0≤x(𝐬)≤1,\quad ∀𝐬∈𝐒 \tag{4}$$

Constraint $(5)$ ensures that only the variables associated with locally compatible paths $𝐬 \in 𝐒_{s_j | 𝐬_{I(j)} }$ of the decision strategy can take value $x(𝐬) = 1$. The effective locally compatible paths are denoted with $| 𝐒^*_{s_j | 𝐬_{I(j)}}|$. The upper bound of the constraint uses the minimum of the *feasible paths* upper bound and the *theoretical* upper bound. The motivation of the feasible paths upper bound is below. For proofs and motivation on the theoretical upper bound see reference [^2].

$$∑_{𝐬 \in 𝐒^*_{s_j | 𝐬_{I(j)}} } x(𝐬) \leq \min ( \ | 𝐒^*_{s_j | 𝐬_{I(j)}}|, \ \frac{| 𝐒_{s_j | 𝐬_{I(j)}}| }{\displaystyle  \prod_{d \in D \setminus \{j, I(j)\}} |𝐒_d|} \ ) \ z(s_j∣𝐬_{I(j)}),\quad \forall j \in D, s_j \in S_j, 𝐬_{I(j)} \in 𝐒_{I(j)} \tag{5}$$

Constraint $(6)$ is called the probability cut constraint and it defines that the sum of the path probabilities of the compatible paths must equal one.

$$∑_{𝐬∈𝐒}x(𝐬) p(𝐬) = 1 \tag{6}$$

### Feasible paths upper bound
The *feasible paths upper bound* for the path compatibility variables is

$$∑_{𝐬 \in 𝐒^*_{s_j | 𝐬_{I(j)}} } x(𝐬) \leq  | 𝐒^*_{s_j | 𝐬_{I(j)}}| \ z(s_j∣𝐬_{I(j)}),\quad \forall j \in D, s_j \in S_j, 𝐬_{I(j)} \in 𝐒_{I(j)}$$

where $𝐒^*_{s_j | s_{I(j)}}$ is the set of effective locally compatible paths. This upper bound is motivated by the implementation of the framework in which path compatibility variables $x(𝐬)$ are only generated for effective paths $𝐬 \in 𝐒^∗$. The ineffective paths are not generated because they do not influence the objective function and having less variables reduces the size of the model.

Therefore, if the model has ineffective paths $𝐬 \in 𝐒^′$, then the number of effective paths is less than the number of all paths.

$$|𝐒^*| < |𝐒|$$

Therefore,

$$|𝐒^*_{s_j | 𝐬_{I(j)}} | < | 𝐒_{s_j | 𝐬_{I(j)}}| .$$

The feasible paths upper bound is used in conjunction with the *theoretical upper bound* as follows.

$$∑_{𝐬 \in 𝐒^*_{s_j | 𝐬_{I(j)}} } x(𝐬) \leq \min ( \ | 𝐒^*_{s_j | 𝐬_{I(j)}}|, \ \frac{| 𝐒_{s_j | 𝐬_{I(j)}}| }{\displaystyle  \prod_{d \in D \setminus \{j, I(j)\}} |𝐒_d|} \ ) \ z(s_j∣𝐬_{I(j)}),\quad \forall j \in D, s_j \in S_j, 𝐬_{I(j)} \in 𝐒_{I(j)}$$

The motivation for using the minimum of these bounds is that it depends on the problem structure which one is tighter. The feasible paths upper bound may be tighter if the set of ineffective paths is large compared to the number of all paths.




## Lazy Probability Cut
Constraint $(6)$ is a complicating constraint involving all path compatibility variables $x(s)$ and thus adding it directly to the model may slow down the overall solution process. It may be beneficial to instead add it as a *lazy constraint*. In the solver, a lazy constraint is only generated when an incumbent solution violates it. In some instances, this allows the MILP solver to prune nodes of the branch-and-bound tree more efficiently.

## Single Policy Update
To obtain (hopefully good) starting solutions, the SPU heuristic described in [^3] can be used. The heuristic finds a locally optimal strategy in the sense that the strategy cannot be improved by changing any single local strategy. With large problems, the heuristic can quickly provide a solution that would otherwise take very long to obtain.


## Expected Value
The **expected value** objective is defined using the path compatibility variables $x(𝐬)$ and their associated path probabilities $p(𝐬)$ and path utilities $\mathcal{U}(𝐬)$.

$$\operatorname{E}(Z) = ∑_{𝐬∈𝐒} x(𝐬) \ p(𝐬) \ \mathcal{U}(𝐬). \tag{7}$$

## Positive and Negative Path Utilities
We can omit the probability cut defined in constraint $(6)$ from the model if we are maximising expected value of utility and use a **positive path utility** function $\mathcal{U}^+$. Similarly, we can use a **negative path utility** function $\mathcal{U}^-$ when minimizing expected value. These functions are affine transformations of the path utility function $\mathcal{U}$ which translate all utility values to positive/negative values. As an example of a positive path utility function, we can subtract the minimum of the original utility function and then add one as follows.

$$\mathcal{U}^+(𝐬) = \mathcal{U}(𝐬) - \min_{𝐬∈𝐒} \mathcal{U}(𝐬) + 1. \tag{8}$$

$$\mathcal{U}^-(𝐬) = \mathcal{U}(𝐬) - \max_{𝐬∈𝐒} \mathcal{U}(𝐬) - 1. \tag{9}$$

## Conditional Value-at-Risk
The section [Measuring Risk](@ref) explains and visualizes the relationships between the formulation of expected value, value-at-risk and conditional value-at-risk for discrete probability distribution.

In this section, CVaR models are defined for both path-based and RJT models.

### Path-based model

Given decision strategy $Z,$ we define the cumulative distribution of compatible paths' probabilities as

$$F_Z(t) = ∑_{𝐬∈𝐒∣\mathcal{U}(𝐬)≤t} x(𝐬) p(𝐬).$$

Given a **probability level** $α∈(0, 1],$ we define the **value-at-risk** as

$$\operatorname{VaR}_α(Z)=u_α=\sup \{\mathcal{U}(𝐬)∣𝐬∈𝐒, F_Z(\mathcal{U}(𝐬))<α\}.$$

Then, we have the paths that have path utility less than and equal to the value-at-risk as

$$𝐒_{α}^{<}=\{𝐬∈𝐒∣\mathcal{U}(𝐬)<u_α\},$$

$$𝐒_{α}^{=}=\{𝐬∈𝐒∣\mathcal{U}(𝐬)=u_α\}.$$

We define **conditional value-at-risk** as

$$\operatorname{CVaR}_α(Z)=\frac{1}{α}\left(∑_{𝐬∈𝐒_α^{<}} x(𝐬) \ p(𝐬) \ \mathcal{U}(𝐬) + \left(α - ∑_{𝐬'∈𝐒_α^{<}} x(𝐬') \ p(𝐬') \right) u_α \right).$$

We can form the conditional value-at-risk as an optimization problem. We have the following pre-computed parameters.

Lower and upper bound of the value-at-risk

$$\operatorname{VaR}_0(Z)=u^-=\min\{\mathcal{U}(𝐬)∣𝐬∈𝐒\}, \tag{11}$$

$$\operatorname{VaR}_1(Z)=u^+=\max\{\mathcal{U}(𝐬)∣𝐬∈𝐒\}. \tag{12}$$

A "large number", specifically the largest difference between path utilities

$$M=u^+-u^-. \tag{13}$$

A "small number", specifically half of the smallest positive difference between path utilities

$$ϵ=\frac{1}{2} \min\{|\mathcal{U}(𝐬)-\mathcal{U}(𝐬^′)| \mid |\mathcal{U}(𝐬)-\mathcal{U}(𝐬^′)| > 0, 𝐬, 𝐬^′∈𝐒\}. \tag{14}$$

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

### RJT model

CVaR formulation for the RJT model is close to that of path-based model. A diagram can have only a single value node, when using RJT-based CVaR. Trying to call the RJT-based CVaR function using a diagram with more than one value node results in an error.

We denote the possible utility values with $u ∈ U$ and suppose we can define the probability $p(u)$ of attaining a given utility value. In the presence of a single value node, we define $p(u) = ∑_{s_{C_v}∈ \text{\{} S_{C_v} \vert U(s_{C_v})=u \text{\}} }µ(s_{C_v})$. We can then pose the constraints

$$η-u≤M λ(u),\quad ∀u∈U \tag{26}$$

$$η-u≥(M+ϵ) λ(u) - M,\quad ∀u∈U \tag{27}$$

$$η-u≤(M+ϵ) \bar{λ}(u) - ϵ,\quad ∀u∈U \tag{28}$$

$$η-u≥M (\bar{λ}(u) - 1),\quad ∀u∈U \tag{29}$$

$$\bar{ρ}(u) ≤ \bar{λ}(u),\quad ∀u∈U \tag{30}$$

$$p(u) - (1 - λ(u)) ≤ ρ(u) ≤ λ(u),\quad ∀u∈U \tag{31}$$

$$ρ(u) ≤ \bar{ρ}(u) ≤ p(u),\quad ∀u∈U \tag{32}$$

$$∑_{u∈U}\bar{ρ}(u) = α \tag{33}$$

$$\bar{λ}(u), λ(u)∈\{0, 1\},\quad ∀u∈U \tag{34}$$

$$\bar{ρ}(u),ρ(u)∈[0, 1],\quad ∀u∈U \tag{35}$$

$$η∈\mathbb{R} \tag{36}$$

where where α is the probability level in VaR<sub>α</sub>.

$CVaR_α$ can be obtained as $1/α ∑_{u∈U} \bar{ρ}(u)u$.

More details, including explanations of variables and constraints, can be found from Herrala et al. (2024)[^4].

## Convex Combination
We can combine expected value and conditional value-at-risk using a convex combination at a fixed probability level $α∈(0, 1]$ as follows

$$w \operatorname{E}(Z) + (1-w) \operatorname{CVaR}_α(Z), \tag{37}$$

where the parameter $w∈[0, 1]$ expresses the decision maker's **risk tolerance**.


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2022). Decision programming for mixed-integer multi-stage optimization under uncertainty. European Journal of Operational Research, 299(2), 550-565.

[^2]: Hölsä, O. (2020). Decision Programming Framework for Evaluating Testing Costs of Disease-Prone Pigs. Retrieved from [http://urn.fi/URN:NBN:fi:aalto-202009295618](http://urn.fi/URN:NBN:fi:aalto-202009295618)

[^3]: Hankimaa, H., Herrala, O., Oliveira, F., Tollander de Balsch, J. (2023). DecisionProgramming.jl -- A framework for modelling decision problems using mathematical programming. Retrieved from [https://arxiv.org/abs/2307.13299](https://arxiv.org/abs/2307.13299)

[^4]: Herrala, O., Terho, T., Oliveira, F., 2024. Risk-averse decision strategies for influence diagrams using rooted junction trees. Retrieved from [https://arxiv.org/abs/2401.03734]