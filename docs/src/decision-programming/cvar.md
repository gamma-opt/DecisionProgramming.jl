# [Conditional value-at-risk](@id cvar)

The section [Measuring Risk](@ref) explains and visualizes the relationships between the formulation of expected value, value-at-risk and conditional value-at-risk for discrete probability distribution.

In this section, CVaR models are defined for both path-based and RJT models.

## Path-based model

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

## RJT model

!!! warning 
    A diagram can have only a single value node when using RJT-based CVaR. Trying to call the RJT-based CVaR function using a diagram with more than one value node results in an error.

CVaR formulation for the RJT model is close to that of path-based model. We denote the possible utility values with $u ∈ U$ and suppose we can define the probability $p(u)$ of attaining a given utility value. In the presence of a single value node, we define $p(u) = ∑_{s_{C_v}∈ \text{\{} S_{C_v} \vert U(s_{C_v})=u \text{\}} }µ(s_{C_v})$. We can then pose the constraints

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

where $α$ is the probability level in $CVaR_α$.

Finally, $CVaR_α$ can be obtained as $1/α ∑_{u∈U} \bar{ρ}(u)u$.

More details, including explanations of variables and constraints, can be found from Herrala et al. (2024)[^1].

## Convex Combination
We can combine expected value and conditional value-at-risk using a convex combination at a fixed probability level $α∈(0, 1]$ as follows

$$w \operatorname{E}(Z) + (1-w) \operatorname{CVaR}_α(Z), \tag{37}$$

where the parameter $w∈[0, 1]$ expresses the decision maker's **risk tolerance**.


## References
[^1]: Herrala, O., Terho, T., Oliveira, F., 2024. Risk-averse decision strategies for influence diagrams using rooted junction trees. Retrieved from [https://arxiv.org/abs/2401.03734]
