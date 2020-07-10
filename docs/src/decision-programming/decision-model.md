# Decision Model
## Formulation
The model is based on [^1], section 3. We highly recommend to read them for motivation, details, and proofs of the formulation explained here.

The mixed-integer linear program maximizes the expected utility (1) over all decision strategies as follows.

$$\underset{Z∈ℤ}{\text{maximize}}\quad
∑_{s∈S} π(s) \mathcal{U}(s) \tag{1}$$

Subject to

$$z(s_j∣s_{I(j)}) ∈ \{0,1\},\quad ∀j∈D, s_j∈S_j, s_{I(j)}∈S_{I(j)} \tag{2}$$

$$∑_{s_j∈S_j} z(s_j∣s_{I(j)})=1,\quad ∀j∈D, s_{I(j)}∈S_{I(j)} \tag{3}$$

$$0≤π(s)≤p(s),\quad ∀s∈S \tag{4}$$

$$π(s) ≤ z(s_j∣s_{I(j)}),\quad ∀j∈D, s∈S \tag{5}$$

$$π(s) ≥ p(s) + ∑_{j∈D} z(s_j∣s_{I(j)}) - |D|,\quad ∀s∈S \tag{6}$$

**Decision variables** $z$ are binary variables (2) that model different decision strategies. The condition (3) limits decisions $s_j$ to one per information path $s_{I(j)}.$ Decision strategy $Z_j(s_I(j))=s_j$ is equivalent to $z(s_j∣s_{I(j)})=1.$

We denote the probability distribution of paths using $π.$ The **path probability** $π(s)$ is between zero and the upper bound of the path probability (4). The path probability is zero on paths where at least one decision variable is zero (5) and equal to the upper bound on paths if all decision variables on the path are one (6).

We can omit the constraint (6) from the model if we use a positive utility function $U^+$ which is an affine transformation of utility function $U.$ As an example, we can normalize and add one to the original utility function.

$$U^+(c) = \frac{U(c) - \min_{c∈ℂ}U(c)}{\max_{c∈ℂ}U(c) - \min_{c∈ℂ}U(c)} + 1.$$

There are also alternative objectives and ways to model risk, which are discussed later.


## Lazy Cuts
Probability sum cut

$$∑_{s∈S}π(s)=1 \tag{7}$$

Number of active paths cut

$$∑_{s∈S} \frac{π(s)}{p(s)}=|S^+| \tag{8}$$


## Conditional Value-at-Risk
Conditional Value at Risk (CVaR), [^1], section 5.3


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)
