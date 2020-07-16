# Multi-period Investment
## Description
[^1], section 4.2

> For instance, assume that the first-stage decisions specify which technology development projects will be started to generate patent-based intellectual property (P) for a platform. This intellectual property contributes subject to some uncertainties to the technical competitiveness (T) of the platform. In the second stage, it is possible to carry out application (A) development projects which, when completed, yield cash flows that depend on the market share of the platform. This market share (M) depends on the competitiveness of the platform and the number of developed applications. The aim is to maximize the cash flows from application projects less the cost of technology and application development projects.


## Formulation
### Individual Projects
![](figures/multi-period-investment.svg)

The influence diagram of an individual multi-period investment project.

Decision states to develop patents

$$d^P∈D^P=\{[q_1^P, q_2^P], [q_2^P, q_3^P], ..., [q_{|D^P|}^P, q_{|D^P|+1}^P]\}$$

Decision states to develop applications

$$d^A∈D^A=\{[q_1^A, q_2^A], [q_2^A, q_3^A], ..., [q_{|D^A|}^A, q_{|D^A|+1}^A]\}$$

Chance states of technical competitiveness $c^T∈C^T$

Chance states of market size $c^M∈C^M$

Probability $ℙ(c^T∣d^P)∈[0,1]$

Probability $ℙ(c^M∣c^T,d^A)∈[0,1]$


### Portfolio Selection
Technology project $i$ costs $r_i^T∈ℝ^+$ and generates $p_i^T∈ℕ$ patents.

Application project $k$ costs $r_k^A∈ℝ^+$ and generates $a_k^A∈ℕ$ applications. If completed, provides cash flow $Y(k∣c^M)∈ℝ^+.$

Decision variables $x^T(i)∈\{0, 1\}$ indicate which technologies are selected.

Decision variables $x^A(k∣c_j^T)∈\{0, 1\}$ indicate which applications are selected.

Number of patents $x^T = ∑_i x^T(i) p_i^T$

Number of applications $x_j^A = ∑_k x^A(k∣c_j^T) a_k^A$

Constraints

$$x^T - y_i^P M ≤ q_i^P ≤ x^T + (1 - y_i^P) M,\quad ∀i$$

$$x_j^A - y_{j,k}^A M ≤ q_k^A ≤ x_j^A + (1 - y_{j,k}^A) M,\quad ∀j, k$$

$$∑_i y_i^P=1$$

$$∑_k y_{j,k}^A=1,\quad ∀j$$

$$y_i^P∈\{0, 1\},\quad ∀i$$

$$y_{j,k}^A∈\{0, 1\},\quad ∀j,k$$

$$y_0^P=y_{j,0}^A=0$$

$$z(d_i^P)=y_i^P-y_{i-1}^P,\quad ∀i$$

$$z(d_k^A∣c_j^T)=y_{j,k}^A-y_{j,k-1}^A,\quad ∀j,k$$

Large constant $M$ (value?)

Path utility

$$\mathcal{U}(s) =
∑_k x^A(k∣c^T) (Y(k∣c^M) - r_k^A) - ∑_i x^T(i) r_i^T$$


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)
