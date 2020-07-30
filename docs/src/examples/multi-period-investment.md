# Multi-period Investment
## Description
[^1], section 4.2

> For instance, assume that the first-stage decisions specify which technology development projects will be started to generate patent-based intellectual property (P) for a platform. This intellectual property contributes subject to some uncertainties to the technical competitiveness (T) of the platform. In the second stage, it is possible to carry out application (A) development projects which, when completed, yield cash flows that depend on the market share of the platform. This market share (M) depends on the competitiveness of the platform and the number of developed applications. The aim is to maximize the cash flows from application projects less the cost of technology and application development projects.


## Formulation
### Projects
![](figures/multi-period-investment.svg)

The influence diagram of an individual multi-period investment project.

There are $i∈\{1,...,n_T\}$ technology development projects and $k∈\{1,...,n_A\}$ application development projects.

Decision states to develop patents

$$d_i^P∈D_i^P=\{[q_1^P, q_2^P], [q_2^P, q_3^P], ..., [q_{|D^P|}^P, q_{|D^P|+1}^P]\}$$

Chance states of technical competitiveness $c_i^T∈C_i^T$

Decision states to develop applications

$$d_k^A∈D^A=\{[q_1^A, q_2^A], [q_2^A, q_3^A], ..., [q_{|D^A|}^A, q_{|D^A|+1}^A]\}$$

Chance states of market size $c_k^M∈C_k^M$

Probability $ℙ(c_i^T∣d_i^P)∈[0,1]$

Probability $ℙ(c_k^M∣d_k^A,c_{n_T}^T,...,c_{1}^T)∈[0,1]$


### Portfolio Selection
Technology project $i$ costs $r_i^T∈ℝ^+$ and generates $p_i^T∈ℕ$ patents.

Application project $k$ costs $r_k^A∈ℝ^+$ and generates $a_k^A∈ℕ$ applications. If completed, provides cash flow $Y(c_k^M)∈ℝ^+.$

Decision variables $x^T(i)∈\{0, 1\}$ indicate which technologies are selected.

Decision variables $x^A(k∣c_{n_T}^T,...,c_{1}^T)∈\{0, 1\}$ indicate which applications are selected.

Number of patents $x^T = ∑_i x^T(i) p_i^T$

Number of applications $x^A = ∑_k x^A(k∣c_{n_T}^T,...,c_{1}^T) a_k^A$

Constraints

$$x^T - y_i^P M ≤ q_i^P ≤ x^T + (1 - y_i^P) M,\quad ∀i$$

$$x^A - y_{k}^A M ≤ q_k^A ≤ x^A + (1 - y_{k}^A) M,\quad ∀ k$$

$$∑_i y_i^P=1$$

$$∑_k y_{k}^A=1$$

$$y_i^P∈\{0, 1\},\quad ∀i$$

$$y_{k}^A∈\{0, 1\},\quad ∀k$$

$$y_0^P=y_{0}^A=0$$

$$z(d_i^P)=y_i^P-y_{i-1}^P,\quad ∀i$$

$$z(d_k^A∣c_{n_T}^T,...,c_{1}^T)=y_{k}^A-y_{k-1}^A,\quad ∀k$$

Large constant $M$ (value?)

Path utility

$$\mathcal{U}(s) =
∑_k x^A(k∣c_{n_T}^T,...,c_{1}^T) (Y(c_k^M) - r_k^A) - ∑_i x^T(i) r_i^T$$


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)
