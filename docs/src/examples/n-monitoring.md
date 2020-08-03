# N-Monitoring
## Description
The $N$-monitoring problem is described in [^1], sections 4.1 and 6.1.


## Formulation
![](figures/2-monitoring.svg)

The $2$-monitoring problem.

![](figures/n-monitoring.svg)

The incluence diagram of generalized $N$-monitoring problem where $N≥1$ and indices $k=1,...,N.$ The nodes are associated with states as follows. **Load state** $L=\{high, low\}$ denotes the load, **report states** $R_k=\{high, low\}$ report the load state to the **action states** $A_k=\{yes, no\}$ which decide whether to fortificate the structure, **failure state** $F=\{failure, success\}.$ Finally, the utility at target $T$ depends on the whether the structure fails in $F$ and the fortification costs.

We draw the magnitude and cost of fortification $c_k∼U(0,1)$ from a uniform distribution. Fortification is defined

$$f(A_k) =
\begin{cases}
c_k, & A_k=yes \\
0, & A_k=no
\end{cases}$$

The probability that the load is high is drawn from uniform distribution $x∼U(0,1)$.

$$ℙ(L=high)=x$$

The probabilities of the report states correspond to the load state. We draw the values $x∼U(0,1)$ and $y∼U(0,1)$ from uniform distribution.

$$ℙ(R_k=high∣L=high)=\max\{x,1-x\}$$

$$ℙ(R_k=low∣L=low)=\max\{y,1-y\}$$

The probabilities of failure which are decresead by fortifications. We draw the values $z∼U(0,1)$ and $w∼U(0,1)$ from uniform distribution.

$$ℙ(F=failure∣A_N,...,A_1,L=high)=\frac{\max\{z,1-z\}}{\exp(∑_{k=1,...,N} f(A_k))}$$

$$ℙ(F=failure∣A_N,...,A_1,L=low)=\frac{\min\{w,1-w\}}{\exp(∑_{k=1,...,N} f(A_k))}$$

Utility from consequences at target $T$ from failure state $F$

$$U(Y(F)) =
\begin{cases}
0, & F = failure \\
100, & F = success
\end{cases}$$

Utility from consequences at target $T$ from action states $A_k$

$$U(Y(A_k))=
\begin{cases}
-c_k, & A_k=yes \\
0, & A_k=no
\end{cases}$$

Total utility at target $T$

$$U(Y(F,A_N,...,A_1))=U(Y(F))+∑_{k=1,...,N} U(Y(A_k)).$$


## References
[^1]: Salo, A., Andelmin, J., & Oliveira, F. (2019). Decision Programming for Multi-Stage Optimization under Uncertainty, 1–35. Retrieved from [http://arxiv.org/abs/1910.09196](http://arxiv.org/abs/1910.09196)
