# CHD preventative care allocation
## Description
 The goal in this optimisation problem is determine an optimal decision strategy for the testing and treatment decisions involved in providing preventative care for coronary heart disease (CHD). The optimality is evaluated from the perspective of the national health care system and is measured through net monetary benefit (NMB). The tests available in this model are the traditional risk score (TRS) and the genetic risk score (GRS) and the form of preventative care is statin treatment. The CHD preventative care allocation problem as it is described in [^1] is seen below.

> The problem setting is such that the patient is assumed to have a prior risk estimate. A risk estimate is a prediction of the patient’s chance of having a CHD event in the next ten years. The risk estimates are grouped into risk levels, which range from 0% to 100%. The first testing decision is made based on the prior risk estimate. The first testing decision entails deciding whether TRS or GRS should be performed or if no testing is needed. If a test is conducted, the risk estimate is updated and based on the new information, the second testing decision is made. The second testing decision entails deciding whether further testing should be conducted or not. The second testing decision is constrained so that the same test which was conducted in the first stage cannot be repeated. If a second test is conducted, the risk estimate is updated again. The treatment decision – dictating whether the patient receives statin therapy or not – is made based on the resulting risk estimate of this testing process. Note that if no tests are conducted, the treatment decision is made based on the prior risk estimate.

In this example, we will showcase the solution to the subproblem that solves for the optimal decision strategy for a single prior risk level. The chosen risk level in this example is 12%. The solution to the main problem is found in [^1].

## Influence Diagram




## Decision Model



## Analyzing Results



## References
[^1]: Hankimaa H. (2021). Optimising the use of genetic testing in prevention of CHD using Decision Programming. http://urn.fi/URN:NBN:fi:aalto-202103302644