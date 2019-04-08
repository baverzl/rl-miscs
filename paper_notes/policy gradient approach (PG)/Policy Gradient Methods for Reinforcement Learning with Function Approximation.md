# Sutton PG

## Main contributions:
 - Showed that their version of policy iteration is convergent to a locally optimal policy.
 - Approximating a value function -> Approximating the policy, updated according to the gradient of expected reward w.r.t the policy parameters.
 - Prove that an unbiased estimate of the policy gradient can be obtained from experience using an approximate value function satisfying certain properties.

 ## The author points out some limitations of the value-function approach
  - First, it is oriented toward finding deterministic policies.
  - Second, an arbitrary small change in the estimated value of an action can cause it to be, or not be, selected. Such discontinuous changes have been identified as a key obstacle to establishing convergence assuracnes for algorithms following the value-function approach.

## Policy Gradient Theorem

## Policy Gradient with Approximation

## Convergence of Policy Iteration with Function Approximation