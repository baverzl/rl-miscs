# Double Q-learning

## Abstract

- Positive bias = Overestimation
- Q-learning uses the maximum Q(s, a) as an approximation for the max expected value.

## The single estimator
 - Uses the maximum of a set of estimators as an approximation.
 - It is a bias related to the Winner's Curse in auctions and it can be shown to follow from Jensen's inequality.
 - The error in approximation thus consists soley of the variance in the estimator and decreases when we obtain more samples.
 - Q-learning uses this method to approximate the value of the next state by maximizing over the estimated action values in that state.

## The double estimator
 - Uses two estimates for each variable and uncouples the selection of an estimator and its value. 
 - This can have a negative bias.
 - The double estimator underestimates because the probabilities P(j = a*) sum to one and therefore the approximation is a weighted estimate of unbiased expected values, which must be lower or equal to the maximum expected value.