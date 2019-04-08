# DDQN: Deep Reinforcement Learning with Double Q-learning

## DQN is susceptible to overestimations

 - DQN sometimes substantially overestimates the values of the actions which lead to sub-optimal policies.
 - A maximization step over estimated action values tends to prefer overestimated to underestimated values.
 - The max operator in standard Q-learning and DQN uses the same values both to select and evaluate an action. This makes it more likely to select overestimated values, resulting in overoptimistic value estimates.
 - If the action values contain random errors uniformly distributed in an interval, their each target is overestimated.
 - Various reasons could exist to explain why overestimations happen: environmental noise, function approximation, non-stationarity. This is important, because in practice any method will incur some inaccuracies during learning, simple due to the fact that "the true values are initially unknown".

## Double DQN
- Decouple/Untangle the selection from the evaluation.
- Avoid overestimations from Q-learning.
- Two value functions are learned by assigning each experience randomly to update one of the two value functions. For each update, one set of weights is used to determine the greedy policy and the other to determine its value. 