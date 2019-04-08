# A3C: Asynchronous Methods for Deep Reinforcement Learning

## Abstract

- Propose a simple and lightweight framework for deep RL that uses asynchronous gradient descent for optimization of deep neural network.
- Asynchronous variants of four standard reinforcement learning algorithms.
- Parallel actor-learners have a stabilizing effect.
- An asynchronous variant of actor-critic surpasses the curent state-of-the-art on the Atari domain while training for half the time on a single multi-core CPU instead of a GPU.
- Shows that async. actor-critic succeeds on problems that require continuous motor control as well as a task of navigating random 3D mazes using a visual input.

## Previous methods
 - Training a RL agent has been difficult due to following two reasons:
   1. The sequence of observed data encountered by an online RL agent is non-stationary.
   2. Online RL updates are strongly correlated.
 - Experience Replay Memory: Resolved the above issue. However, this technique is only confined to "off-policy reinforcement learning algorithms". Also, it uses more memory and computation per real interaction; 
 
 ## Why Async:
   1. Can even be applied to on-policy methods.
   2. Can be trained with standard multi-core CPU achieving better results, in far less time than previous GPU-based algorithms.
   3. Showed descent results on a variety of continuous motor control tasks, and both 2D and 3D games, discrete and continuous action spaces, as well as its ability to train feedforward and recurrent agents makes it the most general and successful reinforcement learning agent.