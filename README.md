# Deep Q-Learning Taxi Driver — Undergraduate Thesis (UNICAMP, 2021)

Reinforcement learning agent trained from scratch to navigate a simulated city and complete sequential passenger pickup-and-dropoff trips. This was my undergraduate thesis in Production Engineering at UNICAMP (Universidade Estadual de Campinas), 2021.

**Full thesis (PDF, in Portuguese — English abstract on page 6):** [link to be added]

> **Note on the code.** This is research-quality code from 2021, kept close to its original state and included as evidence of implementation rather than as a polished release. It was written and run in Google Colab. Things I would do differently today are listed at the bottom.

---

## What it does

The agent (a taxi) learns to operate in a 10×10 grid city built from scratch. The environment contains streets, sidewalks (collision = heavy penalty), tolls (a shortcut with a cost), a passenger, and a destination. The agent must learn to pick the passenger up first and then reach the destination, taking the shortest viable route — a sequential, multi-step objective with delayed reward.

The agent is trained with **Deep Q-Learning**: a convolutional neural network takes the rendered 10×10×3 image of the environment as input and outputs Q-values for 9 possible moves (stay, or step to any of the 8 neighbors). Training uses experience replay, a target network, and epsilon-greedy exploration.

**Final result:** 92% goal-completion rate with the trained network, versus 7% for a random-action baseline.

## Tech stack

Python, TensorFlow/Keras (CNN + DQN), OpenCV (environment rendering), NumPy, TensorBoard (training metrics). Built and run in Google Colab.

---

## The three versions

The thesis documents an iterative process. The agent did not converge to good behavior on the first attempt; each version was a response to a specific observed failure, with a hypothesis and a change to test it.

**v1 — initial**
First reward design and the hyperparameters inherited from the reference DQN implementation. The agent's average reward *decreased* over training and the max-reward plot showed overfitting: it learned only a few specific start positions and otherwise parked itself on a safe cell to avoid the sidewalk penalty.

**v2 — repetition penalties**
Added penalties for revisiting the same cell 3–5 times and 6+ times, to stop the agent from idling or oscillating between two cells. Results improved but the agent still overfit, going long stretches without positive reward.

**v3 — final** (`taxi_dqn_v3_final.ipynb`)
Hypothesis: the hyperparameters were causing overfitting on a simple environment. Changes: raised the discount from 0.99 to 0.995 (longer-horizon decisions), doubled batch size to 128, raised target-update frequency from 5 to 25, slowed epsilon decay, and **removed one convolutional layer** since the visual patterns are simple. Also restructured rewards so reaching the passenger is worth more than reaching the destination, encouraging the correct order. This version converged to a positive average reward and the 92% success rate above.

This notebook is set to evaluation mode (loads the saved trained model, epsilon = 0) — it's the configuration used to measure the final 92% vs 7% comparison.

---

## Files

- `taxi_dqn_v3_final.ipynb` — final version (v3), evaluation configuration

This is the cleaned, lightweight version of the final notebook (training outputs and embedded widgets removed; code unchanged). The earlier experiment notebooks (v1 and v2) are available on request.

---

## What I would do differently today

- Separate the environment, agent, and training loop into modules instead of one notebook.
- Use a standard RL framework (e.g. Gymnasium) and a maintained DQN implementation instead of hand-rolling everything.
- Proper experiment tracking and config files instead of editing constants by hand between runs.
- Unit tests on the reward function and the environment transitions.
- Evaluate generalization more rigorously (held-out start positions, multiple seeds) rather than stopping once behavior looked good.
