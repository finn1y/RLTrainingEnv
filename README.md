# Reinforcement Learning (RL) Training Environment

A training environment for reinforcement learning algorithms using [open-ai gym](https://gym.openai.com/), as first suggested by Brockman et al in [[1]](#1)

## Included Algorithms

Both single agent and multi-agent reinforcement learning algorithms are included within the learning environment
Any single agent algorithm can be used as an independent multi-agent learning algorithm with the multi-agent environments

All algorithms using gradient descent on a neural network use the Adam optimiser, first proposed by Kingma et al in [[2]](#2). They also use the Huber loss function in place of the mean squared error where approriate, suggested by Huber in [[3]](#3).
These functions are implemented as part of the [tensorflow](https://www.tensorflow.org/overview) library. 

All algorithm implementations also include exploration rate (epsilon) and learning rate (alpha) decay where appropriate;
this can be removed by setting the relevant decay rate to 1.

#### [Q-Learning](rl_training_env/algorithms/qlearning.py)

Q-learning is implemented based on the algorithm described by Sutton and Barto in [[4]](#4).

#### [Deep Q-Network](rl_training_env/algorithms/dqn.py) (DQN)

Deep Q-network is implemeneted based on the algorithm described by Minh et al in [[5]](#5).
However it does not use CNNs as the environments used in this training are not array based 
(i.e. not an RGB array screen representation) 

#### [Deep Recurrent Q-Network](rl_training_env/algorithms/dqn.py) (DRQN)

Deep Recurrent Q-Network is implemented based on the alterations to DQN as suggested by Hausknecht and Stone in [[6]](#6). 
Similarly to DQN CNNs are not used. (Note this algorithm is implemented as DQN with a DRQN flag to change the first neural net layer)

#### [Policy Gradient](rl_training_env/algorithms/policy_grad.py) (PG)

Policy Gradient is implemented using the policy gradient equation as derived by Sutton in [[7]](#7) and its counterpart in [[8]](#8) by Silver et al.
The algorithm is similar to the REINFORCE algorithm as suggested by Williams in [[9]](#9)

#### [Advantage Actor Critic](rl_training_env/algorithms/actor_critic.py) (A2C)

Advantage Actor Critic is implemented based on one of the actor critic variations suggested by Bhatnagar et al in [[10]](#10).

#### [Deep Deterministic Policy Gradient](rl_training_env/algorithms/ddpg.py) (DDPG)

Deep Deterministic Policy Gradient is implemented based on the algorithm as suggested in [[11]](#11) by Lillicrap et al.

#### [Multi-Agent Actor Critic](rl_training_env/algorithms/ma_actor_critic.py) (MA Actor Critic)

Multi-Agent Actor Critic is implemented based on a the algorithm described by Lowe et al in [[12]](#12). 
As the multi-agent environments are cooperative there is communication of agent policy so no policy inference is required 
nor are policy ensembles.

#### [Distributed Deep Recurrent Q-Network](rl_training_env/algorithms/ddrqn.py) (DDRQN)

Distributed Deep Recurrent Q-Network is implemented based on the changes to Deep Q-Networks suggested by Foerster et al in [[13]](#13) 
for multi-agent environments. Due to the nature of this simulation instead of direct inter-agent weight sharing (i.e. directly tying all network weights) agents share weights via communication each updating the their network parameters in turn and then communicating the updated weights to the next agent until all agents have performed their updates. 

### Algoithm I/O

Algorithm   | State space       | Action space
------------|-------------------|--------------
Q-Learning  |   Discrete        | Discrete
DQN         |Continuous/Discrete| Discrete 
DRQN        |Continuous/Discrete| Discrete
PG          |Continuous/Discrete| Discrete
A2C         |Continuous/Discrete| Discrete
DDPG        |Continuous/Discrete| Continuous
MAAC        |Continuous/Discrete| Discrete
DDRQN       |Continuous/Discrete| Discrete

## Included Environments

Several of openai gyms' environments are included as single agent environments are included, as well as some custom 
environments which have both single agent and multi-agent variations

#### [Maze](https://github.com/MattChanTK/gym-maze)

A simple openai gym maze environment written by GitHub user 'MattChanTK' in [[14]](#14). A [patch](patches/gym_maze_multi_agent.patch)
which adds multi-agent functionality to the environment has been included in this repository for this environment.

#### [Robot Maze](https://github.com/finn1y/gym-robot-maze)

An openai gym maze envrionment written by GitHub user 'finn1y' in [[15]](#15).  

#### [Cart Pole](https://gym.openai.com/envs/CartPole-v1/)

An openai gym environment shipped with openai gym under classic control environments

#### [Acrobot](https://gym.openai.com/envs/Acrobot-v1/)

An openai gym environment shipped with openai gym under classic control environments

#### [Mountain Car](https://gym.openai.com/envs/MountainCar-v0/)

An openai gym environment shipped with openai gym under classic control environments

#### [Mountain Car Continuous](https://gym.openai.com/envs/MountainCarContinuous-v0/)

An openai gym environment shipped with openai gym under classic control environments

#### [Pendulum](https://gym.openai.com/envs/Pendulum-v0/)

An openai gym environment shipped with openai gym under classic control environments

### Environment I/O

Environment | State space       | Action space
------------|-------------------|--------------
Maze        |   Discrete        | Discrete
Robot Maze  |   Continuous      | Discrete 
Cart Pole   |   Continuous      | Discrete
Acrobot     |   Continuous      | Discrete
Mountain Car|   Continuous      | Discrete
Mountain Car Continuous|Continuous| Continuous
Pendulum    |   Continuous      | Continuous

## Install

1. Clone the repo
```
git clone https://github.com/finn1y/RLTraingingEnv
```
2. Install python dependencies in repo
```
cd RLTrainingEnv
pip install -r requirements.txt
```
3. Apply dependency patches, described in [patches](patches/README.md)
4. Enjoy training some RL agents!

## References

<a id="1">[1]</a>
G. Brockman, V. Cheung, L. Pettersson et al, "OpenAI Gym", *arXiv:1606.01540v1 [cs.LG]*, 2016. Available: [link](https://arxiv.org/abs/1606.01540) [Accessed 2 Feb 2022]

<a id="2">[2]</a> 
D. P. Kingma and J. L. Ba, "Adam: A Method for Stochastic Optimisation", *arXiv:1412.6980 [cs.LG]*, 2015. Available: [link](https://arxiv.org/abs/1412.6980) [Accessed 9 Feb 2022]

<a id="3">[3]</a>
P. J. Huber, ???Robust Estimation of a Location Parameter???, The Annals of Mathematical Statistics 35(1), 1964, pp. 73-101.

<a id="4">[4]</a>
R.S. Sutton and A.G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. The MIT Press, 2018.

<a id="5">[5]</a>
V. Mnih, K. Kavukcuoglu, D. Silver et al, ???Human-level control through deep reinforcement learning???, 
*Nature* **518**, 2015, pp. 529-533. Available: 
[link](https://www.datascienceassn.org/sites/default/files/Human-level%20Control%20Through%20Deep%20Reinforcement%20Learning.pdf) [Accessed 2 Feb 2022]

<a id="6">[6]</a>
M. Hausknecht and P. Stone, ???Deep Recurrent Q-Learning for Partially Observable MDPs???, 
*arXiv:1507.06527v4 [cs.LG]*, 2017. Available: [link](https://arxiv.org/abs/1507.06527) [Accessed 2 Feb 2022]

<a id="7">[7]</a>
R.S. Sutton, D.A. McAllester, S.P. Singh, and Y. Mansour, ???Policy gradient methods for reinforcement learning with function approximation???, 
*Advances in neural information processing systems* **12**, 1999, pp. 1057???1063.

<a id="8">[8]</a>
D. Silver, G. Lever, N. Heess et al, ???Deterministic policy gradient algorithms???, 
*Proceedings of the 31st International Conference on Machine Learning*, 2014, pp. 387???395.

<a id="9">[9]</a>
R. J. Williams, ???Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning???, Machine Learning 8, 1992, pp. 229-256.

<a id="10">[10]</a>
S. Bhatnagar, R. Sutton, M. Ghavamzadeh and M. Lee, "Natural Actor-Critic Algorithms", 
*Automatica* **45**, 2009, pp. 2471-2482.

<a id="11">[11]</a>
T.P. Lillicrap, J.J. Hunt, A. Pritzel et al, ???Continuous Control with Deep Reinforcement Learning???, 
*arXiv:1509.02971v6 [cs.LG]*, 2019. Available: [link](https://arxiv.org/abs/1509.02971) [Accessed 2 Feb 2022]

<a id="12">[12]</a>
R. Lowe, Y. Wu, A. Tamar et al, ???Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments???, 
*arXiv:1706.02275v4 [cs.LG]*, 2020. Available: [link](https://arxiv.org/abs/1706.02275v4) [Accessed 2 Feb 2022]

<a id="13">[13]</a>
J.N. Foerster, Y.M. Assael, N. de Freitas et al, ???Learning to Communicate to Solve Riddles with Deep Distributed Recurrent Q-Networks???, 
*arXiv:1602.02672 [cs.AI]*, 2016. Available: [link](https://arxiv.org/abs/1602.02672) [Accessed 9 Feb 2022]

<a id="14">[14]</a>
M. Chan, "gym-maze", *GitHub*, 2020. Available: [link](https://github.com/MattChanTK/gym-maze) [Accessed 2 Feb 2022]

<a id="15">[15]</a>
F. Middleton-Baird, "gym-robot-maze", *GitHub*, 2021. Available: [link](https://github.com/finn1y/gym-robot-maze) [Accessed 2 Feb 2022]

