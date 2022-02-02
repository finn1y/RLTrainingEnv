# RL Trainging Environment

A training environment for reinforcement learning algorithms using [open-ai gym](https://gym.openai.com/)

## Included Algorithms

Both single agent and multi-agent reinforcement learning algorithms are included within the learning environment
Any single agent algorithm can be used as an independent multi-agent learning algorithm with the multi-agent environments
All implementations also include exploration rate (epsilon) and learning rate (alpha) decay where appropriate;
this can be removed by setting the relevant decay rate to 0

#### [Q-Learning](algorithms/qlearning.py)

Q-learning is implemented based on the algorithm described by Sutton and Barto in [[1]](#1).

#### [Deep Q-Network](algorithms/dqn.py) (DQN)

Deep Q-network is implemeneted based on the algorithm described by Minh et al in [[2]](#2).
However it does not use CNNs as the environments used in this training are not array based (i.e. not an image representation) 

#### [Deep Recurrent Q-Network](algorithms/dqn.py) (DRQN)

Deep Recurrent Q-Network is implemented based on the alterations to DQN as suggested by Hausknecht and Stone in [[3]](#3).

#### [Policy Gradient](algorihms/policy_grad.py) (PG)

Policy Gradient is implemented based on the algorithm suggested by Sutton et al in [[4]](#4) 
and the determinisitc counterpart in [[5]](#5) by Silver et al.

#### [Advantage Actor Critic](algorithms/actor_critic.py) (A2C)

Advantage Actor Critic is implemented based on one of the actor critic variations suggested by Bhatnagar et al in [[6]](#6).

#### [Deep Deterministic Policy Gradient](algorithms/ddpg.py) (DDPG)

Deep Deterministic Policy Gradient is implemented based on the algorithm as suggested in [[7]](#7) by Lillicrap et al.

#### [Multi-Agent Actor Critic](algorithms/ma_actor_critic.py) (MA Actor Critic)

Multi-Agent Actor Critic is implemented based on a the algorithm described by Lowe et al in [[8]](#8). 
As the multi-agent environments are cooperative there is communication of agent policy so no policy inference is required 
nor are policy ensembles.

#### [Distributed Deep Recurrent Q-Network](algorithms/ddrqn.py) (DDRQN)

Distributed Deep Recurrent Q-Network is implemented based on the changes to Deep Q-Networks suggested by Foerster et al in [[9]](#9) 
for multi-agent environments. Communication is pre-defined in this implementation and is not learnt as is the case for RIAL and DIAL. 

## Included Environments

Several of openai gyms' environments are included as single agent environments are included, as well as some custom 
environments which have both single agent and multi-agent variations

#### [Maze](https://github.com/MattChanTK/gym-maze)

#### [Robot Maze](https://github.com/finn1y/gym-robot-maze)

#### [Cart Pole](https://gym.openai.com/envs/CartPole-v1/)

#### [Acrobot](https://gym.openai.com/envs/Acrobot-v1/)

#### [Mountain Car](https://gym.openai.com/envs/MountainCar-v0/)

#### [Mountain Car Continuous](https://gym.openai.com/envs/MountainCarContinuous-v0/)

#### [Pendulum](https://gym.openai.com/envs/Pendulum-v0/)

## Install

1. Clone the repo
```
git clone https://github.com/finn1y/RLTraingingEnv
```
2. Clone dependencies from github
```
git clone https://github.com/finn1y/gym-robot-maze
git clone https://github.com/MattChanTK/gym-maze
```
3. Apply dependency patches, described in [patches](patches/README.md)
4. Install python dependencies in repo
```
cd RLTrainingEnv
pip install -r requirements.txt
```
5. Enjoy training some RL agents!

## References

<a id="1">[1]</a> 
R.S. Sutton and A.G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. The MIT Press, 2018.

<a id="2">[2]</a>
V. Mnih, K. Kavukcuoglu, D. Silver et al, “Human-level control through deep reinforcement learning”, 
*Nature* **518**, 2015, pp. 529-533. Available: 
[link](https://www.datascienceassn.org/sites/default/files/Human-level%20Control%20Through%20Deep%20Reinforcement%20Learning.pdf) [Accessed 2 Feb 2022]

<a id="3">[3]</a>
M. Hausknecht and P. Stone, “Deep Recurrent Q-Learning for Partially Observable MDPs”, 
*arXiv:1507.06527v4* [cs.LG], 2017. Available: [link](https://arxiv.org/abs/1507.06527) [Accessed 2 Feb 2022]

<a id="4">[4]</a>
R.S. Sutton, D.A. McAllester, S.P. Singh, and Y. Mansour, “Policy gradient methods for reinforcement learning with function approximation”, 
*Advances in neural information processing systems* **12**, 1999, pp. 1057–1063.

<a id="5">[5]</a>
D. Silver, G. Lever, N. Heess et al, “Deterministic policy gradient algorithms”, 
*Proceedings of the 31st International Conference on Machine Learning*, 2014, pp. 387–395.

<a id="6">[6]</a>
S. Bhatnagar, R. Sutton, M. Ghavamzadeh and M. Lee, "Natural Actor-Critic Algorithms", 
*Automatica* **45**, 2009, pp. 2471-2482.

<a id="7">[7]</a>
T.P. Lillicrap, J.J. Hunt, A. Pritzel et al, “Continuous Control with Deep Reinforcement Learning”, 
*arXiv:1509.02971v6 [cs.LG]*, 2019. Available: [link](https://arxiv.org/abs/1509.02971) [Accessed 2 Feb 2022]

<a id="8">[8]</a>
R. Lowe, Y. Wu, A. Tamar et al, “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments”, 
*arXiv:1706.02275v4 [cs.LG]*, 2020. Available: [link](https://arxiv.org/abs/1706.02275v4) [Accessed 2 Feb 2022]

<a id="9">[9]</a>
J.N. Foerster, Y.M. Assael, N. de Freitas et al, “Learning to Communicate with Deep Multi-Agent Reinforcement Learning”, 
*arXiv:1605.06676v2 [cs.AI]*, 2016. Available: [link](https://arxiv.org/abs/1605.06676v2) [Accessed 2 Feb 2022]



