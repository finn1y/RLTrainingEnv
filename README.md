# RL Trainging Environment

A training environment for reinforcement learning algorithms using [open-ai gym](https://gym.openai.com/)

## Included Algorithms

Both single agent and multi-agent reinforcement learning algorithms are included within the learning environment
Any single agent algorithm can be used as an independent multi-agent learning algorithm with the multi-agent environments

#### [Q-Learning](algorithms/qlearning.py)

#### [Deep Q-Network](algorithms/dqn.py) (DQN)

#### [Deep Recurrent Q-Network](algorithms/dqn.py) (DRQN)

#### [Policy Gradient](algorihms/policy_grad.py) (PG)

#### [Advantage Actor Critic](algorithms/actor_critic.py) (A2C)

#### [Deep Deterministic Policy Gradient](algorithms/ddpg.py) (DDPG)

#### [Multi-agent Actor Critic](algorithms/ma_actor_critic.py) (MA Actor Critic)

#### [Distributed Deep Recurrent Q-Network](algorithms/ddrqn.py) (DDRQN)

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


