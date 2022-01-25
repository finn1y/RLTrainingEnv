# Dependency Patches

This directory contains patches for dependencies of the RL training environment

## Applying Patches

To apply a patch to a git repository then:

1. copy the patch into the relevant repo
2. move to that repo
3. apply the patch

taking `gym_maze_multi_agent.patch` as an example:

```
cp ./gym_maze_multi_agent.patch ../../gym-maze/
cd ../../gym-maze
git apply gym_maze_multi_agent.patch
```

To view what the patch changes either by looking in the `.patch` file or by using the `git diff` command after applying the patch.

## Patch files

### [gym-maze](https://github.com/MattChanTK/gym-maze)

##### 1. gym\_maze\_multi\_agent.patch

Converts the gym-maze environment created by "MattChanTK" into a multi-agent environment by taking in arrays of actions, 
processing actions for each robot and outputting the rewards and states as arrays


