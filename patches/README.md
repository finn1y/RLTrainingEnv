# Dependency Patches

This directory contains patches for dependencies of the RL training environment

## Applying Patches

To apply a patch to a git repository the following commands should be used, taking `gym_maze_multi_agent.patch` as an example:

```
cp ./gym_maze_multi_agent.patch ../../gym-maze/
cd ../../gym-maze
git apply gym_maze_multi_agent.patch
```

To view what the patch changes either by looking in the `.patch` file or by using the `git diff` command after applying the patch.

## Patch files

### [gym-maze](https://github.com/MattChanTK/gym-maze)

##### 1. `gym_maze_multi_agent.patch`

    Converts the gym-maze environment created by "MattChanTK" into a multi-agent environment by taking in arrays of actions, 
    processing actions for each robot and outputting the rewards and states as arrays


