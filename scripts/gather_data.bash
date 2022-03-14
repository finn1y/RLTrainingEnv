#!/usr/bin/env bash

#script to automatically run comparison scripts to gather data on desired algorithms and environments

#----------------------------------------------------------------------------------------
# Set up script
#----------------------------------------------------------------------------------------

#do not exit on error
set +e

#move to root
cd "$(dirname $0)/../"

#----------------------------------------------------------------------------------------
# Global variables
#----------------------------------------------------------------------------------------

#path to saved maze for gym-robot-maze environments
ROBOT_MAZE_PATH="rl_training_env/robot_maze_sample/"

#list of single agent environments
SA_ENVS=("CartPole-v1" "maze-sample-5x5-v0" "maze-sample-10x10-v0" "maze-sample-100x100-v0" "gym_robot_maze:robot-maze-v0")

#array of multi-agent environments
MA_ENVS=("maze-sample-5x5-v0" "maze-sample-10x10-v0" "maze-sample-100x100-v0" "gym_robot_maze:robot-maze-v0")

#----------------------------------------------------------------------------------------
# main
#----------------------------------------------------------------------------------------

#loop through single agent envs and run comparison script
for env in ${SA_ENVS[@]}; do
    CMD="./scripts/sa_algorithm_comparison.bash $env"

    if [[ "$env" == "gym_robot_maze:robot-maze-v0" ]]; then
        CMD+=" --maze-load-path $ROBOT_MAZE_PATH"
    fi

    echo $CMD
    eval $CMD

    if [ $? -ne 0 ]; then
        echo "ERROR: single agent on $env failed"
    fi
done

#loop through multi-agent envs and run comparison script
for env in ${MA_ENVS[@]}; do
    CMD="./scripts/ma_algorithm_comparison.bash $env"

    if [[ "$env" == "gym_robot_maze:robot-maze-v0" ]]; then
        CMD+=" --maze-load-path $ROBOT_MAZE_PATH"
    fi

    echo $CMD
    eval $CMD

    if [ $? -ne 0 ]; then
        echo "ERROR: multi-agent on $env failed"
    fi
done

exit 0
