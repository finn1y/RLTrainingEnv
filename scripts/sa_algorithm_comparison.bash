#!/usr/bin/env bash

#script to compare single agent algorithms on the same environment

#do not exit on error
set +e

#move to root
cd "$(dirname $0)/../rl_training_env/"

ENVS=("maze-random-5x5-v0" "maze-random-10x10-v0" "maze-random-100x100-v0" 
        "maze-sample-5x5-v0" "maze-sample-10x10-v0" "maze-sample-100x100-v0" "gym_robot_maze:robot-maze-v0" 
        "CartPole-v1" "Acrobot-v1" "MountainCar-v0")

if [[ ! -z $1 && " ${ENVS[@]} " =~ " $1 " ]]; then
    #env is supplied by user
    ENV=$1
else
    echo -e "ERROR: invalid (or no) environment supplied to test algorithms on\nChoose from {${ENVS[@]}}"
    exit 1
fi

#algorithms to test
ALGORITHMS=("qlearning" "dqn" "drqn" "policy_gradient" "actor_critic")

#make logs dir if doesn't exist
if [ ! -d ../logs/ ]; then
    mkdir -p ../logs/
fi

#make algorithm_sa log file if doesn't exist
if [ ! -f ../logs/algorithm_sa_logs.txt ]; then
    touch ../logs/algorithm_sa_logs.txt
fi

#clear log contents
: > ../logs/algorithm_sa_logs.txt

for a in ${ALGORITHMS[@]}; do
    #directory to save data to
    DIR="saved_data/single_agent/$ENV/$a"

    CMD="./rl_training_env.py $ENV $a -e 1000 -d $DIR" 

    echo -e "Running $a on $ENV\n$CMD" | tee -a ../logs/algorithm_sa_logs.txt
    eval $CMD >> ../logs/algorithm_sa_logs.txt
    echo "Saved data to $PWD/$DIR" | tee -a ../logs/algorithm_sa_logs.txt

    #break up each log entry with a newline for readability
    echo -ne "\n" >> ../logs/algorithm_sa_logs.txt
done

echo "Script finished successfully"

exit 0


