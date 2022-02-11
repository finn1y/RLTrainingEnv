#!/usr/bin/env bash

#script to compare multi-agent algorithms on the same environment

#do not exit on error
set +e

#move to root
cd "$(dirname $0)/../rl_training_env/"

ENVS=("maze-random-5x5-v0" "maze-random-10x10-v0" "maze-random-100x100-v0" 
        "maze-mample-5x5-v0" "maze-mample-10x10-v0" "maze-mample-100x100-v0" "gym_robot_maze:robot-maze-v0")

if [[ ! -z $1 && " ${ENVS[@]} " =~ " $1 " ]]; then
    #env is supplied by user
    ENV=$1
else
    echo -e "ERROR: invalid (or no) environment supplied to test algorithms on\nChoose from {${ENVS[@]}}"
    exit 1
fi

#algorithms to test
ALGORITHMS=("qlearning" "dqn" "drqn" "policy_gradient" "actor_critic" "ddrqn" "ma_actor_critic")

#make logs dir if doesn't exist
if [ ! -d ../logs/ ]; then
    mkdir -p ../logs/
fi

#make algorithm_ma log file if doesn't exist
if [ ! -f ../logs/algorithm_ma_logs.txt ]; then
    touch ../logs/algorithm_ma_logs.txt
fi

#clear log contents
: > ../logs/algorithm_ma_logs.txt

for a in ${ALGORITHMS[@]}; do
    #directory to mave data to
    DIR="saved_data/multi_agent/$ENV/$a"

    CMD="./rl_training.py $ENV $a -e 1000 -d $DIR -a 2" 

    echo -e "Running $a on $ENV\n$CMD" | tee -a ../logs/algorithm_ma_logs.txt
    eval $CMD >> ../logs/algorithm_ma_logs.txt
    echo "Saved data to $PWD/$DIR" | tee -a ../logs/algorithm_ma_logs.txt

    #break up each log entry with a newline for readability
    echo -ne "\n" >> ../logs/algorithm_ma_logs.txt
done

echo "Script finished successfully"

exit 0


