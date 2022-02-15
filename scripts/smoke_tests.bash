#!/usr/bin/env bash

#script to test rl_training_env on all variations of environments and algorithms to ensure all can complete training
#these tests do not ensure that the algorithms or environments will perform as designed just that they 
#will run without errors

#----------------------------------------------------------------------------------------
# Set up script
#----------------------------------------------------------------------------------------

#do not exit on error
set +e

#ensure script is executed from root
cd "$(dirname $0)/../rl_training_env/"

#----------------------------------------------------------------------------------------
# Functions
#----------------------------------------------------------------------------------------

Fails_str() {
    for fail in ${FAILS[@]}; do
        #format fail for grep of log file
        local GREP_IN=$(echo "$fail" | tr '!' ' ')
        #find line number in log file for the failed tests' logs
        local LINE_N=$(grep -n "$GREP_IN" ../logs/smoke_logs.txt | cut -f1 -d:)
    
        local STR=$(echo "  $fail" | sed -r 's/[!]+/ with /g')
        STR+=", at line $LINE_N in smoke_logs.txt"

        echo "$STR"
    done
}

#----------------------------------------------------------------------------------------
# Global variables
#----------------------------------------------------------------------------------------

#all discrete envs to be tested
ENVS=("maze-random-5x5-v0" "maze-random-10x10-v0" "maze-random-100x100-v0" 
        "maze-sample-5x5-v0" "maze-sample-10x10-v0" "maze-sample-100x100-v0" "gym_robot_maze:robot-maze-v0" 
        "CartPole-v1" "Acrobot-v1" "MountainCar-v0" "MountainCarContinuous-v0" "Pendulum-v1") 

#all discrete action algorithms to be tested
ALGORITHMS=("qlearning" "dqn" "drqn" "policy_gradient" "actor_critic" "ddrqn" "ma_actor_critic" "ddpg")

#init fails array to store any failed tests
FAILS=()

#----------------------------------------------------------------------------------------
# main
#----------------------------------------------------------------------------------------

#make logs dir if doesn't exist
if [ ! -d ../logs/ ]; then
    mkdir -p ../logs/
fi

#make smoke log file if doesn't exist
if [ ! -f ../logs/smoke_logs.txt ]; then
    touch ../logs/smoke_logs.txt
fi

#clear log contents
: > ../logs/smoke_logs.txt

for e in ${ENVS[@]}; do
    for a in ${ALGORITHMS[@]}; do
        #do not test mutli-agent only algorithms on single agent only envs
        if [[ " ${ENVS[@]:7:4} " =~ " ${e} " && " ${ALGORITHMS[@]:5:2} " =~ " ${a} " ]]; then
            continue
        fi

        #do not test continuous action algorithms on discrete action envs
        if [[ " ${ENVS[@]:0:10} " =~ " ${e} " && " ${ALGORITHMS[7]} " == " ${a} " ]]; then
            continue
        fi

        #do not test discrete action algorithms on continuous action envs
        if [[ " ${ENVS[@]:10:2} " =~ " ${e} " && " ${ALGORITHMS[@]:0:7} " =~ " ${a} " ]]; then
            continue
        fi

        #test each combination of algorithm and environment with a small number of episodes and timesteps
        CMD="./rl_training_env.py $e $a -e 5 -t 100"
        
        #add multiple agents for multi-agent environments
        if [[ " ${ENVS[@]:0:7} " =~ " ${e} " ]]; then
            CMD+=" -a 2"
        fi

        echo -e "Testing $a on $e\n$CMD" | tee -a ../logs/smoke_logs.txt
        eval $CMD >> ../logs/smoke_logs.txt 2>&1
        
        #if exit code is not equal to zero then error occured
        if [ $? -ne 0 ]; then
            #add test to list of failed tests
            FAILS+=("${e}!${a}")
            echo -e "Test failed\n"
        else
            echo -e "Test passed\n"
        fi

        #break up each log entry with a newline for readability
        echo -ne "\n" >> ../logs/smoke_logs.txt

    done
done

#print out number of failed tests and list of which tests failed
echo "CI completed with ${#FAILS[@]} failed tests:"

FAILS_STR=`Fails_str`
echo "$FAILS_STR"

exit 0


