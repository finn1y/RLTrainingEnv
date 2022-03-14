#!/usr/bin/env bash

#script to compare single agent algorithms on the same environment

#----------------------------------------------------------------------------------------
# Set up script
#----------------------------------------------------------------------------------------

#do not exit on error
set +e

#move to root
cd "$(dirname $0)/../rl_training_env/"

#----------------------------------------------------------------------------------------
# Functions
#----------------------------------------------------------------------------------------

#function to print help message to terminal
Help() {
    echo "Usage: ./sa_algorithm_comparison.bash [OPTIONS] ENV"
    echo -e "ENV\t\tEnvironment to train algorithm on.\n\t\tChoose from {${ENVS[@]}}"
    echo -e "OPTIONS:"
    echo -e "  -h, --help\tShows this message"
    echo -e "  --maze-load-path\tPath to load maze from, for use with gym-robot-maze environment"
}

#function to process command line arguments and set to relevant variables
Get_args() {
    #loop until all arguments have been processed
    while [ $# -gt 0 ]; do
        #switch argument to check which option is being processed
        case $1 in

            "-h" | "--help")
            Help
            exit 0
            ;;
    
            "--maze-load-path")
                shift
                MAZE_LOAD_PATH=$1
            ;;
    
            *)
            #no option means either positional argument or not a valid options
            if [[ " ${ENVS[@]} " =~ " $1 " ]]; then
                ENV=$1
            else
                echo "WARNING: $1 is not a valid option. Use -h or --help to see valid options"
            fi
            ;;

        esac

        #shift options to evaluate next
        shift
    done

    #check necessary options supplied by user
    if [ -z $ENV ]; then
        #no env supplied by user
        echo "ERROR: no environment supplied"
        Help
        exit 1
    fi
}

#function to get the name of the save directory for data
Get_save_dir() {
    #find all directories that have multi-agent data
    local DIRS=( $(find -maxdepth 3 -type d -name "single_agent*") )

    #find returns empty string if none found -> number of elements in array will be 1 if none found
    if [ -z "${DIRS[0]}" ]; then
        #if element is empty string then n is 0
        local n=0
    else
        #n is the number of directories total
        local n=${#DIRS[@]}
    fi

    echo "saved_data/single_agent_$n"
}

#----------------------------------------------------------------------------------------
# Global variables
#----------------------------------------------------------------------------------------

#possible envs to test on
ENVS=("maze-random-5x5-v0" "maze-random-10x10-v0" "maze-random-100x100-v0" 
        "maze-sample-5x5-v0" "maze-sample-10x10-v0" "maze-sample-100x100-v0" "gym_robot_maze:RobotMaze-v1" 
        "CartPole-v1" "Acrobot-v1" "MountainCar-v0")

#algorithms to test
ALGORITHMS=("qlearning" "dqn" "drqn" "policy_gradient" "actor_critic")

#----------------------------------------------------------------------------------------
# main
#----------------------------------------------------------------------------------------

#pass script arguments to get args function for processing
Get_args "$@"

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

#get directory prefix
DIR_PREFIX=`Get_save_dir`

for a in ${ALGORITHMS[@]}; do
    #directory to save data to
    DIR="$DIR_PREFIX/$ENV/$a"

    CMD="./rl_training_env.py $ENV $a -e 1000 -d $DIR" 

    #if maze load path supplied and env is robot-maze add option to command
    if [[ -z $MAZE_LOAD_PATH && "$ENV" == "gym_robot_maze:RobotMaze-v1" ]]; then
        CMD+=" --maze-load-path $MAZE_LOAD_PATH"
    fi

    echo -e "Running $a on $ENV\n$CMD" | tee -a ../logs/algorithm_sa_logs.txt
    eval $CMD >> ../logs/algorithm_sa_logs.txt
    echo "Saved data to $PWD/$DIR" | tee -a ../logs/algorithm_sa_logs.txt

    #break up each log entry with a newline for readability
    echo -ne "\n" >> ../logs/algorithm_sa_logs.txt
done

echo "Script finished successfully"

exit 0


