#!/usr/bin/env bash

#script to compare algorithm parameters on the same environment

#do not exit on error
set +e

#move to root
cd "$(dirname $0)/../rl_training_env/"

#possible environments to test on
ENVS=("maze-random-5x5-v0" "maze-random-10x10-v0" "maze-random-100x100-v0" 
        "maze-sample-5x5-v0" "maze-sample-10x10-v0" "maze-sample-100x100-v0" "gym_robot_maze:robot-maze-v0" 
        "CartPole-v1" "Acrobot-v1" "MountainCar-v0" "MountainCarContinuous-v0" "Pendulum-v1")

#possible algorithms to test
ALGORITHMS=("qlearning" "dqn" "drqn" "policy_gradient" "actor_critic" "ma_actor_critic" "ddrqn" "ddpg")

#possible parameters to change
PARAMETERS=("batch-size" "hidden-size" "gamma" "epsilon-min" "epsilon-max" "learning-rate" "learning-rate-decay")

#function to print help message to terminal
Help() {
    echo "Usage: ./parameter_comparison.bash [OPTIONS] ENV ALGORITHM PARAMETER"
    echo -e "ENV\t\tEnvironment to train algorithm on.\n\t\tChoose from {${ENVS[@]}}"
    echo -e "ALGORITHM\tAlgorithm to train agent with.\n\t\tChoose from {${ALGORITHMS[@]}}"
    echo -e "PARAMETER\tParameter to vary.\n\t\tChoose from {${PARAMETERS[@]}}"
    echo -e "OPTIONS:"
    echo -e "  -h, --help\tShows this message"
    echo -e "  --max\tMax value of parameter. Defaults to value based on provided parameter"
    echo -e "  --min\tMin value of parameter. Defaults to value based on provided parameter"
    echo -e "  -s, --step\tStep to increase parameter value by from Min value to Max value. Defaults to value based on provided parameter"
}

#parse command line arguments and options
if [ ! -z $1 ]; then
    while [ $# -gt 0 ]; do
        #switch argument to check which option is being processed
        case $1 in

            "-h" | "--help")
            Help
            exit 0
            ;;

            "--max")
                #shift option to get the value associated with the switch
                shift
                MAX=$1
            ;;

            "--min")
                shift
                MIN=$1
            ;;

            "-s" | "--step")
                shift
                STEP=$1
            ;;

            *)
            #no option means either positional argument or not a valid options
            if [[ " ${ENVS[@]} " =~ " $1 " ]]; then
                ENV=$1
            elif [[ " ${ALGORITHMS[@]} " =~ " $1 " ]]; then
                ALGORITHM=$1
            elif [[ " ${PARAMETERS[@]} " =~ " $1 " ]]; then
                PARAMETER=$1
            else
                echo "WARNING: $1 is not a valid option. Use -h or --help to see valid options"
            fi
            ;;

        esac

        #shift options to evaluate next
        shift
    done
fi

#check necessary options supplied by user
if [ -z $ENV ]; then
    #no env supplied by user
    echo "ERROR: no environment supplied"
    Help
    exit 1

elif [ -z $ALGORITHM ]; then
    #no algorithm supplied by user
    echo "ERROR: no algorithm supplied"
    Help
    exit 1

elif [ -z $PARAMETER ]; then
    #no parameter supplied by user
    echo "ERROR: no parameter supplied"
    Help
    exit 1

elif [ -z $MAX  ] || [ -z $MIN ] || [ -z $STEP ]; then
    #missing values, setting to default values for given parameter

    MAX_SET=0
    MIN_SET=0
    STEP_SET=0

    if [ -z $MAX ]; then
        MAX_SET=1
    fi

    if [ -z $MIN ]; then
        MIN_SET=1
    fi

    if [ -z $STEP ]; then
        STEP_SET=1
    fi

    case "$PARAMETER" in
        "batch-size")
            if [ $MAX_SET -eq 1 ]; then
                MAX=96
            fi

            if [ $MIN_SET -eq 1 ]; then
                MIN=16
            fi

            if [ $STEP_SET -eq 1 ]; then    
                STEP=8
            fi
            ;;

        "hidden-size")
            if [ $MAX_SET -eq 1 ]; then
                MAX=
            fi

            if [ $MIN_SET -eq 1 ]; then
                MIN=64
            fi

            if [ $STEP_SET -eq 1 ]; then    
                STEP=
            fi
            ;;

        "gamma")
            if [ $MAX_SET -eq 1 ]; then
                MAX=0.99
            fi

            if [ $MIN_SET -eq 1 ]; then
                MIN=0.09
            fi

            if [ $STEP_SET -eq 1 ]; then    
                STEP=0.05
            fi
            ;;

        "epsilon-min")
            if [ $MAX_SET -eq 1 ]; then
                MAX=0.1
            fi

            if [ $MIN_SET -eq 1 ]; then
                MIN=0.01
            fi

            if [ $STEP_SET -eq 1 ]; then    
                STEP=0.005
            fi
            ;;

        "epsilon-max")
            if [ $MAX_SET -eq 1 ]; then
                MAX=1.0
            fi

            if [ $MIN_SET -eq 1 ]; then
                MIN=0.5
            fi

            if [ $STEP_SET -eq 1 ]; then    
                STEP=0.025
            fi
            ;;

        "learning-rate")
            if [ $MAX_SET -eq 1 ]; then
                MAX=0.9
            fi

            if [ $MIN_SET -eq 1 ]; then
                MIN=0.1
            fi

            if [ $STEP_SET -eq 1 ]; then    
                STEP=0.05
            fi
            ;;
        
        "learning-rate-decay")
            if [ $MAX_SET -eq 1 ]; then
                MAX=1.0
            fi

            if [ $MIN_SET -eq 1 ]; then
                MIN=0.1
            fi

            if [ $STEP_SET -eq 1 ]; then    
                STEP=0.05
            fi
            ;;
    esac

    #warn user of setting to default values
    if [ $MAX_SET -eq 1 ]; then
        echo "WARNING: no max value supplied setting to default value $MAX"
    fi

    if [ $MIN_SET -eq 1 ]; then
        echo "WARNING: no min value supplied setting to default value $MIN"
    fi

    if [ $STEP_SET -eq 1 ]; then    
        echo "WARNING: no step value supplied setting to default value $STEP"
    fi
fi

#make logs dir if doesn't exist
if [ ! -d ../logs/ ]; then
    mkdir -p ../logs/
fi

#make parameter_sa log file if doesn't exist
if [ ! -f ../logs/parameter_sa_logs.txt ]; then
    touch ../logs/parameter_sa_logs.txt
fi

#clear log contents
: > ../logs/parameter_sa_logs.txt

#create array of values for parameter
P_VALS=()

#fill array with values defined by options
for i in $(seq $MIN $STEP $MAX); do
    P_VALS+=("$i")
done 

for p_val in ${P_VALS[@]}; do
    #directory to save data to
    DIR="saved_data/parameters/${ENV}_${ALGORITHM}/${PARAMETER}_${p_val}"

    CMD="./rl_training.py $ENV $ALGORITHM -e 1000 --$PARAMETER $p_val -d $DIR" 

    echo -e "Running $a on $ENV\n$CMD" | tee -a ../logs/parameter_sa_logs.txt
    eval $CMD >> ../logs/parameter_sa_logs.txt
    echo "Saved data to $PWD/$DIR" | tee -a ../logs/parameter_sa_logs.txt

    #break up each log entry with a newline for readability
    echo -ne "\n" >> ../logs/parameter_sa_logs.txt
done

echo "Script finished successfully"

exit 0


