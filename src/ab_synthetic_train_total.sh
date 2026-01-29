#!/bin/bash

set -e

cd "$(dirname "$0")"

# set CUDA device once (use environment if provided, otherwise default to 1)
export CUDA_VISIBLE_DEVICES=0
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# set CUBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8


echo "choose your task to run:"
echo "1. forward"
echo "2. switch"
echo "3. backward"
echo "4. all tasks (running sequentially in the order of forward, switch, backward)"
echo "q. EXIT"
read -p "please enter (1/2/3/q): " choice
choice=$(echo "$choice" | tr 'A-Z' 'a-z')

case $choice in
    1)
        python core/ab_synthetic_train.py --task "forward"
        ;;
    2)
        python core/ab_synthetic_train.py --task "switch"
        ;;
    3)
        python core/ab_synthetic_train.py --task "backward"
        ;;
    4)
        # running all tasks sequentially
        python core/ab_synthetic_train.py --task "forward"
        python core/ab_synthetic_train.py --task "switch"
        python core/ab_synthetic_train.py --task "backward"
        ;;
    q)
        echo "exited"
        exit 0
        ;;
    *)
        echo "invalid input, please choose 1/2/3/q。"
        exit 1
        ;;
esac
