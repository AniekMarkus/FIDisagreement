#! /bin/bash

## RUN EXPERIMENTS

# format:
# python python/experiments.py [data] [model] [fi_method] [repeats] [rerun]

# [data] in ['iris']
# [model] in ['linear', 'logistic']
# [fi_method] in ['permutation_auc']
# [repeats] in [1, 2, 3, etc.]
# [clean] in [True, False]

# Code testing
python python/experiments.py iris logistic permutation_auc 2 --clean
python python/experiments.py iris logistic permutation_mse 1 --use-model

# Main experiments
# TODO: add here

### AGGREGATE OUTPUT

# format:
# python python/output.py [data]

python python/output.py iris
