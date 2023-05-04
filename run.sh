#! /bin/bash

## RUN EXPERIMENTS

# format:
# python python/experiments.py [data] [model] [fi_method] [repeats]

# [data] in ['iris']
# [model] in ['linear', 'logistic']
# [fi_method] in ['permutation_auc']
# [repeats] in [1, 2, 3, etc.]

# Code testing
# python python/experiments.py iris logistic permutation_auc 2 --clean
python python/experiments.py compas logistic permutation_mse 1 --modify-data

# Main experiments
# TODO: add he

### AGGREGATE OUTPUT

# format:
# python python/output.py [data] [model]
python python/output.py compas logistic

### VISUALIZE OUTPUT
