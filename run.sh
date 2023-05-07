#! /bin/bash

## RUN EXPERIMENTS

folder="output_2023-05-07"

# format:
# python python/experiments.py [data] [model] [fi_method] [repeats]

# [data] in ['iris']
# [model] in ['linear', 'logistic']
# [fi_method] in ['permutation_auc']
# [repeats] in [1, 2, 3, etc.]

# Code testing
# python python/experiments.py compas logistic sage_conditional 1

# Main experiments
python python/experiments.py compas logistic permutation_ba 1 $folder --clean

for d in vote compas
do
  for m in logistic nn xgboost
    do
      python python/experiments.py $d $m permutation_auc 1 $folder

      for f in permutation_mse sage_marginal sage_conditional loco_auc loco_mse kernelshap
      do
          echo "Run for dataset $d model $m and feature importance method $f"
          python python/experiments.py $d $m $f 1 $folder --use-model
      done
    done
done

# GERMAN
#python python/experiments.py german logistic permutation_auc 1 --clean
#python python/experiments.py german logistic permutation_mse 1 --use-model
#python python/experiments.py german logistic permutation_ba 1 --use-model
#python python/experiments.py german logistic loco_auc 1 --use-model
#python python/experiments.py german logistic loco_mse 1 --use-model
#python python/experiments.py german logistic loco_ba 1 --use-model

#python python/experiments.py german nn permutation_auc 1
#python python/experiments.py german nn permutation_mse 1 --use-model
#python python/experiments.py german nn permutation_ba 1 --use-model
#python python/experiments.py german nn loco_auc 1 --use-model
#python python/experiments.py german nn loco_mse 1 --use-model
#python python/experiments.py german nn loco_ba 1 --use-model

#python python/experiments.py german xgboost permutation_auc 1
#python python/experiments.py german xgboost permutation_mse 1 --use-model
#python python/experiments.py german xgboost permutation_ba 1 --use-model
#python python/experiments.py german xgboost loco_auc 1 --use-model
#python python/experiments.py german xgboost loco_mse 1 --use-model
#python python/experiments.py german xgboost loco_ba 1 --use-model

#python python/experiments.py german logistic kernelshap 1 --use-model
#python python/experiments.py german logistic sage_marginal 1 --use-model

# TODO
#python python/experiments.py german nn kernelshap 1 --use-model
#python python/experiments.py german nn sage_marginal 1 --use-model

#python python/experiments.py german xgboost kernelshap 1 --use-model
#python python/experiments.py german xgboost sage_marginal 1 --use-model

### AGGREGATE OUTPUT

# format:
# python python/output.py [data] [model]

#python python/output.py german logistic

#for d in iris
#do
#  for m in logistic nn xgboost
#    do
#      python python/output.py $d $m
#    done
#done

### VISUALIZE OUTPUT
