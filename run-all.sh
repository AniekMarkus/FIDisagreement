#! /bin/bash

## RUN EXPERIMENTS

folder="output_2023-05-10-mac"

# format:
# python python/experiments.py [data] [model] [fi_method] [repeats]

# [data] in ['iris', 'vote', 'compas', 'german', 'copdmortality', 'heartfailurestroke'] # last two not public
# [model] in ['logistic', 'nn', 'xgboost']
# [fi_method] in ['permutation_auc', 'permutation_mse', 'permutation_ba', 'loco_auc', 'loco_mse', 'loco_ba',
#                     'sage_marginal', 'sage_conditional', 'kernelshap']
# [repeats] in [1, 2, 3, etc.]

# TESTING
python python/experiments.py iris logistic permutation_auc 1 $folder --clean

# MAIN EXPERIMENTS
# Baseline
for d in copdmortality heartfailurestroke
do
  for m in logistic nn xgboost
    do
      python python/experiments.py $d $m permutation_auc 1 $folder
      for f in permutation_mse permutation_ba sage_marginal sage_conditional loco_auc loco_mse loco_ba kernelshap
      do
          echo "Run for dataset $d model $m and feature importance method $f"
          python python/experiments.py $d $m $f 1 $folder --use-model
      done
    done
done

# Modify data
# TODO: check if modify_params should depend on data
# TODO: test first on laptop for iris/small data
for d in vote compas german copdmortality heartfailurestroke
do
  for m in logistic nn xgboost
    do
      python python/experiments.py $d $m permutation_auc 1 $folder --modify-data
      for f in permutation_mse permutation_ba sage_marginal sage_conditional loco_auc loco_mse loco_ba kernelshap
      do
          echo "Run for dataset $d model $m and feature importance method $f"
          python python/experiments.py $d $m $f 1 $folder --modify-data --use-model
      done
    done
done

### AGGREGATE OUTPUT

# format:
# python python/output.py [data] [model]

for d in vote compas german copdmortality heartfailurestroke
do
  for m in logistic nn xgboost
    do
      python python/output.py $d $m $folder
    done
done

### VISUALIZE OUTPUT
