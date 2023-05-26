#! /bin/bash

## RUN EXPERIMENTS

folder="output_2023-ICML"

# TESTING
# python python/experiments.py iris logistic permutation_auc 1 $folder --clean

# MAIN EXPERIMENTS
# Baseline
 for d in vote compas german copdmortality heartfailurestroke
 do
  for m in logistic nn
    do
      python python/experiments.py $d $m permutation_auc 5 $folder
      for f in permutation_mse loco_auc loco_mse loco_ba kernelshap sage_marginal sage_conditional
      do
          echo "Run for dataset $d model $m and feature importance method $f"
          python python/experiments.py $d $m $f 5 $folder --use-model
      done
    done
 done

# Modify data
for d in compas german
do
  for m in logistic nn
    do
      python python/experiments.py $d $m permutation_auc 5 $folder --modify-data
      for f in permutation_mse loco_auc loco_mse loco_ba kernelshap sage_marginal sage_conditional
      do
          echo "Run for dataset $d model $m and feature importance method $f"
          python python/experiments.py $d $m $f 5 $folder --modify-data --use-model
      done
    done
done

### AGGREGATE OUTPUT
for d in vote compas german copdmortality heartfailurestroke
do
  for m in logistic nn
    do
      python python/output.py $d $m $folder
    done
done
