This repository contains the code for the evaluation framework and experiments presented in:
***Understanding the Size of the Feature Importance Disagreement Problem in Real-World Data***

## Replicate experiments:

Run experiments for specific settings:
* [data] in ['iris', 'vote', 'compas', 'german', 'copdmortality', 'heartfailurestroke'] # last two not public
* [model] in ['logistic', 'nn', 'xgboost']
* [fi_method] in ['permutation_auc', 'permutation_mse', 'permutation_ba', 'loco_auc', 'loco_mse', 'loco_ba', 'sage_marginal', 'sage_conditional', 'kernelshap']
* [repeats] in [1, 2, 3, etc.]

```
   python python/experiments.py [data] [model] [fi_method] [repeats]
```

Run all experiments:
```
sh run.sh
```

## Check results:

Aggregate output for specific settings:
```
python output.py [data] [model]
```

Aggregate all output:
```
sh run.sh
```

Checkout plots and tables in:
```
/results/
```

Interactive python DASH app:
```
python app/results_explorer.py
```


