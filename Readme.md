## Install :
```

```


## Replicate experiments:

Run pipeline for specific settings:

```
   python python/experiments.py [data] [model] [fi_method] [repeats] [rerun]
```

Run all experiments:
```
sh run.sh
# TODO: add flag --no-output
```

## Check results:

Aggregate output for specific settings:
```
python output.py [data]
```

Aggregate all experiments:
```
sh run.sh
# TODO: add flag --show-output
```

Checkout plots and tables in:
```
/results/
```

Interactive python DASH app:
```
python app/results_explorer.py
```


