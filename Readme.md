## Install :

TODO: To install the core environment dependencies of OpenXAI, use `pip` by cloning this repo into your local environment:

```bash
pip install -e . 
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


