# focal-point-submission

Short repo for running experiments and plotting results related to bargaining / focal-point experiments.

## Main components
- `main.py`: primary entry point for running experiments.
- `src/bargaining_table/bargaining_table.py` / `src/bargaining_table/bargaining_table.sh`: generate bargaining table results.
- `plot_bardsley.py` / `plot_bardsley.sh`: plot Bardsley-related results.
- `plot_bargaining_table_averages.py`: plot averages for bargaining tables.
- `experiments.sh`: run experiment batches.
- `src/`: library code used by the scripts.
- `data/`: input datasets and cached outputs.
- `plots/`: generated figures.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: containerized environment.

## Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick usage
```bash
python main.py
bash experiments.sh
bash plot_bardsley.sh
```
