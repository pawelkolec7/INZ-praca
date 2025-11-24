# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

# ── Ścieżki ──────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\kolec\AppData\Roaming\MetaQuotes\Terminal\Common\Files\ai_forex")
MT5_CSV  = DATA_DIR / "ohlc_EURUSD_H1.csv"   # <- plik z EA (dostosuj nazwę)

# ── Parametry ZigZag / datasetu ─────────────────────────────────
MIN_ZP     = 0.0050
ILE_ZZ     = 30
LICZBA_SW  = 15
DL_ZZ      = 10
BACKSTEP   = 0
TRAIN_FRACTION = 0.90   # train/test

# ── Hiperparametry MLP ──────────────────────────────────────────
HYPERPARAMS = {
    "hidden_sizes": [64, 32],
    "output_dim": 1,
    "learning_rate": 0.00004,
    "batch_size": 32,
    "epochs": 50,      # na start 50; podbijesz jak chcesz
    "dropout_p": 0.6,
    "weight_decay": 0.0002
}

SEED = 12
MODEL_PATH = DATA_DIR / "model.pt"
STATS_PATH = DATA_DIR / "norm_stats.pkl"


# %%
