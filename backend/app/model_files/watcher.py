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
# %run ./config.ipynb
# %run ./data_pipeline.ipynb

# %%
import time, pickle, torch, numpy as np
from pathlib import Path
from datetime import datetime


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, dropout_p=0.5):
        super().__init__()
        layers=[]; prev=input_dim
        for h in hidden_sizes:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(dropout_p)]
            prev = h
        layers += [torch.nn.Linear(prev, output_dim)]
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def load_model_and_stats():
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    # ZMIANA: tutaj NIE budujemy modelu (bo nie znamy jeszcze input_dim).
    # Model zbudujemy dopiero, gdy features_last_bar zwróci X_last.
    return stats

def predict_one(model, X_last_np):
    x = torch.from_numpy(X_last_np)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
        signal = int(prob >= 0.5)
    return signal, prob

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    norm_stats = load_model_and_stats()
    print("Watcher start. Plik:", MT5_CSV)

    model = None  # Zbudujemy, gdy będziemy znali input_dim z X_last
    last_size = MT5_CSV.stat().st_size if MT5_CSV.exists() else 0
    last_mtime = MT5_CSV.stat().st_mtime if MT5_CSV.exists() else 0.0

    while True:
        try:
            if MT5_CSV.exists():
                st = MT5_CSV.stat()
                if st.st_size != last_size or st.st_mtime != last_mtime:
                    time.sleep(1.0)  # debounce na czas zapisu EA
                    df = load_mt5_csv(MT5_CSV)
                    X_last, bar_idx, sr_wys, sr_szer = features_last_bar(df, MIN_ZP, ILE_ZZ, BACKSTEP, DL_ZZ, LICZBA_SW, norm_stats)
                    if X_last is not None:
                        # Jeśli model jeszcze nie powstał — zbuduj z właściwym input_dim
                        if model is None:
                            input_dim = X_last.shape[1]
                            model = MLP(input_dim, HYPERPARAMS["hidden_sizes"], HYPERPARAMS["output_dim"], HYPERPARAMS["dropout_p"])
                            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
                            model.eval()

                        sig, prob = predict_one(model, X_last)
                        print(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] bar={bar_idx}  SYGNAŁ={sig}  "
                            f"Prawdopodobieństwo={prob:.3f}  Śr. wysokość={sr_wys}  Śr. szerokość={sr_szer}"
                        )
                    else:
                        print("Brak kompletnego wiersza do predykcji.")
                    last_size, last_mtime = st.st_size, st.st_mtime
            else:
                print("MT5 CSV nie istnieje – czekam...")
            time.sleep(5.0)
        except KeyboardInterrupt:
            print("Przerwano.")
            break
        except Exception as e:
            print("Błąd watcher:", e)
            time.sleep(5.0)


# %%
