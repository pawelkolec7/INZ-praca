
import pickle, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from config import *
from data_pipeline import *

torch.manual_seed(SEED); np.random.seed(SEED)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, dropout_p=0.5):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_p)]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def accuracy(y_logits, y_true):
    preds = torch.sigmoid(y_logits)
    preds_class = (preds >= 0.5).float()
    return (preds_class.squeeze() == y_true.squeeze()).float().mean().item()

def evaluate(model, X, y, crit, bs):
    X = torch.from_numpy(X); y = torch.from_numpy(y).unsqueeze(1)
    ld = DataLoader(TensorDataset(X,y), batch_size=bs, shuffle=False)
    model.eval(); loss_sum=0; acc_sum=0; n=0
    with torch.no_grad():
        for xb, yb in ld:
            out = model(xb); loss = crit(out, yb)
            k = xb.size(0); loss_sum += loss.item()*k; acc_sum += accuracy(out, yb)*k; n += k
    return loss_sum/n, acc_sum/n

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_mt5_csv(MT5_CSV)
    N = len(df); od, do = 1, N

    df_ZZ = ZZ_caly(df, ILE_ZZ, MIN_ZP, BACKSTEP)
    df_in, df_out, first_bar = oblicz_dane(df, df_ZZ, od, do, DL_ZZ, LICZBA_SW, ILE_ZZ, MIN_ZP, BACKSTEP)
    df_ind = oblicz_wskazniki(df, od, do, len(df_in))
    df_all = polacz_dane(df_in, df_ind)

    # indeks świec = real bars
    sw_index = np.arange(first_bar, do+1)
    df_all.index = sw_index; df_out.index = sw_index

    # ---------------- FIT / VALID (90/10 jeśli TRAIN_FRACTION=0.9) ----------------
    m = len(df_all)
    cut = int(round(m * 0.9))
    fit_end_bar = sw_index[cut-1]

    X_fit_raw = df_all.iloc[:cut].copy();   y_fit_df  = df_out.iloc[:cut].copy()
    X_val_raw = df_all.iloc[cut:].copy();   y_val_df  = df_out.iloc[cut:].copy()

    # purging tylko po stronie FIT
    if "pivot_bar" in y_fit_df.columns:
        mask = (y_fit_df["pivot_bar"] <= fit_end_bar)
        X_fit_raw = X_fit_raw.loc[mask]; y_fit_df = y_fit_df.loc[mask]

    # O_next przez shift(-1) + filtrowanie braków
    O_next_series   = df["O"].shift(-1)  # O następnej świecy
    fit_idx_valid   = X_fit_raw.index.intersection(O_next_series.dropna().index)
    valid_idx_valid = X_val_raw.index.intersection(O_next_series.dropna().index)

    X_fit_raw = X_fit_raw.loc[fit_idx_valid]
    y_fit_df  = y_fit_df.loc[fit_idx_valid]
    X_val_raw = X_val_raw.loc[valid_idx_valid]
    y_val_df  = y_val_df.loc[valid_idx_valid]

    O_fit = O_next_series.loc[fit_idx_valid]
    O_val = O_next_series.loc[valid_idx_valid]

    # normalizacja: fit na FIT, transform na VALID
    X_fit, stats = normalizacja2(X_fit_raw, O_fit, DL_ZZ, LICZBA_SW, stats=None)
    X_val, _     = normalizacja2(X_val_raw,  O_val, DL_ZZ, LICZBA_SW, stats=stats)

    X_fit = X_fit.to_numpy(np.float32); X_val = X_val.to_numpy(np.float32)
    y_fit = y_fit_df["label"].values.astype(np.float32)
    y_val = y_val_df["label"].values.astype(np.float32)

    model = MLP(X_fit.shape[1], HYPERPARAMS["hidden_sizes"], HYPERPARAMS["output_dim"], HYPERPARAMS["dropout_p"])
    crit = nn.BCEWithLogitsLoss()
    opt  = optim.Adam(model.parameters(), lr=HYPERPARAMS["learning_rate"], weight_decay=HYPERPARAMS["weight_decay"])

    ds = TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit).unsqueeze(1))
    ld = DataLoader(ds, batch_size=HYPERPARAMS["batch_size"], shuffle=True)

    # --------- EARLY STOPPING (monitor: Valid loss – mniejszy lepszy) ----------
    patience = 10  # <- możesz podnieść/obniżyć
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(HYPERPARAMS["epochs"]):
        model.train(); runl=runacc=n=0
        for xb,yb in ld:
            opt.zero_grad(); out=model(xb); loss=crit(out,yb); loss.backward(); opt.step()
            k=xb.size(0); runl+=loss.item()*k; n+=k
            with torch.no_grad(): runacc += accuracy(out,yb)*k
        tr_loss=runl/n; tr_acc=runacc/n

        va_loss, va_acc = evaluate(model, X_val, y_val, crit, HYPERPARAMS["batch_size"])
        print(f"[{epoch+1}/{HYPERPARAMS['epochs']}] Fit loss {tr_loss:.4f}, acc {tr_acc:.4f} | Valid loss {va_loss:.4f}, acc {va_acc:.4f}")

        # check improvement
        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping po {epoch+1} epokach (brak poprawy {patience} epok). "
                      f"Najlepszy Valid loss: {best_val_loss:.4f}")
                break

    # przywróć najlepsze wagi (gdy early stop zadziałał lub po treningu)
    if best_state is not None:
        model.load_state_dict(best_state)

    # zapisz model i statystyki
    torch.save(model.state_dict(), MODEL_PATH)
    with open(STATS_PATH, "wb") as f:
        pickle.dump(stats, f)
    print(f"Zapisano: {MODEL_PATH} oraz {STATS_PATH}")


# %%
