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
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path

# ================== Wczytanie CSV z MT5 (EA) ====================
def load_mt5_csv(path: Path) -> pd.DataFrame:
    """
    Obsługuje plik z EA:
    [opcjonalnie: datetime_str],[opcjonalnie: timestamp],O,H,L,C,[opcjonalnie: tick_volume]
    Wyszukuje 4 sąsiednie kolumny będące O,H,L,C.
    """
    raw = pd.read_csv(path, header=None)
    best = None
    for start in range(0, max(0, raw.shape[1]-3)):
        try:
            seg = raw.iloc[:, start:start+4].astype(float)
        except Exception:
            continue
        O, H, L, C = seg.iloc[:,0], seg.iloc[:,1], seg.iloc[:,2], seg.iloc[:,3]
        ok = ((H>=O)&(H>=C)&(L<=O)&(L<=C)).mean()
        if best is None or ok > best[0]:
            best = (ok, seg)
    if best is None:
        raise ValueError("Nie znaleziono kolumn O,H,L,C w MT5 CSV.")
    df = best[1].copy()
    df.columns = ["O","H","L","C"]
    df = df.iloc[::-1].reset_index(drop=True)
    df.index = range(1, len(df)+1)  # 1..N
    return df

# ====================== ZigZag / etykiety =======================
def ZZ_caly(df_swieczki, ileZZ, minzmnpkt, backstep):
    H = df_swieczki["H"].values
    L = df_swieczki["L"].values
    prevH, prevL = deque(), deque()
    pivots = []
    last_pivot = None
    last_n = 0
    for i, (h, l) in enumerate(zip(H, L)):
        bar = i + 1
        if len(prevH) == ileZZ:
            local_max = max(prevH)
            local_min = min(prevL)
            isPeak   = (h >= local_max)
            isTrough = (l <= local_min)
            if isPeak ^ isTrough:
                typ   = 1 if isPeak else -1
                price = h if isPeak else l
                if last_n == 0:
                    pivots.append((bar, price, typ))
                    last_pivot = (bar, price, typ)
                    last_n = 1
                else:
                    lb, lp_price, lp_typ = last_pivot
                    if lp_typ * typ == -1:
                        if abs(price - lp_price) > minzmnpkt and (bar - lb) >= backstep:
                            pivots.append((bar, price, typ))
                            last_pivot = (bar, price, typ)
                            last_n += 1
                    else:
                        if (price - lp_price)*typ > 0:
                            pivots[-1] = (bar, price, typ)
                            last_pivot = (bar, price, typ)
        prevH.append(h); prevL.append(l)
        if len(prevH) > ileZZ:
            prevH.popleft(); prevL.popleft()
    return pd.DataFrame(pivots, columns=["bar", "price", "type"])

def zigzag_v5_py(bar, last_row, last_count, H, L, ileZZ, minzp, backstep):
    idx = bar - 1
    start = max(0, idx - ileZZ)
    segH, segL = H[start:idx+1], L[start:idx+1]
    is_peak   = int(H[idx] >= segH.max())
    is_trough = int(L[idx] <= segL.min())
    if is_peak + is_trough != 1:
        return last_row, last_count, 0
    typ   = 1 if is_peak else -1
    price = H[idx] if is_peak else L[idx]
    new_row = (bar, price, typ)
    if last_count == 0:
        return new_row, 1, 1
    prev_bar, prev_price, prev_typ = last_row
    if prev_typ * typ == -1:
        if abs(price - prev_price) > minzp and (bar - prev_bar) >= backstep:
            return new_row, last_count + 1, 1
        else:
            return last_row, last_count, 0
    else:
        if (price - prev_price) * typ > 0:
            return new_row, last_count, 0
        else:
            return last_row, last_count, 0

def etykieta(sw, df_zz_all, O:pd.Series):
    try:
        next_open = O.loc[sw+1]
    except KeyError:
        return np.nan, np.nan
    mask = df_zz_all["bar"] > sw
    if not mask.any():
        return np.nan, np.nan
    piv = df_zz_all.loc[mask, :].iloc[0]
    pivot_bar   = float(piv["bar"])
    pivot_price = float(piv["price"])
    label = 1.0 if pivot_price > next_open else 0.0
    return label, pivot_bar

# ============== Budowa cech (zakres lub 1 wiersz) ==============
def oblicz_dane(df_swieczki, df_zz_all, od, do, dl_zz, liczba_sw, ile_zz, min_zp, backstep):
    H = df_swieczki["H"].values
    L = df_swieczki["L"].values
    C = df_swieczki["C"]

    rows_input, rows_output = [], []
    piv_deque = deque(maxlen=dl_zz+1)
    last_row, last_count = None, 0

    for sw in range(od, do+1):
        new_row, new_count, flag = zigzag_v5_py(sw, last_row, last_count, H, L, ile_zz, min_zp, backstep)
        if flag == 1 and last_row is not None:
            piv_deque.append(last_row)
        last_row, last_count = new_row, new_count

        all_dyn = list(piv_deque)
        if last_row is not None:
            all_dyn.append(last_row)

        needed = dl_zz + 1
        slice_piv = all_dyn[-needed:] if len(all_dyn) >= needed else all_dyn
        prices = [p[1] for p in slice_piv]
        if len(prices) < needed:
            prices = [np.nan]*(needed-len(prices)) + prices

        closes = []
        for j in range(liczba_sw-1, -1, -1):
            idx = sw - j
            closes.append(C.loc[idx] if idx >= 1 else np.nan)

        rows_input.append(prices + closes)
        lab, piv_bar = etykieta(sw, df_zz_all, df_swieczki["O"])
        rows_output.append([lab, piv_bar])

    cols_input = [f"ZZ_price_{k+1}" for k in range(dl_zz+1)] + [f"C_close_{k+1}" for k in range(liczba_sw)]
    df_input  = pd.DataFrame(rows_input,  index=range(od, do+1), columns=cols_input)
    df_output = pd.DataFrame(rows_output, index=range(od, do+1), columns=["label","pivot_bar"])

    valid = df_input.notna().all(axis=1) & df_output["label"].notna()
    df_wej_f = df_input.loc[valid].copy().reset_index(drop=True)
    df_wyj_f = df_output.loc[valid].copy().reset_index(drop=True)

    pierwsza_sw = od + ( (~valid).cumsum().iloc[-1] if len(valid) else 0 )
    return df_wej_f, df_wyj_f, pierwsza_sw

# ========================= Wskaźniki ============================
def oblicz_wskazniki(df_swieczki, od, do, m):
    sma_short, sma_mid, sma_long = 5, 20, 50
    rsi_period = 14
    stoch_k, stoch_d = 14, 3
    atr_period = 14
    cci_period = 20
    mom_period = 10

    df_range = df_swieczki.loc[od:do].copy()
    Close = df_range["C"].to_numpy()
    High  = df_range["H"].to_numpy()
    Low   = df_range["L"].to_numpy()
    N = len(df_range)

    SMA5  = pd.Series(Close).rolling(sma_short, min_periods=sma_short).mean().to_numpy()
    SMA20 = pd.Series(Close).rolling(sma_mid  , min_periods=sma_mid  ).mean().to_numpy()
    SMA50 = pd.Series(Close).rolling(sma_long , min_periods=sma_long ).mean().to_numpy()

    def ema(arr, period):
        a = 2/(period+1)
        out = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            out[period-1] = arr[:period].mean()
            for t in range(period, len(arr)):
                out[t] = a*arr[t] + (1-a)*out[t-1]
        return out

    EMA5, EMA20, EMA50 = ema(Close,5), ema(Close,20), ema(Close,50)
    EMA12, EMA26 = ema(Close, 12), ema(Close, 26)
    MACD = EMA12 - EMA26

    Signal = np.full_like(Close, np.nan, dtype=float)
    valid = ~np.isnan(MACD)
    if valid.sum() >= 9:
        idxs = np.flatnonzero(valid)
        start = idxs[0] + 9 - 1
        Signal[start] = MACD[idxs[0]:idxs[0]+9].mean()
        for t in range(start+1, len(MACD)):
            Signal[t] = 2/(9+1)*MACD[t] + (1-2/(9+1))*Signal[t-1]

    rolling20 = pd.Series(Close).rolling(sma_mid, min_periods=sma_mid)
    BB_mid = rolling20.mean().to_numpy()
    BB_std = rolling20.std().to_numpy()
    BB_up  = BB_mid + 2*BB_std
    BB_low = BB_mid - 2*BB_std

    delta = np.diff(Close, prepend=np.nan)
    gain  = np.where(delta>0, delta, 0)
    loss  = np.where(delta<0, -delta, 0)

    RSI = np.full_like(Close, np.nan, dtype=float)
    if N > rsi_period:
        avg_gain = gain[1:rsi_period+1].mean()
        avg_loss = loss[1:rsi_period+1].mean()
        RS = avg_gain/avg_loss if avg_loss!=0 else np.inf
        RSI[rsi_period] = 100 - 100/(1+RS)
        for t in range(rsi_period+1, N):
            avg_gain = (avg_gain*(rsi_period-1) + gain[t]) / rsi_period
            avg_loss = (avg_loss*(rsi_period-1) + loss[t]) / rsi_period
            RS = avg_gain/avg_loss if avg_loss!=0 else np.inf
            RSI[t] = 100 - 100/(1+RS)

    StochK = np.full_like(Close, np.nan, dtype=float)
    StochD = np.full_like(Close, np.nan, dtype=float)
    if N >= stoch_k:
        low_min  = pd.Series(Low).rolling(stoch_k, min_periods=stoch_k).min().to_numpy()
        high_max = pd.Series(High).rolling(stoch_k, min_periods=stoch_k).max().to_numpy()
        rawK = 100*(Close - low_min)/(high_max - low_min)
        StochK[stoch_k-1:] = rawK[stoch_k-1:]
        StochD = pd.Series(StochK).rolling(stoch_d, min_periods=stoch_d).mean().to_numpy()

    TR = np.maximum.reduce([
        High - Low,
        np.abs(High - np.concatenate([[np.nan], Close[:-1]])),
        np.abs(Low  - np.concatenate([[np.nan], Close[:-1]])),
    ])
    ATR = pd.Series(TR).rolling(atr_period, min_periods=1).mean().to_numpy()

    TP = (High + Low + Close)/3.0
    SMA_TP = pd.Series(TP).rolling(cci_period, min_periods=1).mean().to_numpy()
    MD = pd.Series(np.abs(TP - SMA_TP)).rolling(cci_period, min_periods=1).mean().to_numpy()
    Den = 0.015 * MD
    CCI = np.divide(TP - SMA_TP, Den, out=np.zeros_like(TP, dtype=float), where=Den != 0)

    Momentum = np.full_like(Close, np.nan, dtype=float)
    if N > 10:
        Momentum[10:] = Close[10:] - Close[:-10]

    Indicators = np.column_stack([
        SMA5, SMA20, SMA50,
        EMA5, EMA20, EMA50,
        BB_mid, BB_up, BB_low,
        MACD, Signal,
        RSI,
        StochK, StochD,
        ATR,
        CCI,
        Momentum
    ])
    df_ind = pd.DataFrame(
        Indicators, index=df_range.index,
        columns=["SMA5","SMA20","SMA50","EMA5","EMA20","EMA50",
                 "BB_mid","BB_up","BB_low","MACD","Signal","RSI",
                 "StochK","StochD","ATR","CCI","Momentum"]
    )
    return df_ind.tail(m).reset_index(drop=True).fillna(0.0)

def polacz_dane(df_ceny, df_wskazniki):
    if len(df_ceny) != len(df_wskazniki):
        raise ValueError(f"Liczba wierszy się nie zgadza: {len(df_ceny)} vs {len(df_wskazniki)}")
    return pd.concat([df_ceny, df_wskazniki], axis=1, ignore_index=True)

# ========================= Normalizacja =========================
def normalizacja2(W:pd.DataFrame, O:pd.Series, dl_zz:int, liczba_sw:int, stats=None):
    W = W.copy().reset_index(drop=True).astype(float)
    O = O.reset_index(drop=True).astype(float)

    wsp = 100.0
    n_base = dl_zz + liczba_sw + 10

    piv = [f"ZZ_price_{i+1}" for i in range(dl_zz+1)]
    clo = [f"C_close_{i+1}" for i in range(liczba_sw)]
    ind = ["SMA5","SMA20","SMA50","EMA5","EMA20","EMA50",
           "BB_mid","BB_up","BB_low","MACD","Signal","RSI",
           "StochK","StochD","ATR","CCI","Momentum"]
    W.columns = piv + clo + ind

    W.iloc[:, :n_base] = (W.iloc[:, :n_base].values - O.values.reshape(-1,1)) * wsp

    if stats is None:
        stats = {}
        for col in ["MACD","Signal","Momentum"]:
            mu, sd = W[col].mean(), W[col].std(ddof=0) or 1.0
            stats[col] = (mu, sd)
        mu, sd = (W["ATR"]/O).mean(), (W["ATR"]/O).std(ddof=0) or 1.0
        stats["ATR_ratio"] = (mu, sd)

    W["MACD"]     = (W["MACD"]   - stats["MACD"][0])    / stats["MACD"][1]
    W["Signal"]   = (W["Signal"] - stats["Signal"][0])  / stats["Signal"][1]
    W["Momentum"] = (W["Momentum"]- stats["Momentum"][0])/ stats["Momentum"][1]
    W["RSI"]      = W["RSI"] / 100.0
    for col in ["StochK","StochD"]:
        W[col] = (W[col] - 50.0) / 100.0
    W["ATR"] = ((W["ATR"]/O) - stats["ATR_ratio"][0]) / stats["ATR_ratio"][1]
    W["CCI"] = W["CCI"] / 100.0

    return W, stats

# ========== Helper: features dla OSTATNIEJ zamkniętej świecy ==========
def features_last_bar(df_swieczki, MIN_ZP, ILE_ZZ, BACKSTEP, DL_ZZ, LICZBA_SW, norm_stats):
    N = len(df_swieczki)
    if N < 2:
        return None, None

    # 1) budujemy pipeline na zakresie 1..N-1 (bo potrzebujemy O_next dla sw)
    od, do = 1, N - 1

    df_ZZ = ZZ_caly(df_swieczki, ILE_ZZ, MIN_ZP, BACKSTEP)
    df_in, df_out, first_bar = oblicz_dane(df_swieczki, df_ZZ, od, do, DL_ZZ, LICZBA_SW, ILE_ZZ, MIN_ZP, BACKSTEP)
    if len(df_in) == 0:
        return None, None

    df_ind = oblicz_wskazniki(df_swieczki, od, do, len(df_in))
    df_all = polacz_dane(df_in, df_ind)

    # 2) indeks świec = real bars, O_next dostępne z shift(-1)
    sw_index = np.arange(first_bar, do + 1)
    df_all.index = sw_index

    O_next_series = df_swieczki["O"].shift(-1)  # O_next istnieje tylko do N-1
    ready_idx = df_all.index.intersection(O_next_series.dropna().index)
    if len(ready_idx) == 0:
        return None, None

    # 3) idziemy od końca i bierzemy pierwszy „gotowy”
    for bar_idx in sorted(ready_idx, reverse=True):
        X_raw = df_all.loc[[bar_idx]].copy()
        O_next = O_next_series.loc[[bar_idx]]
        X_norm, _ = normalizacja2(X_raw, O_next, DL_ZZ, LICZBA_SW, stats=norm_stats)
        if not X_norm.isna().any().any():
            return X_norm.to_numpy(np.float32), int(bar_idx)

    return None, None



# %%
