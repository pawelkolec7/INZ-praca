# backend/service.py
import csv
import io
import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Iterator

import numpy as np
import pandas as pd
import torch
import MetaTrader5 as mt5

from .schemas import Candle, PredictionResponse, RetrainResponse

from .model_files.config import (
    HYPERPARAMS, DL_ZZ, LICZBA_SW, MODEL_PATH, STATS_PATH,
    MIN_ZP, ILE_ZZ, BACKSTEP, MT5_CSV
)

from .model_files.data_pipeline import (
    load_mt5_csv,
    features_last_bar
)


@dataclass
class SignalService:
    model_root: Path

    def __post_init__(self):
        default_csv = self.model_root / "ohlc_EURUSD_H1.csv"
        self.sample_csv = MT5_CSV if MT5_CSV.exists() else default_csv

        self.model: torch.nn.Module | None = None
        self._input_dim: int | None = None
        self.norm_stats = self._load_norm_stats()

    # ===================== TWOJA LOGIKA WSKAŹNIKÓW =====================

    def _calculate_user_indicators(self, candles: List[Candle]) -> List[Candle]:
        """
        Implementuje DOKŁADNIE Twój kod obliczeniowy na danych z MT5/CSV.
        """
        if not candles:
            return []

        # 1. Przygotowanie DataFrame zgodnego z Twoim kodem (kolumny C, H, L)
        data = []
        for c in candles:
            data.append({
                "C": c.close,
                "H": c.high,
                "L": c.low,
                "O": c.open,  # opcjonalnie, jeśli potrzebne
                "original_obj": c
            })

        df_range = pd.DataFrame(data)

        # === POCZĄTEK TWOJEGO KODU (Dostosowany do zmiennych) ===

        # Parametry (możesz je wyciągnąć do configu)
        sma_short = 5
        sma_mid = 20
        sma_long = 50
        rsi_period = 14
        stoch_k = 14
        stoch_d = 3
        atr_period = 14
        cci_period = 20
        mom_period = 10

        Close = df_range["C"].to_numpy()
        High = df_range["H"].to_numpy()
        Low = df_range["L"].to_numpy()
        N = len(df_range)

        SMA5 = pd.Series(Close).rolling(sma_short, min_periods=1).mean().to_numpy()  # min_periods=1 dla bezpieczeństwa
        SMA20 = pd.Series(Close).rolling(sma_mid, min_periods=1).mean().to_numpy()
        SMA50 = pd.Series(Close).rolling(sma_long, min_periods=1).mean().to_numpy()

        def ema(arr, period):
            a = 2 / (period + 1)
            out = np.full_like(arr, np.nan, dtype=float)
            if len(arr) >= period:
                out[period - 1] = arr[:period].mean()
                for t in range(period, len(arr)):
                    out[t] = a * arr[t] + (1 - a) * out[t - 1]
            # Fill NaN at start for safety in rendering
            mask = np.isnan(out)
            out[mask] = arr[mask]
            return out

        EMA5, EMA20, EMA50 = ema(Close, 5), ema(Close, 20), ema(Close, 50)
        EMA12, EMA26 = ema(Close, 12), ema(Close, 26)
        MACD = EMA12 - EMA26

        Signal = np.full_like(Close, np.nan, dtype=float)
        valid = ~np.isnan(MACD)
        if valid.sum() >= 9:
            idxs = np.flatnonzero(valid)
            start = idxs[0] + 9 - 1
            if start < len(MACD):
                Signal[start] = MACD[idxs[0]:idxs[0] + 9].mean()
                for t in range(start + 1, len(MACD)):
                    Signal[t] = 2 / (9 + 1) * MACD[t] + (1 - 2 / (9 + 1)) * Signal[t - 1]

        rolling20 = pd.Series(Close).rolling(sma_mid, min_periods=1)
        BB_mid = rolling20.mean().to_numpy()
        BB_std = rolling20.std().to_numpy()
        BB_up = BB_mid + 2 * BB_std
        BB_low = BB_mid - 2 * BB_std

        delta = np.diff(Close, prepend=np.nan)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        RSI = np.full_like(Close, np.nan, dtype=float)
        # Uproszczone zabezpieczenie, żeby nie wywaliło błędu na krótkiej historii
        if N > rsi_period:
            avg_gain = gain[1:rsi_period + 1].mean()
            avg_loss = loss[1:rsi_period + 1].mean()
            RS = avg_gain / avg_loss if avg_loss != 0 else np.inf
            RSI[rsi_period] = 100 - 100 / (1 + RS)
            for t in range(rsi_period + 1, N):
                avg_gain = (avg_gain * (rsi_period - 1) + gain[t]) / rsi_period
                avg_loss = (avg_loss * (rsi_period - 1) + loss[t]) / rsi_period
                RS = avg_gain / avg_loss if avg_loss != 0 else np.inf
                RSI[t] = 100 - 100 / (1 + RS)

        StochK = np.full_like(Close, np.nan, dtype=float)
        StochD = np.full_like(Close, np.nan, dtype=float)
        if N >= stoch_k:
            low_min = pd.Series(Low).rolling(stoch_k, min_periods=stoch_k).min().to_numpy()
            high_max = pd.Series(High).rolling(stoch_k, min_periods=stoch_k).max().to_numpy()
            # Zabezpieczenie przed dzieleniem przez zero
            denom = high_max - low_min
            rawK = np.divide(100 * (Close - low_min), denom, out=np.zeros_like(Close), where=denom != 0)
            StochK[stoch_k - 1:] = rawK[stoch_k - 1:]
            StochD = pd.Series(StochK).rolling(stoch_d, min_periods=1).mean().to_numpy()

        TR = np.maximum.reduce([
            High - Low,
            np.abs(High - np.concatenate([[np.nan], Close[:-1]])),
            np.abs(Low - np.concatenate([[np.nan], Close[:-1]])),
        ])
        ATR = pd.Series(TR).rolling(atr_period, min_periods=1).mean().to_numpy()

        TP = (High + Low + Close) / 3.0
        SMA_TP = pd.Series(TP).rolling(cci_period, min_periods=1).mean().to_numpy()
        MD = pd.Series(np.abs(TP - SMA_TP)).rolling(cci_period, min_periods=1).mean().to_numpy()
        Den = 0.015 * MD
        CCI = np.divide(TP - SMA_TP, Den, out=np.zeros_like(TP, dtype=float), where=Den != 0)

        Momentum = np.full_like(Close, np.nan, dtype=float)
        if N > 10:
            Momentum[10:] = Close[10:] - Close[:-10]

        # === KONIEC TWOJEGO KODU ===

        # Przypisanie wyników z powrotem do obiektów Candle
        df_range["SMA5"] = SMA5
        df_range["SMA20"] = SMA20
        df_range["SMA50"] = SMA50
        df_range["EMA5"] = EMA5
        df_range["EMA20"] = EMA20
        df_range["EMA50"] = EMA50
        df_range["BB_mid"] = BB_mid
        df_range["BB_up"] = BB_up
        df_range["BB_low"] = BB_low
        df_range["MACD"] = MACD
        df_range["Signal"] = Signal
        df_range["RSI"] = RSI
        df_range["StochK"] = StochK
        df_range["StochD"] = StochD
        df_range["ATR"] = ATR
        df_range["CCI"] = CCI
        df_range["Momentum"] = Momentum

        # Wypełnienie NaN zerami lub None (żeby JSON był poprawny)
        df_range = df_range.where(pd.notnull(df_range), None)

        updated_candles = []
        for i, row in df_range.iterrows():
            c = row["original_obj"]
            # Aktualizacja pól w obiekcie Candle
            c.SMA5 = row["SMA5"]
            c.SMA20 = row["SMA20"]
            c.SMA50 = row["SMA50"]
            c.EMA5 = row["EMA5"]
            c.EMA20 = row["EMA20"]
            c.EMA50 = row["EMA50"]
            c.BB_mid = row["BB_mid"]
            c.BB_up = row["BB_up"]
            c.BB_low = row["BB_low"]
            c.MACD = row["MACD"]
            c.Signal = row["Signal"]
            c.RSI = row["RSI"]
            c.StochK = row["StochK"]
            c.StochD = row["StochD"]
            c.ATR = row["ATR"]
            c.CCI = row["CCI"]
            c.Momentum = row["Momentum"]
            updated_candles.append(c)

        return updated_candles

    # ===================== Integracja MT5 =====================

    def _fetch_live_candle(self) -> Optional[Candle]:
        try:
            if not mt5.initialize():
                print(f"MT5 init failed: {mt5.last_error()}")
                return None

            rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1)
            mt5.shutdown()

            if rates is None or len(rates) == 0:
                return None

            r = rates[0]
            dt = datetime.fromtimestamp(r['time'], tz=timezone.utc)

            return Candle(
                time=dt.isoformat(),
                open=float(r['open']),
                high=float(r['high']),
                low=float(r['low']),
                close=float(r['close'])
            )
        except Exception as e:
            print(f"Error fetching live candle: {e}")
            return None

    # ===================== Historia =====================

    def _parse_csv_text(self, text: str) -> List[Candle]:
        candles: List[Candle] = []
        reader = csv.reader(io.StringIO(text))

        for row in reader:
            if len(row) < 6:
                continue
            try:
                dt = datetime.fromisoformat(row[0])
                ts_str = dt.isoformat()
            except ValueError:
                ts_str = row[0] or None

            try:
                o, h, l, c = map(float, row[2:6])
            except ValueError:
                continue

            candles.append(Candle(time=ts_str, open=o, high=h, low=l, close=c))

        candles.reverse()
        return candles

    def load_history(self, limit: int = 10000) -> List[Candle]:
        history: List[Candle] = []

        if self.sample_csv.exists():
            text = self.sample_csv.read_text(encoding="utf-8", errors="ignore")
            history = self._parse_csv_text(text)

        live_candle = self._fetch_live_candle()

        if history:
            history.pop()

        if live_candle:
            history.append(live_candle)

        # OBLICZANIE WSKAŹNIKÓW TWOIM KODEM
        full_history = self._calculate_user_indicators(history)

        return full_history[-limit:] if limit and len(full_history) > limit else full_history

    def load_sample_candles(self, limit: int = 12) -> List[Candle]:
        return self.load_history(limit)

    def load_history_from_bytes(self, content: bytes, limit: int | None = 10000) -> List[Candle]:
        if not content:
            raise ValueError("Plik CSV jest pusty")

        text = content.decode("utf-8", errors="ignore")
        history = self._parse_csv_text(text)
        if not history:
            raise ValueError("Nie znaleziono świec w pliku CSV")

        full_history = self._calculate_user_indicators(history)
        return full_history[-limit:] if limit and len(full_history) > limit else full_history

    # ===================== Model + Douczanie =====================

    class MLP(torch.nn.Module):
        def __init__(self, input_dim, hidden_sizes, output_dim, dropout_p=0.5):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_sizes:
                layers += [
                    torch.nn.Linear(prev, h),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_p),
                ]
                prev = h
            layers += [torch.nn.Linear(prev, output_dim)]
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def _load_norm_stats(self):
        if not STATS_PATH.exists():
            if not MODEL_PATH.exists(): pass
            return {}
        with STATS_PATH.open("rb") as f:
            return pickle.load(f)

    def _ensure_model(self, input_dim: int):
        if self.model is None or self._input_dim != input_dim:
            self.model = self.MLP(
                input_dim=input_dim,
                hidden_sizes=HYPERPARAMS["hidden_sizes"],
                output_dim=HYPERPARAMS["output_dim"],
                dropout_p=HYPERPARAMS["dropout_p"],
            )
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Brak pliku modelu: {MODEL_PATH}")
            state = torch.load(MODEL_PATH, map_location="cpu")
            self.model.load_state_dict(state)
            self.model.eval()
            self._input_dim = input_dim

    def retrain_model_stream(self) -> Iterator[str]:
        train_script = self.model_root / "train_model.py"
        if not train_script.exists():
            yield f"BŁĄD: Nie znaleziono pliku: {train_script}\n"
            return

        try:
            process = subprocess.Popen(
                [sys.executable, "-u", str(train_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.model_root,
                bufsize=1
            )
            for line in process.stdout:
                yield line
            process.wait()

            if process.returncode == 0:
                self.model = None
                self.norm_stats = self._load_norm_stats()
                yield "[SYSTEM] Sukces. Model przeładowany.\n"
        except Exception as e:
            yield f"\n[SYSTEM] Błąd: {str(e)}\n"

    # ===================== Pipeline predykcji =====================

    def _predict_from_df(self, df_prices: pd.DataFrame) -> PredictionResponse:
        if len(df_prices) < 2:
            raise ValueError("Za mało danych do predykcji")
        if not self.norm_stats:
            self.norm_stats = self._load_norm_stats()

        X_last, bar_idx, sr_wys, sr_szer = features_last_bar(
            df_prices,
            MIN_ZP,
            ILE_ZZ,
            BACKSTEP,
            DL_ZZ,
            LICZBA_SW,
            self.norm_stats,
        )

        if X_last is None:
            raise ValueError("Brak kompletnego wiersza do predykcji")

        input_dim = X_last.shape[1]
        self._ensure_model(input_dim)

        x = torch.from_numpy(X_last)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()
        recommendation = "buy" if prob >= 0.5 else "sell"
        return PredictionResponse(
            probability=prob,
            recommendation=recommendation,
            avg_pivot_height_pips=sr_wys,
            avg_pivot_width_bars=sr_szer,
        )

    def _df_from_candles(self, candles: List[Candle]) -> pd.DataFrame:
        if len(candles) < 2:
            raise ValueError("Za mało świec")
        data = [(c.open, c.high, c.low, c.close) for c in candles]
        df = pd.DataFrame(data, columns=["O", "H", "L", "C"])
        df.index = range(1, len(df) + 1)
        return df

    def _df_from_bytes_mt5(self, content: bytes) -> pd.DataFrame:
        from io import StringIO
        raw = pd.read_csv(StringIO(content.decode("utf-8", errors="ignore")), header=None)
        best = None
        for start in range(0, max(0, raw.shape[1] - 3)):
            try:
                seg = raw.iloc[:, start:start + 4].astype(float)
            except Exception:
                continue
            O, H, L, C = seg.iloc[:, 0], seg.iloc[:, 1], seg.iloc[:, 2], seg.iloc[:, 3]
            ok = ((H >= O) & (H >= C) & (L <= O) & (L <= C)).mean()
            if best is None or ok > best[0]: best = (ok, seg)
        if best is None: raise ValueError("Błąd formatu CSV.")
        df = best[1].copy()
        df.columns = ["O", "H", "L", "C"]
        df = df.iloc[::-1].reset_index(drop=True)
        df.index = range(1, len(df) + 1)
        return df

    def predict_direction(self, candles: List[Candle]) -> PredictionResponse:
        df = self._df_from_candles(candles)
        return self._predict_from_df(df)

    def predict_from_history(self) -> PredictionResponse:
        candles = self.load_history(limit=0)
        if len(candles) < 100: raise ValueError("Za mało danych")
        df = self._df_from_candles(candles)
        return self._predict_from_df(df)

    def predict_from_upload(self, content: bytes) -> PredictionResponse:
        df = self._df_from_bytes_mt5(content)
        return self._predict_from_df(df)