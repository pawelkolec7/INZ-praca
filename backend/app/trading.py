import pandas as pd

def load_candles(csv_path="model_files/data.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp")
    df.reset_index(drop=True, inplace=True)
    return df


def generate_signal(df):
    # Liczymy 20- i 50-sesyjną średnią kroczącą
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()

    last = df.iloc[-1]

    # Sygnał — MA20 przebija MA50
    if last["MA20"] > last["MA50"]:
        return "KUP"
    elif last["MA20"] < last["MA50"]:
        return "SPRZEDAJ"
    else:
        return "TRZYMAJ"
