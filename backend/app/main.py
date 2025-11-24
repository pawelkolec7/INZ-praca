# backend/main.py
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse  # <--- Ważny import do streamingu

from .schemas import (
    Candle,
    HistoryResponse,
    PredictionRequest,
    PredictionResponse,
    SampleResponse,
    RetrainResponse
)
from .service import SignalService


app = FastAPI(title="Trade Signal API", version="0.3.2-streaming")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ścieżka do folderu model_files (obok main.py)
MODEL_ROOT = Path(__file__).resolve().parent / "model_files"

print(f"DEBUG: MODEL_ROOT is set to: {MODEL_ROOT}")

service = SignalService(model_root=MODEL_ROOT)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/sample", response_model=SampleResponse)
def sample(limit: int = 12) -> SampleResponse:
    candles = service.load_sample_candles(limit)
    if not candles:
        raise HTTPException(status_code=404, detail="Sample data unavailable")
    return SampleResponse(candles=candles)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if len(request.candles) < 5:
        raise HTTPException(status_code=400, detail="At least 5 candles are required")
    try:
        result = service.predict_direction(request.candles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/history", response_model=HistoryResponse)
def history(limit: int = 10000) -> HistoryResponse:
    candles = service.load_history(limit)
    if not candles:
        raise HTTPException(status_code=404, detail="History unavailable")
    return HistoryResponse(candles=candles, total=len(candles))


@app.get("/predict/from-file", response_model=PredictionResponse)
def predict_from_file() -> PredictionResponse:
    try:
        return service.predict_from_history()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/history/upload", response_model=HistoryResponse)
async def history_upload(limit: int = 10000, file: UploadFile = File(...)) -> HistoryResponse:
    try:
        content = await file.read()
        candles = service.load_history_from_bytes(content, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return HistoryResponse(candles=candles, total=len(candles))


@app.post("/predict/from-upload", response_model=PredictionResponse)
async def predict_from_upload(file: UploadFile = File(...)) -> PredictionResponse:
    try:
        content = await file.read()
        return service.predict_from_upload(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/retrain")
def retrain():
    """
    Uruchamia proces ponownego treningu modelu na serwerze.
    Zwraca StreamingResponse, aby frontend mógł widzieć logi na bieżąco.
    """
    # Przekazujemy generator z serwisu bezpośrednio do odpowiedzi HTTP
    return StreamingResponse(service.retrain_model_stream(), media_type="text/plain")


# ZMODYFIKOWANA FUNKCJA: DODANO WERYFIKACJĘ BACKSTEP PRZY TWORZENIU KANDYDATÓW NA PIVOTY
def zigzag_basic(candles, depth=23, deviation=0.00554, backstep=4):
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    pivots_high = []
    pivots_low = []

    # 'backstep' nie wpływa na zakres pętli, wpływa na logikę filtrowania poniżej
    start_index = depth
    end_index = len(candles) - depth

    for i in range(start_index, end_index):
        # Okno dla warunku Peak/Trough (zgodne z Depth)
        window_h = highs[i - depth: i + depth + 1]
        window_l = lows[i - depth: i + depth + 1]

        # Standardowy warunek ZigZag (Depth + Deviation)
        is_pivot_high = highs[i] == max(window_h) and \
                        (highs[i] - min(window_l)) >= deviation

        is_pivot_low = lows[i] == min(window_l) and \
                       (max(window_h) - lows[i]) >= deviation

        if is_pivot_high:
            pivots_high.append({
                "index": i,
                "time": candles[i].time,
                "price": highs[i],
                "type": "high"
            })

        if is_pivot_low:
            pivots_low.append({
                "index": i,
                "time": candles[i].time,
                "price": lows[i],
                "type": "low"
            })

    # --- SCALENIE I SORTOWANIE ---
    all_pivots = pivots_high + pivots_low
    all_pivots.sort(key=lambda x: x["index"])

    # --- FILTR NAPRZEMIENNY (MT5) Z WERYFIKACJĄ BACKSTEP ---
    clean = []
    last_type = None

    for p in all_pivots:
        # Pusty zbiór lub zmiana typu pivota
        if last_type is None or p["type"] != last_type:
            # Warunek Backstep: Sprawdź, czy pivot jest wystarczająco daleko od OSTATNIEGO piku
            if not clean or (p["index"] - clean[-1]["index"]) >= backstep:
                clean.append(p)
                last_type = p["type"]
            else:
                if p["type"] == "low":
                    if p["price"] < clean[-1]["price"]:
                        clean[-1] = p
                # HIGH: wybieramy wyższy
                else:
                    if p["price"] > clean[-1]["price"]:
                        clean[-1] = p

        # Jeśli są dwa pivoty tego samego typu z rzędu (i["type"] == last_type)
        else:
            # Jeśli są dwa LOW z rzędu → wybierz "lepszy"
            # LOW: wybieramy ten niższy
            if p["type"] == "low":
                if p["price"] < clean[-1]["price"]:
                    clean[-1] = p
            # HIGH: wybieramy wyższy
            else:
                if p["price"] > clean[-1]["price"]:
                    clean[-1] = p

    # --- NOWA LOGIKA FILTROWANIA Z UWZGLĘDNIENIEM BACKSTEP ---
    # Musimy zacząć od nowa, ponieważ obecna logika nie wspiera backstepu przy zmianie typu.

    clean_with_backstep = []

    for p in all_pivots:
        if not clean_with_backstep:
            clean_with_backstep.append(p)
            continue

        last_pivot = clean_with_backstep[-1]

        # 1. Pivot tego samego typu: aktualizuj, jeśli jest bardziej ekstremalny
        if p["type"] == last_pivot["type"]:
            if (p["type"] == "low" and p["price"] < last_pivot["price"]) or \
                    (p["type"] == "high" and p["price"] > last_pivot["price"]):
                clean_with_backstep[-1] = p

        # 2. Pivot przeciwnego typu: dodaj tylko, jeśli spełnia warunek Backstep
        elif p["type"] != last_pivot["type"]:
            if (p["index"] - last_pivot["index"]) >= backstep:
                clean_with_backstep.append(p)
            else:
                # Jeśli jest za blisko, ignoruj go (nie jest bardziej ekstremalny,
                # bo to nowy pivot, nie może zastąpić starego, bo ma inny typ)
                pass

    return pivots_high, pivots_low, clean_with_backstep


# ZMODYFIKOWANY ENDPOINT: Dodano parametr 'backstep'
@app.get("/zigzag-test")
def zigzag_test(depth: int = 23, deviation: float = 0.00554, backstep: int = 4):
    candles = service.load_history(3000)

    pivots_high, pivots_low, zigzag = zigzag_basic(
        candles, depth, deviation, backstep  # Przekazanie backstep
    )

    return {
        "total_candles": len(candles),
        "pivot_highs": pivots_high,
        "pivot_lows": pivots_low,
        "count_highs": len(pivots_high),
        "count_lows": len(pivots_low),
        "zigzag": zigzag  # <-- FRONTEND używa tylko TEGO
    }