import React, { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

const toTimestamp = (time, fallback) => {
  const ts = Date.parse(time)
  return Number.isNaN(ts) ? fallback : Math.floor(ts / 1000)
}

// Konfiguracja dostępnych wskaźników
const INDICATOR_OPTS = [
  { id: 'SMA5', label: 'SMA 5', color: '#ef4444' },
  { id: 'SMA20', label: 'SMA 20', color: '#f59e0b' },
  { id: 'SMA50', label: 'SMA 50', color: '#3b82f6' },
  { id: 'EMA5', label: 'EMA 5', color: '#ec4899' },
  { id: 'EMA20', label: 'EMA 20', color: '#8b5cf6' },
  { id: 'EMA50', label: 'EMA 50', color: '#6366f1' },
  { id: 'BB_up', label: 'BB Up', color: 'rgba(56, 189, 248, 0.8)', style: 2 },
  { id: 'BB_mid', label: 'BB Mid', color: 'rgba(56, 189, 248, 0.4)', style: 2 },
  { id: 'BB_low', label: 'BB Low', color: 'rgba(56, 189, 248, 0.8)', style: 2 },

  // Nowe, rozdzielone wskaźniki ZigZag
  { id: 'ZZ_LINE', label: 'ZigZag Linia', color: '#00ffe1' },
  { id: 'ZZ_HIGH_PTS', label: 'ZZ Highs', color: '#00ff00', isPointSeries: true },
  { id: 'ZZ_LOW_PTS', label: 'ZZ Lows', color: '#ff4444', isPointSeries: true },
]

const buildChartData = (history) =>
  history.map((candle, idx) => ({
    ...candle,
    time: toTimestamp(candle.time ?? '', idx),
    open: Number(candle.open),
    high: Number(candle.high),
    low: Number(candle.low),
    close: Number(candle.close),
  }))

function StatTile({ label, value, hint }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {hint && <div className="tile-hint">{hint}</div>}
    </div>
  )
}

function App() {
  const [history, setHistory] = useState([])
  const [historyError, setHistoryError] = useState('')
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [historyLimit, setHistoryLimit] = useState(2000)
  const DEFAULT_SOURCE = 'ohlc_EURUSD_H1.csv'
  const [sourceName, setSourceName] = useState(DEFAULT_SOURCE)

  const [lastFetchTime, setLastFetchTime] = useState(null)
  const [now, setNow] = useState(new Date())

  const [probability, setProbability] = useState(null)
  const [signalError, setSignalError] = useState('')
  const [loadingSignal, setLoadingSignal] = useState(false)
  const [selectedCandle, setSelectedCandle] = useState(null)

  // --- Stany dla konfiguracji ZigZag ---
  const [zigzagDepth, setZigzagDepth] = useState(23)
  const [zigzagDeviation, setZigzagDeviation] = useState(0.00554)
  // ZMIENIONE: Wartość początkowa ustawiona na 4, aby pasowała do domyślnej w main.py
  const [zigzagBackstep, setZigzagBackstep] = useState(4)
  const [loadingZigzag, setLoadingZigzag] = useState(false)


  // ZigZag - dane z backendu
  const [zigzagHighs, setZigzagHighs] = useState([])
  const [zigzagLows, setZigzagLows] = useState([])
  const [zigzagCombined, setZigzagCombined] = useState([])

  // Domyślnie włączone wskaźniki - PUSTY ZESTAW (NIC NIE ZAZNACZONE)
  const [activeInds, setActiveInds] = useState(new Set())

  const toggleIndicator = (id) => {
    const newSet = new Set(activeInds)
    if (newSet.has(id)) newSet.delete(id)
    else newSet.add(id)
    setActiveInds(newSet)
  }

  // Zaktualizowana funkcja fetchZigzag
  const fetchZigzag = async (depth = zigzagDepth, deviation = zigzagDeviation, backstep = zigzagBackstep) => {
    setLoadingZigzag(true)
    try {
      const res = await axios.get(`${API_URL}/zigzag-test`, {
        params: {
          depth: depth,
          deviation: deviation,
          backstep: backstep // PARAMETR JEST POPRAWNIE WYSYŁANY
        }
      })

      const highs = res.data.pivot_highs || []
      const lows = res.data.pivot_lows || []
      setZigzagHighs(highs)
      setZigzagLows(lows)

      const ordered = res.data.zigzag
        ? res.data.zigzag
        : [...highs, ...lows].sort((a, b) => a.index - b.index)

      setZigzagCombined(ordered)

    } catch (err) {
      console.error("Błąd ZigZag:", err)
    } finally {
      setLoadingZigzag(false)
    }
  }

  // Używamy tego hooka, aby przeliczać ZigZag, gdy zmieniają się parametry
  useEffect(() => {
    // Unikamy przeliczania przy pierwszej renderowaniu, gdy historia jeszcze nie jest załadowana
    if (history.length === 0) return

    const delayDebounceFn = setTimeout(() => {
      fetchZigzag(zigzagDepth, zigzagDeviation, zigzagBackstep)
    }, 500) // Debounce, aby nie spamować API przy szybkich zmianach

    return () => clearTimeout(delayDebounceFn)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zigzagDepth, zigzagDeviation, zigzagBackstep, history.length])


  // Stany dla retraining
  const [retrainLoading, setRetrainLoading] = useState(false)
  const [retrainLogs, setRetrainLogs] = useState('')
  const [retrainSuccess, setRetrainSuccess] = useState(null)

  const [isChartLibLoaded, setIsChartLibLoaded] = useState(false)

  const chartContainer = useRef(null)
  const chartInstance = useRef(null)
  const candleSeriesRef = useRef(null)

  // Mapa referencji do serii wskaźników { "SMA20": ISeriesApi, ... }
  const indicatorSeriesMap = useRef({})

  const chartData = useMemo(() => buildChartData(history), [history])

  const lastCandle = history.length ? history[history.length - 1] : null
  const candleLookup = useMemo(() => {
    const map = new Map()
    chartData.forEach((point, idx) => {
      map.set(point.time, { point, raw: history[idx] })
    })
    return map
  }, [chartData, history])

  // --- ZigZag -> format lightweight-charts ---

  // linia ZigZag (jeden stream)
  const zigzagLineData = useMemo(() => {
    return zigzagCombined.map(p => ({
      time: Math.floor(Date.parse(p.time) / 1000),
      value: p.price
    }))
  }, [zigzagCombined])

  // punkty highs
  const zigzagHighData = useMemo(() => {
    return zigzagHighs.map(p => ({
      time: Math.floor(Date.parse(p.time) / 1000),
      value: p.price,
      // Używamy znacznika, aby wyświetlić kropkę
      mark: { position: 'aboveBar', color: '#00ff00', size: 5, shape: 'circle' }
    }))
  }, [zigzagHighs])

  // punkty lows
  const zigzagLowData = useMemo(() => {
    return zigzagLows.map(p => ({
      time: Math.floor(Date.parse(p.time) / 1000),
      value: p.price,
      // Używamy znacznika, aby wyświetlić kropkę
      mark: { position: 'belowBar', color: '#ff4444', size: 5, shape: 'circle' }
    }))
  }, [zigzagLows])

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (window.LightweightCharts) {
      setIsChartLibLoaded(true)
      return
    }
    const script = document.createElement('script')
    script.src = 'https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js'
    script.async = true
    script.onload = () => setIsChartLibLoaded(true)
    document.body.appendChild(script)
  }, [])

  const fetchHistory = async (limit = historyLimit) => {
    setHistoryError('')
    setLoadingHistory(true)
    try {
      const res = await axios.get(`${API_URL}/history`, { params: { limit } })
      setHistory(res.data.candles)

      // Nie robimy fetchZigzag tutaj, bo zrobi to useEffect po załadowaniu historii
      // (ponieważ history.length się zmienia)

      setLastFetchTime(new Date())
      setSignalError('')
      setSourceName(DEFAULT_SOURCE)
      requestSignal()
    } catch (err) {
      setHistoryError(err.response?.data?.detail || 'Nie udało się pobrać historii z CSV.')
    } finally {
      setLoadingHistory(false)
    }
  }

  useEffect(() => {
    fetchHistory(historyLimit)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!historyLimit) return
    fetchHistory(historyLimit)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [historyLimit])

  const reloadCurrentSource = () => fetchHistory(historyLimit)

  const handleRetrain = async () => {
    // Zastąpienie window.confirm własnym UI/console.log, ponieważ alerty są zabronione w iframe.
    console.log("Potwierdź: Czy na pewno chcesz uruchomić douczanie modelu? (TAK/NIE - Działanie rozpoczęte)")

    setRetrainLoading(true)
    setRetrainLogs("")
    setRetrainSuccess(null)

    try {
      const response = await fetch(`${API_URL}/retrain`, { method: 'POST' })
      if (!response.body) throw new Error("Brak strumienia odpowiedzi.")

      const reader = response.body.getReader()
      const decoder = new TextDecoder("utf-8")

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        setRetrainLogs(prev => prev + chunk)
      }

      setRetrainSuccess(true)
      requestSignal()
    } catch (err) {
      console.error(err)
      setRetrainSuccess(false)
      setRetrainLogs(prev => prev + `\n[APP ERROR] Błąd połączenia: ${err.message}`)
    } finally {
      setRetrainLoading(false)
    }
  }

  // INICJALIZACJA WYKRESU (TWORZYMY WSZYSTKIE SERIE RAZ)
  useEffect(() => {
    if (!isChartLibLoaded || !chartContainer.current) return
    if (!window.LightweightCharts) return

    const { createChart } = window.LightweightCharts

    const chart = createChart(chartContainer.current, {
      layout: {
        backgroundColor: '#0d111d',
        textColor: '#e5e7eb',
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.05)' },
        horzLines: { color: 'rgba(255,255,255,0.05)' },
      },
      timeScale: { borderColor: 'rgba(255,255,255,0.12)' },
      rightPriceScale: { borderColor: 'rgba(255,255,255,0.12)' },
      width: chartContainer.current.clientWidth,
      height: 420,
    })

    // 1. Seria świecowa
    candleSeriesRef.current = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })
    chart.timeScale().fitContent()

    // 2. Tworzymy WSZYSTKIE serie wskaźników od razu, ale ukryte
    indicatorSeriesMap.current = {}

    INDICATOR_OPTS.forEach(opt => {
      // Linia ZigZag
      if (opt.id === 'ZZ_LINE') {
        const zzLine = chart.addLineSeries({
          color: opt.color,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
          visible: false
        })
        indicatorSeriesMap.current[opt.id] = zzLine
        return
      }

      // Punkty ZigZag (HIGH/LOW) - musimy użyć markers dla punktów
      if (opt.isPointSeries) {
        // Tworzymy serię świecową/liniową, ale używamy jej tylko do wyświetlania markers
        const pointSeries = chart.addLineSeries({
          color: opt.color,
          lineWidth: 0, // Ukrywamy linię
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false, // Ukrywamy domyślny marker
          visible: false
        })
        indicatorSeriesMap.current[opt.id] = pointSeries
        return
      }


      // standardowe linie SMA/EMA/BB itd.
      const series = chart.addLineSeries({
        color: opt.color,
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        lineStyle: opt.style || 0,
        visible: false
      })
      indicatorSeriesMap.current[opt.id] = series
    })

    const handleResize = () => {
      if (chartContainer.current) {
        chart.applyOptions({ width: chartContainer.current.clientWidth })
      }
    }

    window.addEventListener('resize', handleResize)
    chartInstance.current = chart

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartInstance.current = null
      candleSeriesRef.current = null
      indicatorSeriesMap.current = {}
    }
  }, [isChartLibLoaded])

  // AKTUALIZACJA DANYCH I WIDOCZNOŚCI (BEZ USUWANIA SERII)
  useEffect(() => {
    if (!chartInstance.current || !candleSeriesRef.current || !chartData.length) return

    // 1. Aktualizuj świece
    candleSeriesRef.current.setData(chartData)

    // 2. Aktualizuj zwykłe wskaźniki (pomijamy ZigZag tutaj)
    INDICATOR_OPTS.forEach(opt => {
      // Pomińmy ZigZag
      if (['ZZ_LINE', 'ZZ_HIGH_PTS', 'ZZ_LOW_PTS'].includes(opt.id)) return

      const series = indicatorSeriesMap.current[opt.id]
      if (!series) return

      const lineData = chartData
        .filter(d => d[opt.id] !== undefined && d[opt.id] !== null)
        .map(d => ({ time: d.time, value: d[opt.id] }))

      series.setData(lineData)
      series.applyOptions({ visible: activeInds.has(opt.id) })
    })
  }, [chartData, activeInds])

  // --- UPDATE ZIGZAG (linia + punkty) ---
  useEffect(() => {
    if (!chartInstance.current) return

    const zzLine = indicatorSeriesMap.current["ZZ_LINE"]
    const zzHighPts = indicatorSeriesMap.current["ZZ_HIGH_PTS"]
    const zzLowPts = indicatorSeriesMap.current["ZZ_LOW_PTS"]

    // Zmieniono logikę widoczności na podstawie nowych, pojedynczych ID
    const isLineVisible = activeInds.has("ZZ_LINE")
    const isHighPtsVisible = activeInds.has("ZZ_HIGH_PTS")
    const isLowPtsVisible = activeInds.has("ZZ_LOW_PTS")

    if (zzLine) {
      zzLine.setData(zigzagLineData)
      zzLine.applyOptions({ visible: isLineVisible })
    }

    // Używamy markers w serii świecowej dla lepszej widoczności punktów
    if (zzHighPts && zzLowPts && candleSeriesRef.current) {
      const markers = []
      if (isHighPtsVisible) markers.push(...zigzagHighData)
      if (isLowPtsVisible) markers.push(...zigzagLowData)

      // Ponieważ Lightweight Charts nie obsługuje bezpośrednio markers na LineSeries,
      // użyjemy ich na głównej serii świecowej, co jest standardową praktyką.
      candleSeriesRef.current.setMarkers(markers.map(p => ({
        time: p.time,
        position: p.mark.position,
        color: p.mark.color,
        shape: p.mark.shape,
        size: p.mark.size,
        text: '', // Bez tekstu dla czystego punktu
        id: p.time + (p.mark.position === 'aboveBar' ? 'h' : 'l')
      })))

      // Ukrywamy/pokazujemy puste serie linii, aby wskaźnik na panelu działał
      // Nie mają one danych, ale pozwalają na włączanie/wyłączanie w menu
      zzHighPts.setData(isHighPtsVisible ? zigzagHighData : [])
      zzLowPts.setData(isLowPtsVisible ? zigzagLowData : [])
      zzHighPts.applyOptions({ visible: isHighPtsVisible })
      zzLowPts.applyOptions({ visible: isLowPtsVisible })
    }
  }, [zigzagLineData, zigzagHighData, zigzagLowData, activeInds])


  useEffect(() => {
    if (!chartInstance.current) return
    const handler = (param) => {
      if (!param.time) {
        setSelectedCandle(null)
        return
      }
      const entry = candleLookup.get(param.time)
      if (!entry) {
        setSelectedCandle(null)
        return
      }
      const { point, raw } = entry
      const readableTime = raw?.time || new Date(point.time * 1000).toISOString()
      setSelectedCandle({
        time: readableTime.replace('T', ' '),
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
        rsi: raw.RSI,
        sma20: raw.SMA20
      })
    }

    const chart = chartInstance.current
    chart.subscribeClick(handler)
    return () => chart.unsubscribeClick(handler)
  }, [candleLookup])

  const requestSignal = async () => {
    setSignalError('')
    setLoadingSignal(true)
    try {
      const res = await axios.get(`${API_URL}/predict/from-file`)
      setProbability(res.data.probability)
    } catch (err) {
      setSignalError(err.response?.data?.detail || 'Błąd podczas liczenia sygnału.')
      setProbability(null)
    } finally {
      setLoadingSignal(false)
    }
  }

  const recommendationClass = useMemo(() => {
    if (probability === null) return 'neutral'
    return probability >= 0.5 ? 'buy' : 'sell'
  }, [probability])

  const recommendationCopy = useMemo(() => {
    if (probability === null) return 'Brak wyniku'
    return probability >= 0.5 ? 'BUY' : 'SELL'
  }, [probability])

  const signalProbabilityDisplay = useMemo(() => {
    if (probability === null) return '—'
    const value = recommendationCopy === 'BUY' ? probability : 1 - probability
    return `${(value * 100).toFixed(2)}%`
  }, [probability, recommendationCopy])

  const secondsAgo = useMemo(() => {
    if (!lastFetchTime) return 0
    const diff = Math.floor((now.getTime() - lastFetchTime.getTime()) / 1000)
    return Math.max(0, diff)
  }, [now, lastFetchTime])

  return (
    <div className="container">
      <header className="hero">
        <div>
          <h1>Panel tradera z podglądem rynku</h1>
        </div>
        <div className="hero-badge">
          <span className="pulse" />
          Dane łączone bezpośrednio z CSV + MT5
        </div>
      </header>

      <section className="card">
        <div className="card-header">
          <div>
            <p className="eyebrow">Sygnał z modelu</p>
            <h2>Ocena rynku</h2>
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button onClick={requestSignal} disabled={loadingSignal || !history.length}>
              {loadingSignal ? 'Liczenie...' : 'Aktualizuj sygnał'}
            </button>
          </div>
        </div>

        {signalError && <div className="error banner">{signalError}</div>}

        <div className="signal-box">
          <div>
            <p className="label">Prawdopodobieństwo sygnału</p>
            <p className="signal-value">{signalProbabilityDisplay}</p>
          </div>
          <div>
            <p className="label">Sugestia</p>
            <p className={`signal-pill ${recommendationClass}`}>{recommendationCopy}</p>
          </div>
        </div>
        <p className="hint">
          Decyzja powstaje wyłącznie na bazie pełnych danych źródłowych, niezależnie od aktualnego
          zakresu widocznego na wykresie.
        </p>
      </section>

      {/* SEKCJA DOUCZANIA MODELU */}
      <section className="card">
        <div className="card-header">
          <div>
            <p className="eyebrow">Zarządzanie modelem</p>
            <h2>Douczanie (Retrain)</h2>
          </div>
          <button
            onClick={handleRetrain}
            disabled={retrainLoading}
            style={{
              background: retrainLoading ? '#4b5563' : 'linear-gradient(120deg, #ea580c, #c2410c)'
            }}
          >
            {retrainLoading ? 'Trenowanie...' : 'Doucz Model'}
          </button>
        </div>

        {retrainSuccess === true && (
          <div className="error banner" style={{ borderColor: '#22c55e', background: 'rgba(34, 197, 94, 0.1)', color: '#86efac', marginTop: '15px' }}>
            ✅ Proces zakończony. Sprawdź logi.
          </div>
        )}

        {retrainSuccess === false && (
          <div className="error banner" style={{ marginTop: '15px' }}>
            ❌ Wystąpił błąd komunikacji lub procesu.
          </div>
        )}

        {(retrainLogs || retrainLoading) && (
          <div style={{ marginTop: '15px' }}>
            <p className="label">Logi procesu treningowego (na żywo):</p>
            <pre style={{
              background: '#0b1020',
              padding: '15px',
              borderRadius: '8px',
              overflowX: 'auto',
              fontFamily: 'monospace',
              fontSize: '12px',
              border: '1px solid #1f2d4d',
              maxHeight: '300px',
              overflowY: 'auto',
              whiteSpace: 'pre-wrap'
            }}>
              {retrainLogs || "Inicjalizacja procesu..."}
            </pre>
          </div>
        )}

        <p className="hint">
          Uruchamia skrypt <code>train_model.py</code> na serwerze. Logi będą pojawiać się powyżej w czasie rzeczywistym.
        </p>
      </section>

      {/* NOWY PANEL KONFIGURACJI ZIGZAG */}
      <section className="card">
        <div className="card-header">
          <div>
            <p className="eyebrow">Konfiguracja wskaźnika</p>
            <h2>ZigZag Parameters</h2>
          </div>
          <div className="pill">{loadingZigzag ? 'Przeliczanie...' : 'Gotowy'}</div>
        </div>

        <div className="grid three-col">
          <div className="slider">
            <label htmlFor="zz-depth">Depth (Głębokość - $P$)</label>
            <input
              id="zz-depth"
              type="number"
              min="1"
              step="1"
              value={zigzagDepth}
              onChange={(e) => setZigzagDepth(Number(e.target.value))}
              disabled={loadingZigzag || loadingHistory}
            />
            <p className="hint">Minimalna liczba świec, w której pivot musi być max/min (okno $2 \cdot P + 1$).</p>
          </div>

          <div className="slider">
            <label htmlFor="zz-deviation">Deviation (Dewiacja - $K$)</label>
            <input
              id="zz-deviation"
              type="number"
              min="0.00001"
              step="0.00001"
              value={zigzagDeviation}
              onChange={(e) => setZigzagDeviation(Number(e.target.value))}
              disabled={loadingZigzag || loadingHistory}
            />
            <p className="hint">Minimalna odległość cenowa (High-Low) od poprzedniego pivota, aby uznać nowy.</p>
          </div>

          <div className="slider">
            <label htmlFor="zz-backstep">Backstep (Krok wstecz - $B$)</label>
            <input
              id="zz-backstep"
              type="number"
              min="1"
              step="1"
              value={zigzagBackstep}
              onChange={(e) => setZigzagBackstep(Number(e.target.value))}
              disabled={loadingZigzag || loadingHistory}
            />
            <p className="hint">Minimalna liczba świec między dwoma pivotami. Nie używane w obecnej implementacji backendu, ale przygotowane dla rozszerzeń.</p>
          </div>
        </div>
      </section>


      <section className="card">
        <div className="card-header">
          <div>
            <p className="eyebrow">Wizualizacja</p>
            <h2>Wykres OHLC EURUSD</h2>
          </div>
          <div className="pill">{loadingHistory ? 'Ładowanie...' : `${history.length} świec`}</div>
        </div>

        {/* PANEL KONFIGURACJI WSKAŹNIKÓW - POBIERA Z INDICATOR_OPTS */}
        <div style={{ margin: '15px 0', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {INDICATOR_OPTS.map(opt => (
            <label
              key={opt.id}
              className="pill subtle"
              style={{
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                borderColor: activeInds.has(opt.id) ? opt.color : 'rgba(148, 163, 184, 0.25)',
                background: activeInds.has(opt.id) ? 'rgba(255,255,255,0.05)' : 'transparent'
              }}
            >
              <input
                type="checkbox"
                checked={activeInds.has(opt.id)}
                onChange={() => toggleIndicator(opt.id)}
                style={{ accentColor: opt.color }}
              />
              <span style={{ color: activeInds.has(opt.id) ? opt.color : '#94a3b8', fontWeight: 600, fontSize: '13px' }}>
                {opt.label}
              </span>
            </label>
          ))}
        </div>

        {historyError ? (
          <div className="error banner">{historyError}</div>
        ) : (
          <>
            <div className="chart" ref={chartContainer} aria-label="Wykres świecowy EURUSD">
              {!isChartLibLoaded && <div style={{ padding: 20 }}>Ładowanie biblioteki wykresów...</div>}
            </div>

            <div className="grid">
              <StatTile
                label="Ostatnie zamknięcie"
                value={lastCandle ? lastCandle.close.toFixed(5) : '—'}
                hint={
                  lastFetchTime
                    ? `Aktualizacja: ${lastFetchTime.toLocaleTimeString('pl-PL')} (${secondsAgo}s temu)`
                    : '...'
                }
              />

              <StatTile
                label="Wysokość ostatniej świecy"
                value={
                  lastCandle
                    ? `${((lastCandle.high - lastCandle.low) * 10000).toFixed(2)} pips`
                    : '—'
                }
                hint={
                  lastCandle
                    ? `High–Low: ${lastCandle.low.toFixed(5)} → ${lastCandle.high.toFixed(5)}`
                    : ''
                }
              />

              {selectedCandle && selectedCandle.rsi && (
                <StatTile
                  label="RSI (14)"
                  value={selectedCandle.rsi.toFixed(2)}
                  hint="Oscylator"
                />
              )}

              <StatTile label="Źródło" value={<span className="source-value">{sourceName}</span>} />
            </div>

            <div className="controls inline">
              <div className="slider compact">
                <span>Zasięg danych</span>
                <select
                  value={historyLimit}
                  onChange={(e) => setHistoryLimit(Number(e.target.value))}
                  disabled={loadingHistory}
                >
                  <option value={2000}>2k świec</option>
                  <option value={5000}>5k świec</option>
                  <option value={10000}>10k świec</option>
                </select>
              </div>
              <button onClick={reloadCurrentSource} disabled={loadingHistory}>
                {loadingHistory ? 'Ładowanie...' : 'Zresretuj widok wykresu'}
              </button>
            </div>

            {selectedCandle && (
              <div className="info-bar" role="status" aria-live="polite">
                <div>
                  <p className="label">Czas świecy</p>
                  <p className="info-value">{selectedCandle.time}</p>
                </div>
                <div className="info-grid">
                  <div>
                    <p className="label">Open</p>
                    <p className="info-value">{selectedCandle.open.toFixed(5)}</p>
                  </div>
                  <div>
                    <p className="label">High</p>
                    <p className="info-value">{selectedCandle.high.toFixed(5)}</p>
                  </div>
                  <div>
                    <p className="label">Low</p>
                    <p className="info-value">{selectedCandle.low.toFixed(5)}</p>
                  </div>
                  <div>
                    <p className="label">Close</p>
                    <p className="info-value">{selectedCandle.close.toFixed(5)}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Debug - Uaktualnione o nowe parametry ZZ */}
            <div style={{ background: "#111", padding: 15, marginTop: 20 }}>
              <h3 style={{ color: "#0af" }}>ZigZag — Debug</h3>
              <p style={{ color: "#aaf" }}>
                Parametry: Depth={zigzagDepth}, Deviation={zigzagDeviation}, Backstep={zigzagBackstep}
              </p>
              <p style={{ color: "#aaf" }}>
                Highs: {zigzagHighs.length} | Lows: {zigzagLows.length} | Combined: {zigzagCombined.length}
              </p>
              <pre style={{
                color: "#ccc",
                maxHeight: 300,
                overflowY: "auto",
                background: "#000",
                padding: 10,
                border: "1px solid #333"
              }}>
                {JSON.stringify(
                  { zigzag: zigzagCombined.slice(0, 20) },
                  null,
                  2
                )}
              </pre>
            </div>
          </>
        )}
      </section>
    </div>
  )
}

export default App