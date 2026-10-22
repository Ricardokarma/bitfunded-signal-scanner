import os
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml

st.set_page_config(
    page_title="Bitfunded Signal Scanner Pro",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="collapsed",
)

JOURNAL_PATH = "trades_journal.csv"

DEFAULT_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT",
    "PEPE/USDT", "WIF/USDT", "AAVE/USDT", "UNI/USDT", "LTC/USDT",
    "BCH/USDT", "ATOM/USDT", "FIL/USDT", "INJ/USDT", "TIA/USDT",
    "SEI/USDT", "RENDER/USDT", "FET/USDT", "IMX/USDT", "ORDI/USDT",
]

MODE_THRESHOLDS = {
    "Scalp": 52,
    "Swing": 58,
    "Challenge": 65,
}


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


@st.cache_data(ttl=120)
def load_config():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


config = load_config()


@st.cache_resource
def get_exchange():
    return ccxt.binanceusdm(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
            "timeout": 15000,
        }
    )


@st.cache_data(ttl=90)
def fetch_btc_dominance():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=8)
        r.raise_for_status()
        data = r.json().get("data", {})
        return safe_float(data.get("market_cap_percentage", {}).get("btc"), 0.0)
    except Exception:
        return None


@st.cache_data(ttl=20)
def fetch_ticker(symbol):
    exchange = get_exchange()
    try:
        ticker = exchange.fetch_ticker(symbol)
        return {
            "last": safe_float(ticker.get("last"), 0.0),
            "change": safe_float(ticker.get("percentage"), 0.0),
            "quoteVolume": safe_float(ticker.get("quoteVolume"), 0.0),
        }
    except Exception:
        return {"last": 0.0, "change": 0.0, "quoteVolume": 0.0}


@st.cache_data(ttl=25)
def fetch_ohlcv(symbol, timeframe="15m", limit=180):
    exchange = get_exchange()
    try:
        bars = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=40)
def fetch_funding(symbol):
    exchange = get_exchange()
    try:
        funding = exchange.fetch_funding_rate(symbol)
        return safe_float(funding.get("fundingRate"), 0.0) * 100
    except Exception:
        return 0.0


def _ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(high, low, close, length=14):
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def _adx(high, low, close, length=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
        / atr.replace(0, np.nan)
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
        / atr.replace(0, np.nan)
    )
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1 / length, min_periods=length, adjust=False).mean().fillna(0.0)


def _macd(series, fast=12, slow=26, signal=9):
    macd_line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line


def add_indicators(df):
    if df.empty:
        return df
    out = df.copy()
    out["ema_9"] = _ema(out["close"], 9)
    out["ema_21"] = _ema(out["close"], 21)
    out["ema_50"] = _ema(out["close"], 50)
    out["rsi"] = _rsi(out["close"], 14)
    out["atr"] = _atr(out["high"], out["low"], out["close"], 14)
    out["adx"] = _adx(out["high"], out["low"], out["close"], 14)
    out["volume_sma"] = out["volume"].rolling(window=20).mean()
    macd_line, signal_line = _macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    return out


def score_setup(df, mode):
    empty = {
        "direction": "NEUTRAL",
        "score": 0.0,
        "confidence": 0.0,
        "win_rate": 0.0,
        "reasons": ["No data"],
        "price": 0.0,
        "atr": 0.0,
        "rsi": 50.0,
        "regime": "RANGE",
    }
    if df.empty or len(df) < 60:
        return empty

    last = df.iloc[-1]
    bullish = 0.0
    bearish = 0.0
    reasons = []

    ema_9 = safe_float(last.get("ema_9"))
    ema_21 = safe_float(last.get("ema_21"))
    ema_50 = safe_float(last.get("ema_50"))
    rsi = safe_float(last.get("rsi"), 50)
    adx = safe_float(last.get("adx"), 0)
    price = safe_float(last.get("close"), 0)
    atr = safe_float(last.get("atr"), 0)
    macd_v = safe_float(last.get("macd"))
    macd_s = safe_float(last.get("macd_signal"))
    vol = safe_float(last.get("volume"))
    vol_sma = safe_float(last.get("volume_sma"))

    weights = {
        "Scalp": {"trend": 1.1, "mom": 1.35, "vol": 1.3},
        "Swing": {"trend": 1.35, "mom": 1.15, "vol": 1.05},
        "Challenge": {"trend": 1.45, "mom": 1.0, "vol": 1.0},
    }
    w = weights.get(mode, weights["Challenge"])

    if ema_9 > ema_21 > ema_50:
        bullish += 24 * w["trend"]
        reasons.append("Bullish EMA stack")
    elif ema_9 < ema_21 < ema_50:
        bearish += 24 * w["trend"]
        reasons.append("Bearish EMA stack")

    if 55 <= rsi <= 70:
        bullish += 16 * w["mom"]
        reasons.append("RSI bullish")
    elif 30 <= rsi <= 45:
        bearish += 16 * w["mom"]
        reasons.append("RSI bearish")
    elif rsi > 70:
        bullish += 8 * w["mom"]
        reasons.append("Strong momentum up")
    elif rsi < 30:
        bearish += 8 * w["mom"]
        reasons.append("Strong momentum down")

    if macd_v > macd_s:
        bullish += 14 * w["mom"]
        reasons.append("MACD bullish")
    elif macd_v < macd_s:
        bearish += 14 * w["mom"]
        reasons.append("MACD bearish")

    if adx >= 25:
        if price > ema_21:
            bullish += 12 * w["trend"]
            reasons.append("ADX trend up")
        else:
            bearish += 12 * w["trend"]
            reasons.append("ADX trend down")

    if vol_sma > 0:
        if vol > vol_sma * 1.2:
            bullish += 10 * w["vol"]
            reasons.append("Volume expansion")
        elif vol < vol_sma * 0.75:
            bearish += 8 * w["vol"]
            reasons.append("Weak volume")

    score = max(bullish, bearish) + abs(bullish - bearish) * 0.45
    score = round(min(100.0, score), 1)
    min_score = MODE_THRESHOLDS.get(mode, 60)

    if bullish > bearish and score >= min_score:
        direction = "LONG"
    elif bearish > bullish and score >= min_score:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    if ema_9 > ema_21 > ema_50 and adx > 25:
        regime = "TRENDING_UP"
    elif ema_9 < ema_21 < ema_50 and adx > 25:
        regime = "TRENDING_DOWN"
    else:
        regime = "RANGE"

    confidence = min(99.0, 45 + score * 0.52) if direction != "NEUTRAL" else 0.0
    win_rate = max(38.0, min(76.0, 48 + (score - 50) * 0.5)) if direction != "NEUTRAL" else 0.0

    return {
        "direction": direction,
        "score": score,
        "confidence": round(confidence, 1),
        "win_rate": round(win_rate, 1),
        "reasons": reasons[:5] if reasons else ["Weak setup"],
        "price": price,
        "atr": atr,
        "rsi": rsi,
        "regime": regime,
    }


def build_signal(symbol, mode):
    setups = []
    for tf in ("15m", "1h"):
        df = fetch_ohlcv(symbol, timeframe=tf, limit=180)
        if df.empty:
            continue
        prepared = add_indicators(df)
        s = score_setup(prepared, mode)
        s["timeframe"] = tf
        setups.append(s)

    if not setups:
        return {
            "direction": "NEUTRAL",
            "score": 0.0,
            "confidence": 0.0,
            "win_rate": 0.0,
            "reasons": ["No data"],
            "timeframes": [],
            "price": 0.0,
            "atr": 0.0,
            "regime": "RANGE",
        }

    avg_score = round(sum(x["score"] for x in setups) / len(setups), 1)
    long_votes = sum(1 for x in setups if x["direction"] == "LONG")
    short_votes = sum(1 for x in setups if x["direction"] == "SHORT")
    min_score = MODE_THRESHOLDS.get(mode, 60)

    if long_votes >= short_votes and avg_score >= min_score:
        direction = "LONG"
    elif short_votes > long_votes and avg_score >= min_score:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    reasons = []
    for item in setups:
        reasons.extend(item["reasons"])

    return {
        "direction": direction,
        "score": avg_score,
        "confidence": round(min(99.0, 42 + avg_score * 0.55), 1) if direction != "NEUTRAL" else 0.0,
        "win_rate": round(max(38.0, min(76.0, 48 + (avg_score - 50) * 0.5)), 1) if direction != "NEUTRAL" else 0.0,
        "reasons": list(dict.fromkeys(reasons))[:6],
        "timeframes": [x["timeframe"] for x in setups],
        "price": setups[-1]["price"],
        "atr": setups[-1]["atr"],
        "regime": setups[-1]["regime"],
    }


def load_journal():
    cols = ["timestamp", "symbol", "direction", "entry", "exit", "result", "notes", "mode"]
    if not os.path.exists(JOURNAL_PATH):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(JOURNAL_PATH)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=cols)


def save_trade_to_journal(symbol, direction, entry, exit_price, result, notes, mode):
    df = load_journal()
    new_row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "exit": exit_price,
        "result": result,
        "notes": notes,
        "mode": mode,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(JOURNAL_PATH, index=False)
    return df


def build_learning_profile(journal_df):
    if journal_df is None or journal_df.empty:
        return {
            "direction_bias": {},
            "mode_bias": {},
            "symbol_bias": {},
            "summary": "Sem trades no journal ainda. O scanner aprende automaticamente a medida que gravar resultados.",
        }

    df = journal_df.copy()
    df["direction"] = df["direction"].astype(str).str.upper()
    df["mode"] = df["mode"].astype(str).str.title().replace("", "Challenge")
    df["result"] = pd.to_numeric(df["result"], errors="coerce").fillna(0.0)
    df["symbol"] = df["symbol"].astype(str).str.upper()

    direction_bias = {}
    for d in ("LONG", "SHORT"):
        sub = df[df["direction"] == d]
        if sub.empty:
            direction_bias[d] = {"avg_result": 0.0, "multiplier": 1.0, "count": 0}
            continue
        avg = float(sub["result"].mean())
        direction_bias[d] = {
            "avg_result": round(avg, 2),
            "multiplier": round(clamp(1 + avg / 120.0, 0.85, 1.2), 3),
            "count": int(len(sub)),
        }

    mode_bias = {}
    for m in sorted(df["mode"].unique()):
        sub = df[df["mode"] == m]
        avg = float(sub["result"].mean())
        mode_bias[m] = {
            "avg_result": round(avg, 2),
            "multiplier": round(clamp(1 + avg / 120.0, 0.85, 1.2), 3),
            "count": int(len(sub)),
        }

    symbol_bias = {}
    for s in sorted(df["symbol"].unique()):
        sub = df[df["symbol"] == s]
        avg = float(sub["result"].mean())
        wr = float((sub["result"] > 0).mean()) * 100
        symbol_bias[s] = {
            "avg_result": round(avg, 2),
            "win_rate": round(wr, 1),
            "multiplier": round(clamp(1 + avg / 120.0, 0.8, 1.25), 3),
            "count": int(len(sub)),
        }

    return {
        "direction_bias": direction_bias,
        "mode_bias": mode_bias,
        "symbol_bias": symbol_bias,
        "summary": f"Learning ativo: {len(df)} trades em {len(symbol_bias)} simbolos.",
    }


def apply_learning(score, symbol, mode, direction, profile):
    d_m = safe_float(profile["direction_bias"].get(direction, {}).get("multiplier", 1.0), 1.0)
    m_m = safe_float(profile["mode_bias"].get(mode, {}).get("multiplier", 1.0), 1.0)
    key1 = symbol.upper().replace("/USDT", "")
    key2 = symbol.upper()
    s_m = safe_float(profile["symbol_bias"].get(key1, {}).get("multiplier", 1.0), 1.0)
    if s_m == 1.0:
        s_m = safe_float(profile["symbol_bias"].get(key2, {}).get("multiplier", 1.0), 1.0)
    adjusted = clamp(round(score * d_m * m_m * s_m, 1), 0.0, 100.0)
    return adjusted, f"x{d_m:.2f} dir x{m_m:.2f} mode x{s_m:.2f} pair"


def calculate_position_size(price, atr, account_size, risk_pct):
    price = safe_float(price)
    atr = safe_float(atr)
    account_size = safe_float(account_size)
    risk_pct = safe_float(risk_pct)
    if price <= 0 or atr <= 0 or account_size <= 0:
        return 0.0
    risk_amount = account_size * (risk_pct / 100)
    stop_distance = atr * 1.5
    if stop_distance <= 0:
        return 0.0
    return round(risk_amount / stop_distance, 4)


def get_scan_pairs(cfg):
    custom = cfg.get("pairs") or []
    if isinstance(custom, dict):
        custom = list(custom.keys())
    if custom:
        return [p if "/" in str(p) else f"{p}/USDT" for p in custom]
    return DEFAULT_PAIRS


@st.cache_data(ttl=45, show_spinner=False)
def scan_market(mode, pair_list_key):
    pairs = pair_list_key.split("|")
    rows = []
    errors = 0
    for symbol in pairs:
        try:
            signal = build_signal(symbol, mode)
            if signal["direction"] == "NEUTRAL":
                continue
            ticker = fetch_ticker(symbol)
            funding = fetch_funding(symbol)
            rows.append(
                {
                    "Symbol": symbol.replace("/USDT", ""),
                    "Direction": signal["direction"],
                    "Score": signal["score"],
                    "Confidence": signal["confidence"],
                    "Win Rate": signal["win_rate"],
                    "Price": ticker.get("last") or signal.get("price", 0.0),
                    "24h %": ticker.get("change", 0.0),
                    "Volume": ticker.get("quoteVolume", 0.0),
                    "Funding %": funding,
                    "Reasons": " | ".join(signal["reasons"]),
                    "Timeframes": ", ".join(signal["timeframes"]),
                    "Regime": signal.get("regime", "RANGE"),
                    "ATR": signal.get("atr", 0.0),
                }
            )
        except Exception:
            errors += 1
            continue
    rows = sorted(rows, key=lambda x: x["Score"], reverse=True)
    return rows, errors, len(pairs)


# --------------- UI ---------------
st.title("Bitfunded Signal Scanner Pro")
st.caption("Scanner de futuros USDT para challenges Bitfunded | execucao manual | learning automatico")

if "account_size" not in st.session_state:
    st.session_state.account_size = float(config.get("account_size", 10000))
if "risk_pct" not in st.session_state:
    st.session_state.risk_pct = float(config.get("risk_per_trade_pct", 0.75))
if "mode" not in st.session_state:
    st.session_state.mode = "Challenge"

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1, 1.2, 1])
with ctrl1:
    st.session_state.account_size = st.number_input(
        "Account size ($)",
        min_value=1000.0,
        max_value=500000.0,
        value=float(st.session_state.account_size),
        step=1000.0,
    )
with ctrl2:
    st.session_state.risk_pct = st.number_input(
        "Risk / trade (%)",
        min_value=0.25,
        max_value=2.0,
        value=float(st.session_state.risk_pct),
        step=0.25,
    )
with ctrl3:
    st.session_state.mode = st.selectbox(
        "Trade mode",
        ["Scalp", "Swing", "Challenge"],
        index=["Scalp", "Swing", "Challenge"].index(st.session_state.mode),
    )
with ctrl4:
    st.write("")
    st.write("")
    refresh = st.button("Refresh scan", type="primary", use_container_width=True)

if refresh:
    st.cache_data.clear()
    st.rerun()

btc_dom = fetch_btc_dominance()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Account", f"${st.session_state.account_size:,.0f}")
m2.metric("Risk / trade", f"{st.session_state.risk_pct}%")
m3.metric("Daily loss cap", f"{config.get('daily_loss_limit_pct', 4.0)}%")
m4.metric("BTC Dominance", f"{btc_dom:.1f}%" if btc_dom is not None else "N/A")

with st.expander("Challenge rules & settings", expanded=False):
    st.markdown(
        f"""
- Max open positions: **{config.get('max_open_positions', 2)}**
- Min hold: **{config.get('min_hold_minutes', 12)} min**
- Daily loss cap: **{config.get('daily_loss_limit_pct', 4.0)}%**
- Hard max loss: **{config.get('max_loss_limit_pct', 6.0)}%**
- Mode threshold (score minimo): **{MODE_THRESHOLDS[st.session_state.mode]}**
- Pares no scan: top 30 liquidos (rapido e estavel no Streamlit Cloud)
        """
    )

journal_df = load_journal()
learning = build_learning_profile(journal_df)

pairs = get_scan_pairs(config)
pair_key = "|".join(pairs)

with st.spinner(f"A escanear {len(pairs)} pares em modo {st.session_state.mode}..."):
    results, scan_errors, total = scan_market(st.session_state.mode, pair_key)

adjusted_rows = []
for row in results:
    adj, note = apply_learning(
        row["Score"],
        row["Symbol"],
        st.session_state.mode,
        row["Direction"],
        learning,
    )
    row = dict(row)
    row["Score base"] = row["Score"]
    row["Score"] = adj
    row["Learning"] = note
    adjusted_rows.append(row)

adjusted_rows = sorted(adjusted_rows, key=lambda x: x["Score"], reverse=True)
min_score = MODE_THRESHOLDS[st.session_state.mode]
adjusted_rows = [r for r in adjusted_rows if r["Score"] >= min_score]

st.caption(f"Scan: {total} pares | {len(adjusted_rows)} setups | erros API: {scan_errors} | {learning['summary']}")

if adjusted_rows:
    df = pd.DataFrame(adjusted_rows)
    show_cols = [
        "Symbol", "Direction", "Score", "Score base", "Confidence", "Win Rate",
        "Price", "24h %", "Funding %", "Regime", "Timeframes", "Reasons", "Learning",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    st.dataframe(
        df[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "Score base": st.column_config.NumberColumn("Base", format="%.1f"),
            "Confidence": st.column_config.ProgressColumn("Conf %", min_value=0, max_value=100, format="%.1f"),
            "Win Rate": st.column_config.NumberColumn("WR %", format="%.1f"),
            "Price": st.column_config.NumberColumn(format="%.6g"),
            "24h %": st.column_config.NumberColumn(format="%.2f"),
            "Funding %": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    symbols = df["Symbol"].tolist()
    selected = st.selectbox("Detalhe do setup", symbols)
    full_symbol = f"{selected}/USDT"
    row = df[df["Symbol"] == selected].iloc[0]

    tf = st.radio("Timeframe do grafico", ["15m", "1h"], horizontal=True)
    chart_df = fetch_ohlcv(full_symbol, timeframe=tf, limit=180)
    if not chart_df.empty:
        chart_df = add_indicators(chart_df)
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df["open"],
                high=chart_df["high"],
                low=chart_df["low"],
                close=chart_df["close"],
                name=selected,
            )
        )
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_9"], mode="lines", name="EMA 9", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_21"], mode="lines", name="EMA 21", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_50"], mode="lines", name="EMA 50", line=dict(width=1)))
        fig.update_layout(
            title=f"{selected} | {tf}",
            xaxis_rangeslider_visible=False,
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    price = safe_float(row.get("Price"))
    atr = safe_float(row.get("ATR"))
    if atr <= 0 and price > 0:
        atr = price * 0.008
    size = calculate_position_size(
        price, atr, st.session_state.account_size, st.session_state.risk_pct
    )
    st.info(
        f"**{row['Direction']} {selected}** | Score {row['Score']} (base {row.get('Score base', row['Score'])}) | "
        f"Conf {row['Confidence']}% | WR est. {row['Win Rate']}% | "
        f"Tamanho sugerido ~ **{size}** unidades | Risk {st.session_state.risk_pct}% de ${st.session_state.account_size:,.0f}"
    )
else:
    st.warning(
        "Nenhum setup forte neste momento. "
        "Tenta **Scalp** (threshold mais baixo), clica **Refresh scan**, ou espera mais volatilidade."
    )

st.divider()
st.subheader("Journal + auto-learning")
st.caption("Grava o resultado real de cada trade. O scanner ajusta sozinho o ranking (direcao, modo e simbolo).")

with st.form("journal_form", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        j_symbol = st.text_input("Symbol", value="BTC/USDT")
    with c2:
        j_dir = st.selectbox("Direction", ["LONG", "SHORT"])
    with c3:
        j_entry = st.number_input("Entry", value=0.0, format="%.6f")
    with c4:
        j_exit = st.number_input("Exit", value=0.0, format="%.6f")
    c5, c6, c7 = st.columns(3)
    with c5:
        j_result = st.number_input("Result %", value=0.0, format="%.2f", help="Lucro/prejuizo em % (ex: 1.5 ou -0.8)")
    with c6:
        j_mode = st.selectbox(
            "Mode",
            ["Scalp", "Swing", "Challenge"],
            index=["Scalp", "Swing", "Challenge"].index(st.session_state.mode),
        )
    with c7:
        j_notes = st.text_input("Notes", value="")
    if st.form_submit_button("Guardar trade"):
        if j_symbol.strip():
            save_trade_to_journal(
                j_symbol.strip(), j_dir, j_entry, j_exit, j_result, j_notes, j_mode
            )
            st.success(f"Trade guardado: {j_symbol} | learning atualizado.")
            st.rerun()

journal_df = load_journal()
if not journal_df.empty:
    st.dataframe(journal_df.tail(15), use_container_width=True, hide_index=True)
    learning = build_learning_profile(journal_df)
    st.write(learning["summary"])
    lc1, lc2 = st.columns(2)
    with lc1:
        if learning["direction_bias"]:
            st.markdown("**Performance por direcao**")
            st.dataframe(
                pd.DataFrame.from_dict(learning["direction_bias"], orient="index"),
                use_container_width=True,
            )
    with lc2:
        if learning["symbol_bias"]:
            st.markdown("**Top simbolos (avg result %)**")
            sym_df = (
                pd.DataFrame.from_dict(learning["symbol_bias"], orient="index")
                .reset_index()
                .rename(columns={"index": "Symbol"})
                .sort_values("avg_result", ascending=False)
                .head(10)
            )
            st.dataframe(sym_df, use_container_width=True, hide_index=True)
else:
    st.caption("Ainda sem trades no journal.")

st.divider()
st.markdown(
    """
**Como usar em challenge Bitfunded**
1. Escolhe o **modo** (Challenge = mais seletivo).
2. Ajusta **account size** e **risk** aos teus numeros reais.
3. So entra em setups com score alto e razoes claras.
4. Depois de cada trade, grava o **Result %** 鈥� o learning ajusta o ranking sozinho.
5. Respeita daily loss e max loss da challenge.

*Ferramenta de suporte a decisao. Nao executa ordens e nao garante resultados.*
"""
)
