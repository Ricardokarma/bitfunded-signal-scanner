import os
from datetime import datetime

import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml

st.set_page_config(page_title="Bitfunded Signal Scanner Pro", page_icon="📈", layout="wide")

JOURNAL_PATH = "trades_journal.csv"


@st.cache_data(ttl=60)
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


config = load_config()


@st.cache_resource
def get_exchange():
    return ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


@st.cache_data(ttl=60)
def fetch_btc_dominance():
    try:
        response = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        response.raise_for_status()
        data = response.json().get("data", {})
        market_caps = data.get("market_cap_percentage", {})
        return safe_float(market_caps.get("btc", 0.0), 0.0)
    except Exception:
        return None


@st.cache_data(ttl=15)
def fetch_ticker(symbol):
    exchange = get_exchange()
    try:
        ticker = exchange.fetch_ticker(symbol)
        return {
            "last": safe_float(ticker.get("last"), 0.0),
            "change": safe_float(ticker.get("percentage"), 0.0),
            "baseVolume": safe_float(ticker.get("baseVolume"), 0.0),
            "quoteVolume": safe_float(ticker.get("quoteVolume"), 0.0),
            "bid": safe_float(ticker.get("bid"), 0.0),
            "ask": safe_float(ticker.get("ask"), 0.0),
        }
    except Exception:
        return {"last": 0.0, "change": 0.0, "baseVolume": 0.0, "quoteVolume": 0.0, "bid": 0.0, "ask": 0.0}


@st.cache_data(ttl=30)
def fetch_ohlcv(symbol, timeframe="15m", limit=220):
    exchange = get_exchange()
    try:
        bars = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_funding_and_oi(symbol):
    exchange = get_exchange()
    result = {"fundingRate": 0.0, "openInterest": 0.0, "longShortRatio": 0.0}

    try:
        funding = exchange.fetch_funding_rate(symbol)
        result["fundingRate"] = safe_float(funding.get("fundingRate"), 0.0) * 100
    except Exception:
        pass

    try:
        stats = exchange.fetch_open_interest(symbol)
        result["openInterest"] = safe_float(stats.get("openInterest"), 0.0)
    except Exception:
        pass

    try:
        long_short = exchange.fetch_long_short_ratio(symbol=symbol)
        if isinstance(long_short, dict):
            ratio = next(iter(long_short.values()), 0.0)
            result["longShortRatio"] = safe_float(ratio, 0.0)
    except Exception:
        pass

    return result


@st.cache_data(ttl=180)
def get_active_pairs():
    exchange = get_exchange()
    try:
        markets = exchange.load_markets()
    except Exception:
        return []

    valid = []
    for symbol, market in markets.items():
        if not market.get("active", False):
            continue
        if market.get("quote") != "USDT":
            continue
        if market.get("type") not in ("future", "swap"):
            continue
        if "/USDT" not in symbol:
            continue
        valid.append(symbol)

    return sorted(set(valid))


def add_indicators(df):
    if df.empty:
        return df

    out = df.copy()
    out["ema_9"] = ta.ema(out["close"], length=9)
    out["ema_21"] = ta.ema(out["close"], length=21)
    out["ema_50"] = ta.ema(out["close"], length=50)
    out["rsi"] = ta.rsi(out["close"], length=14)
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["adx"] = ta.adx(out["high"], out["low"], out["close"], length=14)
    out["volume_sma"] = out["volume"].rolling(window=20).mean()

    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame):
        for column in macd.columns:
            out[column] = macd[column]

    return out


def compute_regime(df):
    if df.empty:
        return "RANGE"

    last = df.iloc[-1]
    ema_9 = safe_float(last.get("ema_9"))
    ema_21 = safe_float(last.get("ema_21"))
    ema_50 = safe_float(last.get("ema_50"))
    adx = safe_float(last.get("adx"), 0.0)

    if ema_9 > ema_21 > ema_50 and adx > 25:
        return "TRENDING_UP"
    if ema_9 < ema_21 < ema_50 and adx > 25:
        return "TRENDING_DOWN"
    return "RANGE"


def score_setup(df, mode):
    if df.empty:
        return {"direction": "NEUTRAL", "score": 0.0, "confidence": 0.0, "win_rate": 0.0, "reasons": ["No data"], "price": 0.0, "atr": 0.0, "rsi": 0.0, "atr_pct": 0.0, "regime": "RANGE"}

    mode_profile = {
        "Scalp": {"trend_weight": 1.2, "momentum_weight": 1.3, "volume_weight": 1.4, "min_score": 58},
        "Swing": {"trend_weight": 1.35, "momentum_weight": 1.15, "volume_weight": 1.1, "min_score": 62},
        "Challenge": {"trend_weight": 1.45, "momentum_weight": 1.05, "volume_weight": 1.0, "min_score": 68},
    }
    profile = mode_profile.get(mode, mode_profile["Challenge"])

    last = df.iloc[-1]
    bullish_score = 0.0
    bearish_score = 0.0
    reasons = []

    ema_9 = safe_float(last.get("ema_9"))
    ema_21 = safe_float(last.get("ema_21"))
    ema_50 = safe_float(last.get("ema_50"))
    rsi = safe_float(last.get("rsi"), 50)
    adx = safe_float(last.get("adx"), 0.0)
    price = safe_float(last.get("close"), 0.0)
    atr = safe_float(last.get("atr"), 0.0)
    atr_pct = ((atr / price) * 100) if price > 0 and atr > 0 else 0.0

    if ema_9 > ema_21 > ema_50:
        bullish_score += 22 * profile["trend_weight"]
        reasons.append("Bullish EMA stack")
    elif ema_9 < ema_21 < ema_50:
        bearish_score += 22 * profile["trend_weight"]
        reasons.append("Bearish EMA stack")

    if 55 <= rsi <= 70:
        bullish_score += 18 * profile["momentum_weight"]
        reasons.append("RSI bullish")
    elif 30 <= rsi <= 45:
        bearish_score += 18 * profile["momentum_weight"]
        reasons.append("RSI bearish")
    elif rsi > 70:
        bullish_score += 10 * profile["momentum_weight"]
        reasons.append("Strong bullish momentum")
    elif rsi < 30:
        bearish_score += 10 * profile["momentum_weight"]
        reasons.append("Strong bearish momentum")

    macd_value = safe_float(last.get("MACD_12_26_9"))
    signal_value = safe_float(last.get("MACDs_12_26_9"))
    if macd_value > signal_value:
        bullish_score += 14 * profile["momentum_weight"]
        reasons.append("MACD above signal")
    elif macd_value < signal_value:
        bearish_score += 14 * profile["momentum_weight"]
        reasons.append("MACD below signal")

    if adx >= 25:
        if price > ema_21:
            bullish_score += 12 * profile["trend_weight"]
            reasons.append("ADX trend confirmation")
        else:
            bearish_score += 12 * profile["trend_weight"]
            reasons.append("ADX trend rejection")

    volume = safe_float(last.get("volume"), 0.0)
    volume_sma = safe_float(last.get("volume_sma"), 0.0)
    if volume_sma > 0:
        if volume > volume_sma * 1.2:
            bullish_score += 12 * profile["volume_weight"]
            reasons.append("Volume expansion")
        elif volume < volume_sma * 0.8:
            bearish_score += 12 * profile["volume_weight"]
            reasons.append("Weak volume")

    score = max(bullish_score, bearish_score) + abs(bullish_score - bearish_score) * 0.5
    score = round(min(100.0, score), 1)

    if bullish_score > bearish_score and score >= profile["min_score"]:
        direction = "LONG"
    elif bearish_score > bullish_score and score >= profile["min_score"]:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    confidence = min(99.0, 48 + score * 0.52) if direction in ["LONG", "SHORT"] else 0.0
    win_rate = max(35.0, min(78.0, 50 + (score - 50) * 0.55)) if direction in ["LONG", "SHORT"] else 0.0

    return {
        "direction": direction,
        "score": score,
        "confidence": round(confidence, 1),
        "win_rate": round(win_rate, 1),
        "reasons": reasons if reasons else ["Weak setup"],
        "price": price,
        "atr": atr,
        "rsi": rsi,
        "atr_pct": round(atr_pct, 2),
        "regime": compute_regime(df),
    }


def build_signal(symbol, mode):
    setups = []
    for tf in ["15m", "1h"]:
        df = fetch_ohlcv(symbol, timeframe=tf, limit=220)
        if df.empty:
            continue
        prepared = add_indicators(df)
        signal = score_setup(prepared, mode)
        signal["timeframe"] = tf
        setups.append(signal)

    if not setups:
        return {"direction": "NEUTRAL", "score": 0.0, "confidence": 0.0, "win_rate": 0.0, "reasons": ["No data"], "timeframes": []}

    overall_score = round(sum(item["score"] for item in setups) / len(setups), 1)
    long_votes = sum(1 for item in setups if item["direction"] == "LONG")
    short_votes = sum(1 for item in setups if item["direction"] == "SHORT")

    if long_votes >= short_votes and overall_score >= 58:
        direction = "LONG"
    elif short_votes > long_votes and overall_score >= 58:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    reasons = []
    for item in setups:
        reasons.extend(item["reasons"])

    confidence = min(99.0, 45 + overall_score * 0.55) if direction in ["LONG", "SHORT"] else 0.0
    win_rate = max(35.0, min(78.0, 50 + (overall_score - 50) * 0.55)) if direction in ["LONG", "SHORT"] else 0.0

    return {
        "direction": direction,
        "score": overall_score,
        "confidence": round(confidence, 1),
        "win_rate": round(win_rate, 1),
        "reasons": reasons[:6],
        "timeframes": [item["timeframe"] for item in setups],
    }


def build_learning_profile():
    df = load_journal()
    if df.empty:
        return {
            "direction_bias": {},
            "mode_bias": {},
            "symbol_bias": {},
            "summary": "No journal data yet. The scanner is learning from future trades.",
        }

    df["direction"] = df["direction"].astype(str).str.upper().fillna("LONG")
    df["mode"] = df["mode"].astype(str).str.title().fillna("Challenge")
    df["result"] = pd.to_numeric(df["result"], errors="coerce").fillna(0.0)
    df["symbol"] = df["symbol"].astype(str).str.upper().fillna("UNKNOWN")

    direction_bias = {}
    for direction in ["LONG", "SHORT"]:
        subset = df[df["direction"] == direction]
        if subset.empty:
            direction_bias[direction] = {"avg_result": 0.0, "multiplier": 1.0, "count": 0}
            continue
        avg_result = float(subset["result"].mean())
        direction_bias[direction] = {
            "avg_result": round(avg_result, 2),
            "multiplier": round(1 + (avg_result / 100.0), 3),
            "count": int(len(subset)),
        }

    mode_bias = {}
    for mode in sorted(df["mode"].unique()):
        subset = df[df["mode"] == mode]
        avg = float(subset["result"].mean())
        mode_bias[mode] = {
            "avg_result": round(avg, 2),
            "multiplier": round(1 + (avg / 100.0), 3),
            "count": int(len(subset)),
        }

    symbol_bias = {}
    for symbol in sorted(df["symbol"].unique()):
        subset = df[df["symbol"] == symbol]
        avg = float(subset["result"].mean())
        win_rate = float((subset["result"] > 0).mean()) * 100.0
        symbol_bias[symbol] = {
            "avg_result": round(avg, 2),
            "win_rate": round(win_rate, 1),
            "multiplier": round(1 + (avg / 100.0), 3),
            "count": int(len(subset)),
        }

    summary = "Learning active: {} trade records analyzed across {} symbols.".format(len(df), len(symbol_bias))
    return {"direction_bias": direction_bias, "mode_bias": mode_bias, "symbol_bias": symbol_bias, "summary": summary}


def apply_learning_adjustment(signal, symbol, mode, direction, profile):
    direction_data = profile["direction_bias"].get(direction, {"multiplier": 1.0})
    mode_data = profile["mode_bias"].get(mode, {"multiplier": 1.0})
    symbol_data = profile["symbol_bias"].get(symbol.upper(), {"multiplier": 1.0})

    direction_multiplier = safe_float(direction_data.get("multiplier", 1.0), 1.0)
    mode_multiplier = safe_float(mode_data.get("multiplier", 1.0), 1.0)
    symbol_multiplier = safe_float(symbol_data.get("multiplier", 1.0), 1.0)

    adjusted = signal["score"] * direction_multiplier * mode_multiplier * symbol_multiplier
    adjusted = clamp(round(adjusted, 1), 0.0, 100.0)

    note = "Learned adjustments: {} × {} × {} = {:.1f}".format(
        round(direction_multiplier, 3),
        round(mode_multiplier, 3),
        round(symbol_multiplier, 3),
        adjusted,
    )
    return adjusted, note


def load_journal():
    if not os.path.exists(JOURNAL_PATH):
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "entry", "exit", "result", "notes", "mode"])

    df = pd.read_csv(JOURNAL_PATH)
    for column in ["timestamp", "symbol", "direction", "entry", "exit", "result", "notes", "mode"]:
        if column not in df.columns:
            df[column] = ""
    df["result"] = pd.to_numeric(df["result"], errors="coerce").fillna(0.0)
    df["mode"] = df["mode"].fillna("Challenge")
    return df


def save_trade_to_journal(symbol, direction, entry, exit_price, result, notes, mode):
    df = load_journal()
    new_row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
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


@st.cache_data(ttl=30)
def scan_market(mode):
    learning_profile = build_learning_profile()
    pair_list = config.get("pairs") or []
    if not pair_list:
        pair_list = get_active_pairs()
    if isinstance(pair_list, str):
        pair_list = [pair_list]

    rows = []
    for symbol in pair_list:
        signal = build_signal(symbol, mode)
        if signal["direction"] == "NEUTRAL":
            continue

        adjusted_score, learning_note = apply_learning_adjustment(signal, symbol, mode, signal["direction"], learning_profile)
        signal["score"] = adjusted_score
        signal["confidence"] = clamp(round(signal["confidence"] * 0.9, 1), 0.0, 99.0)
        signal["win_rate"] = clamp(round(signal["win_rate"] * 0.9, 1), 0.0, 78.0)

        if signal["score"] < 58:
            continue

        ticker = fetch_ticker(symbol)
        funding = fetch_funding_and_oi(symbol)
        rows.append(
            {
                "Symbol": symbol.replace("/USDT", ""),
                "Direction": signal["direction"],
                "Score": signal["score"],
                "Confidence": signal["confidence"],
                "Win Rate": signal["win_rate"],
                "Price": ticker.get("last", 0.0),
                "24h %": ticker.get("change", 0.0),
                "Volume": ticker.get("quoteVolume", 0.0),
                "Funding %": funding.get("fundingRate", 0.0),
                "OI": funding.get("openInterest", 0.0),
                "Reasons": " • ".join(signal["reasons"]),
                "Timeframes": ", ".join(signal["timeframes"]),
                "Learning": learning_note,
            }
        )

    return sorted(rows, key=lambda item: item["Score"], reverse=True)


st.title("📈 Bitfunded Signal Scanner Pro")
st.caption("Self-learning market scanner using journal feedback to improve future trade suggestions.")

btc_dom = fetch_btc_dominance()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Account size", f"${config.get('account_size', 0):,}")
with col2:
    st.metric("Risk / trade", f"{config.get('risk_per_trade_pct', 0)}%")
with col3:
    st.metric("Daily loss", f"{config.get('daily_loss_limit_pct', 0)}%")
with col4:
    st.metric("BTC Dominance", f"{btc_dom:.1f}%" if btc_dom is not None else "N/A")

st.divider()

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Trade mode", ["Scalp", "Swing", "Challenge"], index=2)
    min_score = st.slider("Minimum score", 50, 85, 58)

    learning_profile = build_learning_profile()
    st.subheader("Auto-learning")
    st.write(learning_profile["summary"])

    if st.button("Refresh scan"):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        f"""
        - Max open positions: {config.get('max_open_positions', 0)}
        - Min hold: {config.get('min_hold_minutes', 12)} min
        - Max loss cap: {config.get('max_loss_limit_pct', 0)}%
        """
    )

results = scan_market(mode)
results = [row for row in results if row["Score"] >= min_score]

if results:
    df = pd.DataFrame(results)
    st.subheader("Scanner results")
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f"),
            "Win Rate": st.column_config.ProgressColumn("Win Rate", min_value=0, max_value=100, format="%.1f"),
            "Price": st.column_config.NumberColumn(format="%.4f"),
            "24h %": st.column_config.NumberColumn(format="%.2f"),
            "Funding %": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    selected_symbol = st.selectbox("Inspect symbol", df["Symbol"].unique())
    selected_tf = st.selectbox("Timeframe", ["15m", "1h"])
    symbol = f"{selected_symbol}/USDT"

    chart_df = fetch_ohlcv(symbol, timeframe=selected_tf, limit=220)
    if not chart_df.empty:
        chart_df = add_indicators(chart_df)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name=symbol,
        ))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_9"], mode="lines", name="EMA 9"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_21"], mode="lines", name="EMA 21"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_50"], mode="lines", name="EMA 50"))
        fig.update_layout(title=f"{symbol} {selected_tf}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    row = df[df["Symbol"] == selected_symbol].iloc[0]
    price = safe_float(row.get("Price"))
    atr_estimate = max(price * 0.003, 0.01)
    position_size = round(
        (safe_float(config.get("account_size")) * (safe_float(config.get("risk_per_trade_pct")) / 100)) / (atr_estimate * 1.5),
        4,
    )
    st.info(
        f"{row['Direction']} setup on {selected_symbol}/USDT | Score {row['Score']} | Confidence {row['Confidence']}% | Estimated win rate {row['Win Rate']}% | Suggested size {position_size} units"
    )
    if "Learning" in row:
        st.caption(row["Learning"])
else:
    st.warning("No quality setups meet this mode’s filter. Lower the minimum score or wait for stronger trend conditions.")

st.divider()

st.subheader("Manual trade journal")
with st.form("journal_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        journal_symbol = st.text_input("Symbol", value="BTC/USDT")
    with c2:
        journal_direction = st.selectbox("Direction", ["LONG", "SHORT"])
    with c3:
        journal_entry = st.number_input("Entry", value=0.0, format="%.4f")

    c4, c5, c6 = st.columns(3)
    with c4:
        journal_exit = st.number_input("Exit", value=0.0, format="%.4f")
    with c5:
        journal_result = st.number_input("Result %", value=0.0, format="%.2f")
    with c6:
        journal_mode = st.selectbox("Mode", ["Scalp", "Swing", "Challenge"], index=2)

    journal_notes = st.text_input("Notes", value="")
    submitted = st.form_submit_button("Save trade")
    if submitted:
        symbol_text = journal_symbol.strip()
        if symbol_text:
            save_trade_to_journal(symbol_text, journal_direction, journal_entry, journal_exit, journal_result, journal_notes, journal_mode)
            st.success(f"Trade saved for {symbol_text}.")
            st.cache_data.clear()

journal_df = load_journal()
if not journal_df.empty:
    st.dataframe(journal_df.tail(10), use_container_width=True)
else:
    st.caption("No journal entries yet.")

st.divider()

st.subheader("Learning insights")
profile = build_learning_profile()
if profile["direction_bias"]:
    direction_df = pd.DataFrame.from_dict(profile["direction_bias"], orient="index").reset_index().rename(columns={"index": "Direction"})
    st.dataframe(direction_df, use_container_width=True)

    symbol_df = pd.DataFrame.from_dict(profile["symbol_bias"], orient="index").reset_index().rename(columns={"index": "Symbol"})
    top_symbols = symbol_df.sort_values("avg_result", ascending=False).head(10)
    st.dataframe(top_symbols, use_container_width=True)

st.info("The scanner learns from your recorded results. Over time it adjusts its suggestions based on trade quality by direction, mode, and symbol performance.")

