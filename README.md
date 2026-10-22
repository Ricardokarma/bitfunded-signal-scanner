import os
import time
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


@st.cache_data(ttl=30)
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


@st.cache_data(ttl=120)
def get_all_active_usdt_pairs():
    exchange = get_exchange()
    try:
        markets = exchange.load_markets()
    except Exception:
        return []

    valid = []
    for symbol, market in markets.items():
        if not market.get("active", False):
            continue
        if market.get("settle") != "USDT":
            continue
        if market.get("quote") != "USDT":
            continue
        if market.get("type") != "future":
            continue
        if symbol.endswith(("/USDT", "USDT")):
            pass
        if "/USDT" in symbol and not symbol.startswith(("BTC/USDT", "ETH/USDT")):
            valid.append(symbol)
        if symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "SUI/USDT"]:
            valid.append(symbol)

    seen = set()
    cleaned = []
    for s in valid:
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    cleaned.sort()
    return cleaned


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@st.cache_data(ttl=60)
def fetch_btc_dominance():
    try:
        response = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        response.raise_for_status()
        data = response.json()
        market_caps = data.get("data", {}).get("market_cap_percentage", {})
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


def combine_signal_scores(score_15m, score_1h):
    return round(min(100.0, (score_15m * 0.65) + (score_1h * 0.35)), 1)


def estimate_win_rate(score, market_regime_strength, volatility_penalty):
    base = 50 + (score - 50) * 0.6 + market_regime_strength
    estimate = max(35.0, min(78.0, base - volatility_penalty))
    return round(estimate, 1)


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


def score_setup(df):
    if df.empty:
        return {
            "direction": "NEUTRAL",
            "score": 0.0,
            "confidence": 0.0,
            "win_rate": 0.0,
            "reasons": ["No data"],
            "price": 0.0,
            "atr": 0.0,
            "rsi": 0.0,
            "atr_pct": 0.0,
            "regime": "RANGE",
            "trend_strength": 0.0,
            "volatility_penalty": 0.0,
        }

    last = df.iloc[-1]
    bullish_score = 0
    bearish_score = 0
    reasons = []

    ema_9 = safe_float(last.get("ema_9"))
    ema_21 = safe_float(last.get("ema_21"))
    ema_50 = safe_float(last.get("ema_50"))

    if ema_9 > ema_21 > ema_50:
        bullish_score += 25
        reasons.append("Bullish EMA stack")
    elif ema_9 < ema_21 < ema_50:
        bearish_score += 25
        reasons.append("Bearish EMA stack")

    rsi = safe_float(last.get("rsi"), 50)
    if 55 <= rsi <= 70:
        bullish_score += 18
        reasons.append("RSI bullish")
    elif 30 <= rsi <= 45:
        bearish_score += 18
        reasons.append("RSI bearish")
    elif rsi > 70:
        bullish_score += 8
        reasons.append("Strong bullish momentum")
    elif rsi < 30:
        bearish_score += 8
        reasons.append("Strong bearish momentum")

    macd_value = safe_float(last.get("MACD_12_26_9"))
    signal_value = safe_float(last.get("MACDs_12_26_9"))
    if macd_value > signal_value:
        bullish_score += 16
        reasons.append("MACD above signal")
    elif macd_value < signal_value:
        bearish_score += 16
        reasons.append("MACD below signal")

    adx = safe_float(last.get("adx"))
    if adx >= 25:
        if safe_float(last.get("close")) > ema_21:
            bullish_score += 12
            reasons.append("Trend confirmation")
        else:
            bearish_score += 12
            reasons.append("Trend rejection")

    volume = safe_float(last.get("volume"))
    volume_sma = safe_float(last.get("volume_sma"))
    if volume_sma > 0:
        if volume > volume_sma * 1.2:
            bullish_score += 10
            reasons.append("Volume expansion")
        elif volume < volume_sma * 0.8:
            bearish_score += 10
            reasons.append("Weak volume")

    price = safe_float(last.get("close"), 0.0)
    atr = safe_float(last.get("atr"), 0.0)
    atr_pct = ((atr / price) * 100) if price > 0 and atr > 0 else 0.0

    trend_strength = abs(bullish_score - bearish_score) * 0.7
    volatility_penalty = 0.0 if atr_pct < 0.7 else (atr_pct - 0.7) * 2

    score = max(bullish_score, bearish_score) + abs(bullish_score - bearish_score) * 0.5
    score = round(min(100.0, score), 1)
    regime = compute_regime(df)

    if bullish_score > bearish_score and score >= 55:
        direction = "LONG"
    elif bearish_score > bullish_score and score >= 55:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    confidence = min(99.0, 45 + score * 0.55) if direction in ["LONG", "SHORT"] else 0.0
    win_rate = estimate_win_rate(score, trend_strength, volatility_penalty)

    return {
        "direction": direction,
        "score": score,
        "confidence": round(confidence, 1),
        "win_rate": win_rate,
        "reasons": reasons if reasons else ["Weak setup"],
        "price": price,
        "atr": atr,
        "rsi": rsi,
        "atr_pct": round(atr_pct, 2),
        "regime": regime,
        "trend_strength": round(trend_strength, 2),
        "volatility_penalty": round(volatility_penalty, 2),
    }


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


def build_signal_for_symbol(symbol):
    timeframes = ["15m", "1h"]
    viewed = []
    for timeframe in timeframes:
        df = fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
        if df.empty:
            continue
        df = add_indicators(df)
        setup = score_setup(df)
        setup["timeframe"] = timeframe
        viewed.append(setup)

    if not viewed:
        return {"direction": "NEUTRAL", "score": 0.0, "confidence": 0.0, "win_rate": 0.0, "reasons": ["No data"], "timeframes": []}

    score_15m = next((x["score"] for x in viewed if x["timeframe"] == "15m"), 0.0)
    score_1h = next((x["score"] for x in viewed if x["timeframe"] == "1h"), 0.0)
    combined = combine_signal_scores(score_15m, score_1h)

    long_votes = sum(1 for x in viewed if x["direction"] == "LONG")
    short_votes = sum(1 for x in viewed if x["direction"] == "SHORT")

    if long_votes >= short_votes and combined >= 55:
        direction = "LONG"
    elif short_votes > long_votes and combined >= 55:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    reasons = []
    for item in viewed:
        reasons.extend(item["reasons"])

    final_confidence = min(99.0, (combined * 0.7) + (max(long_votes, short_votes) * 6)) if direction != "NEUTRAL" else 0.0
    final_win_rate = min(78.0, max(35.0, combined * 0.7)) if direction != "NEUTRAL" else 0.0

    return {
        "direction": direction,
        "score": round(combined, 1),
        "confidence": round(final_confidence, 1),
        "win_rate": round(final_win_rate, 1),
        "reasons": reasons[:6],
        "timeframes": [x["timeframe"] for x in viewed],
    }


@st.cache_data(ttl=30)
def scan_market():
    rows = []
    market_pairs = get_all_active_usdt_pairs()
    if not market_pairs:
        market_pairs = config.get("pairs", [])
        if isinstance(market_pairs, dict):
            market_pairs = list(market_pairs.keys())

    for symbol in market_pairs:
        signal = build_signal_for_symbol(symbol)
        if signal["direction"] == "NEUTRAL":
            continue
        if signal["score"] < 55:
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
            }
        )

    return sorted(rows, key=lambda item: item["Score"], reverse=True)


def load_journal():
    if not os.path.exists(JOURNAL_PATH):
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "entry", "exit", "result", "notes"])
    return pd.read_csv(JOURNAL_PATH)


def save_trade_to_journal(symbol, direction, entry, exit_price, result, notes):
    df = load_journal()
    new_row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "exit": exit_price,
        "result": result,
        "notes": notes,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(JOURNAL_PATH, index=False)


st.title("📈 Bitfunded Signal Scanner Pro")
st.caption("Dynamic futures scanner with market-wide filtering, confidence scoring, and trade journaling.")

btc_dom = fetch_btc_dominance()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Account size", f"${config.get('account_size', 0):,}")
with col2:
    st.metric("Risk / trade", f"{config.get('risk_per_trade_pct', 0)}%")
with col3:
    st.metric("Daily loss cap", f"{config.get('daily_loss_limit_pct', 0)}%")
with col4:
    st.metric("BTC Dominance", f"{btc_dom:.1f}%" if btc_dom is not None else "N/A")

st.divider()

with st.sidebar:
    st.header("Challenge controls")
    st.markdown(
        f"""
        - Max open positions: {config.get('max_open_positions', 0)}
        - Min hold: {config.get('min_hold_minutes', 12)} min
        - Daily loss cap: {config.get('daily_loss_limit_pct', 0)}%
        - Hard max loss: {config.get('max_loss_limit_pct', 0)}%
        """
    )

    if st.button("Refresh scan"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("Trade mode")
    mode = st.radio("Select mode", ["Scalp", "Swing", "Challenge"], index=2)

results = scan_market()

if results:
    df = pd.DataFrame(results)
    st.subheader("Dynamic market watchlist")
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
    symbol = f"{selected_symbol}/USDT"
    selected_tf = st.selectbox("Timeframe", ["15m", "1h"])

    chart_df = fetch_ohlcv(symbol, timeframe=selected_tf, limit=220)
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
                name=symbol,
            )
        )
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_9"], mode="lines", name="EMA 9"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_21"], mode="lines", name="EMA 21"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_50"], mode="lines", name="EMA 50"))
        fig.update_layout(title=f"{symbol} {selected_tf}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    row = df[df["Symbol"] == selected_symbol].iloc[0]
    price = safe_float(row.get("Price"))
    atr_pct = 0.65
    position_size = calculate_position_size(price, price * (atr_pct / 100), config.get("account_size", 10000), config.get("risk_per_trade_pct", 0.75))
    st.info(
        f"Setup: {row['Direction']} | Score {row['Score']} | Confidence {row['Confidence']}% | Expected win rate {row['Win Rate']}% | Suggested size {position_size} units"
    )
else:
    st.warning("No strong candidates found. The market is quiet or the score threshold is too strict for current conditions.")

st.divider()

st.subheader("Manual trade journal")
with st.form("trade_journal_form"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        journal_symbol = st.text_input("Symbol", value="BTC/USDT")
    with col_b:
        journal_direction = st.selectbox("Direction", ["LONG", "SHORT"])
    with col_c:
        journal_entry = st.number_input("Entry", value=0.0, format="%.4f")

    col_d, col_e, col_f = st.columns(3)
    with col_d:
        journal_exit = st.number_input("Exit", value=0.0, format="%.4f")
    with col_e:
        journal_result = st.number_input("Result %", value=0.0, format="%.2f")
    with col_f:
        journal_notes = st.text_input("Notes", value="")

    submitted = st.form_submit_button("Save trade")
    if submitted:
        symbol_text = journal_symbol.strip()
        if symbol_text:
            save_trade_to_journal(
                symbol=symbol_text,
                direction=journal_direction,
                entry=journal_entry,
                exit_price=journal_exit,
                result=journal_result,
                notes=journal_notes,
            )
            st.success(f"Trade saved for {symbol_text}.")

journal_df = load_journal()
if not journal_df.empty:
    st.dataframe(journal_df.tail(10), use_container_width=True)
else:
    st.caption("No journal entries yet.")

st.divider()

st.markdown(
    """
    ### Better reliability recommendations
    - Keep the market scan wide, but only trade the strongest-ranked setups
    - Use multi-timeframe confirmation before executing
    - Respect challenge risk limits strictly
    - Track actual outcomes to improve the model over time
    - Reduce exposure when BTC dominance or funding conditions are hostile
    - Separate scalp, swing, and challenge execution
    """
)

st.info("This app helps you scan the full active futures market, rank only the strongest setups, and manage risk with manual execution discipline. It does not guarantee profit or future outcomes.")

if mode == "Scalp":
    st.caption("Scalp mode: prioritize fast momentum, clean trend, volume confirmation, and small risk per trade.")
elif mode == "Swing":
    st.caption("Swing mode: favor stronger trend behavior, multi-timeframe confirmation, and better persistence.")
else:
    st.caption("Challenge mode: prioritize risk discipline, low exposure, and only the best-quality setups.")
