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


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@st.cache_data(ttl=30)
def fetch_coingecko_btc_dominance():
    try:
        response = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        response.raise_for_status()
        data = response.json()
        return safe_float(data.get("data", {}).get("market_cap_percentage", {}).get("btc", 0.0), 0.0)
    except Exception:
        return None


@st.cache_data(ttl=15)
def fetch_ticker(symbol):
    exchange = get_exchange()
    try:
        ticker = exchange.fetch_ticker(symbol)
        return {
            "last": safe_float(ticker.get("last")),
            "change": safe_float(ticker.get("percentage"), 0.0),
            "baseVolume": safe_float(ticker.get("baseVolume"), 0.0),
            "quoteVolume": safe_float(ticker.get("quoteVolume"), 0.0),
            "bid": safe_float(ticker.get("bid"), 0.0),
            "ask": safe_float(ticker.get("ask"), 0.0),
        }
    except Exception:
        return {
            "last": 0.0,
            "change": 0.0,
            "baseVolume": 0.0,
            "quoteVolume": 0.0,
            "bid": 0.0,
            "ask": 0.0,
        }


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
        result["longShortRatio"] = safe_float(long_short.get("ratio"), 0.0)
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
        for col in macd.columns:
            out[col] = macd[col]

    return out


def estimate_win_rate(score, trend_strength, volatility_penalty):
    base = 50 + (score - 50) * 0.6 + trend_strength
    estimate = max(35.0, min(78.0, base - volatility_penalty))
    return round(estimate, 1)


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

    price = safe_float(last.get("close"), 0)
    atr = safe_float(last.get("atr"), 0)
    atr_pct = ((atr / price) * 100) if price > 0 and atr > 0 else 0.0

    trend_strength = abs(bullish_score - bearish_score) * 0.7
    volatility_penalty = 0.0 if atr_pct < 0.7 else (atr_pct - 0.7) * 2

    score = max(bullish_score, bearish_score) + abs(bullish_score - bearish_score) * 0.5
    score = round(min(100.0, score), 1)

    if bullish_score > bearish_score and score >= 55:
        direction = "LONG"
    elif bearish_score > bullish_score and score >= 55:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    if direction == "LONG":
        confidence = min(99.0, 40 + score * 0.6)
    elif direction == "SHORT":
        confidence = min(99.0, 40 + score * 0.6)
    else:
        confidence = 0.0

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


@st.cache_data(ttl=30)
def scan_pairs():
    pair_list = config.get("pairs", [])
    if isinstance(pair_list, dict):
        pair_list = list(pair_list.keys())

    results = []
    timeframes = config.get("timeframes", ["15m"])
    if isinstance(timeframes, str):
        timeframes = [timeframes]

    for symbol in pair_list:
        for timeframe in timeframes:
            df = fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
            if df.empty:
                continue

            df = add_indicators(df)
            setup = score_setup(df)
            if setup["direction"] != "NEUTRAL" and setup["score"] >= 55:
                ticker = fetch_ticker(symbol)
                funding = fetch_funding_and_oi(symbol)
                results.append(
                    {
                        "Symbol": symbol.replace("/USDT", ""),
                        "Direction": setup["direction"],
                        "Score": setup["score"],
                        "Confidence": setup["confidence"],
                        "Win Rate": setup["win_rate"],
                        "Price": setup["price"],
                        "RSI": round(setup["rsi"], 1),
                        "ATR %": setup["atr_pct"],
                        "Last Price": ticker.get("last", 0.0),
                        "24h %": ticker.get("change", 0.0),
                        "Volume": ticker.get("quoteVolume", 0.0),
                        "Funding %": funding.get("fundingRate", 0.0),
                        "OI": funding.get("openInterest", 0.0),
                        "Timeframe": timeframe,
                        "Reasons": " • ".join(setup["reasons"]),
                    }
                )

    return sorted(results, key=lambda row: row["Score"], reverse=True)


st.title("📈 Bitfunded Signal Scanner Pro")
st.caption("Real-time market scanner for scalps and 1–5 day crypto trades. Historical probability model only — not guaranteed income.")

btc_dom = fetch_coingecko_btc_dominance()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Account Size", f"${config.get('account_size', 0):,}")
with col2:
    st.metric("Risk / Trade", f"{config.get('risk_per_trade_pct', 0)}%")
with col3:
    st.metric("BTC Dominance", f"{btc_dom:.1f}%" if btc_dom is not None else "N/A")
with col4:
    if st.button("Refresh market scan", type="primary"):
        st.cache_data.clear()
        st.rerun()

st.divider()

results = scan_pairs()

if results:
    df_results = pd.DataFrame(results)
    st.subheader("Ranked opportunities")
    st.dataframe(
        df_results,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f"),
            "Win Rate": st.column_config.ProgressColumn("Win Rate", min_value=0, max_value=100, format="%.1f"),
            "Price": st.column_config.NumberColumn(format="%.4f"),
            "Last Price": st.column_config.NumberColumn(format="%.4f"),
            "24h %": st.column_config.NumberColumn(format="%.2f"),
            "Funding %": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    selected_symbol = st.selectbox("Inspect symbol", df_results["Symbol"].unique())
    selected_tf = st.selectbox("Timeframe", config.get("timeframes", ["15m"]))
    chart_df = fetch_ohlcv(f"{selected_symbol}/USDT", timeframe=selected_tf, limit=220)

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
                name=f"{selected_symbol}/USDT",
            )
        )
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_9"], mode="lines", name="EMA 9"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_21"], mode="lines", name="EMA 21"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_50"], mode="lines", name="EMA 50"))
        fig.update_layout(title=f"{selected_symbol}/USDT {selected_tf}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    summary_row = df_results[df_results["Symbol"] == selected_symbol].iloc[0]
    price = safe_float(summary_row.get("Last Price")) or safe_float(summary_row.get("Price"))
    atr_pct = safe_float(summary_row.get("ATR %"))
    account_size = safe_float(config.get("account_size"), 10000)
    risk_pct = safe_float(config.get("risk_per_trade_pct"), 0.75)
    position_size = calculate_position_size(price, price * (atr_pct / 100), account_size, risk_pct)

    st.info(
        f"Model result: {summary_row['Direction']} | Score {summary_row['Score']} | Confidence {summary_row['Confidence']}% | Expected win rate {summary_row['Win Rate']}% | Suggested size {position_size} units"
    )
else:
    st.warning("No strong trade candidates are active right now. Use a wider market filter or lower the score threshold.")

st.divider()

st.subheader("Market structure overview")
summary = {
    "Long setups": int((df_results["Direction"] == "LONG").sum()) if "df_results" in locals() else 0,
    "Short setups": int((df_results["Direction"] == "SHORT").sum()) if "df_results" in locals() else 0,
    "Avg score": round(df_results["Score"].mean(), 1) if "df_results" in locals() and not df_results.empty else 0.0,
    "Avg win rate": round(df_results["Win Rate"].mean(), 1) if "df_results" in locals() and not df_results.empty else 0.0,
}

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Long setups", summary["Long setups"])
with col_b:
    st.metric("Short setups", summary["Short setups"])
with col_c:
    st.metric("Avg score", f"{summary['Avg score']:.1f}")
with col_d:
    st.metric("Avg win rate", f"{summary['Avg win rate']:.1f}%")

st.divider()
st.markdown(
    """
    ### How to make this more reliable
    - Use more than one timeframe: 15m, 1h, 4h, and daily
    - Add liquidity + spread filters to avoid bad execution quality
    - Add a market regime filter: trending vs ranging
    - Add a sector/market-side filter: BTC dominance, funding, OI, liquidations
    - Use a smaller set of high-quality pairs instead of every coin
    - Track real trade outcomes and update the model with live performance data
    - Do not overtrade when the market is weak or noisy
    - Keep risk per trade small and consistent
    """
)

st.info("This app is a decision-support tool for manual execution. It does not guarantee successful trades or future profits. Use it to improve discipline, risk control, and market filtering.")


# Optional: allows the app to stay responsive in a local dev environment
# Keep a tiny delay only when a refresh is explicitly requested.
if st.session_state.get("_manual_refresh"):
    time.sleep(0.2)
    st.session_state["_manual_refresh"] = False
