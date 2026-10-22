import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import yaml
import plotly.graph_objects as go

st.set_page_config(page_title="Bitfunded Signal Scanner", page_icon="📈", layout="wide")


@st.cache_data
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


config = load_config()


@st.cache_resource
def get_exchange():
    return ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


def safe_number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def fetch_ohlcv(symbol, timeframe="15m", limit=220):
    exchange = get_exchange()
    try:
        bars = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            bars,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
        return df
    except Exception as exc:
        st.warning(f"Unable to fetch {symbol} on {timeframe}: {exc}")
        return pd.DataFrame()


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


def calculate_position_size(price, atr, account_size, risk_pct):
    price = safe_number(price)
    atr = safe_number(atr)
    account_size = safe_number(account_size)
    risk_pct = safe_number(risk_pct)

    if price <= 0 or atr <= 0 or account_size <= 0:
        return 0.0

    risk_amount = account_size * (risk_pct / 100)
    stop_distance = atr * 1.5
    if stop_distance <= 0:
        return 0.0
    return round(risk_amount / stop_distance, 4)


def score_setup(df):
    if df.empty:
        return {
            "direction": "NEUTRAL",
            "score": 0,
            "reasons": ["No data"],
            "price": 0.0,
            "atr": 0.0,
            "rsi": 0.0,
            "atr_pct": 0.0,
        }

    last = df.iloc[-1]
    bullish_score = 0
    bearish_score = 0
    reasons = []

    ema_9 = safe_number(last.get("ema_9"))
    ema_21 = safe_number(last.get("ema_21"))
    ema_50 = safe_number(last.get("ema_50"))

    if ema_9 > ema_21 > ema_50:
        bullish_score += 25
        reasons.append("Bullish EMA stack")
    elif ema_9 < ema_21 < ema_50:
        bearish_score += 25
        reasons.append("Bearish EMA stack")

    rsi = safe_number(last.get("rsi"), 50)
    if 55 <= rsi <= 70:
        bullish_score += 18
        reasons.append("RSI bullish")
    elif 30 <= rsi <= 45:
        bearish_score += 18
        reasons.append("RSI bearish")
    elif rsi > 70:
        bullish_score += 10
        reasons.append("Momentum strong")
    elif rsi < 30:
        bearish_score += 10
        reasons.append("Momentum weak")

    macd_value = safe_number(last.get("MACD_12_26_9"))
    signal_value = safe_number(last.get("MACDs_12_26_9"))
    if macd_value > signal_value:
        bullish_score += 12
        reasons.append("MACD above signal")
    elif macd_value < signal_value:
        bearish_score += 12
        reasons.append("MACD below signal")

    adx = safe_number(last.get("adx"))
    if adx >= 25:
        if last.get("close", 0) > ema_21:
            bullish_score += 12
            reasons.append("Trend confirmed")
        else:
            bearish_score += 12
            reasons.append("Trend broken")

    volume = safe_number(last.get("volume"))
    volume_sma = safe_number(last.get("volume_sma"))
    if volume_sma > 0:
        if volume > volume_sma * 1.2:
            bullish_score += 10
            reasons.append("Volume expansion")
        elif volume < volume_sma * 0.8:
            bearish_score += 10
            reasons.append("Weak volume")

    price = safe_number(last.get("close"), 0)
    atr = safe_number(last.get("atr"), 0)
    atr_pct = ((atr / price) * 100) if price > 0 and atr > 0 else 0.0

    score = max(bullish_score, bearish_score) + abs(bullish_score - bearish_score) * 0.5
    score = round(min(100.0, score), 1)

    if bullish_score > bearish_score and score >= 55:
        direction = "LONG"
    elif bearish_score > bullish_score and score >= 55:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "score": score,
        "reasons": reasons if reasons else ["Weak setup"],
        "price": price,
        "atr": atr,
        "rsi": rsi,
        "atr_pct": round(atr_pct, 2),
    }


@st.cache_data
def scan_pairs():
    pairs = config.get("pairs", [])
    if isinstance(pairs, dict):
        pairs = list(pairs.keys())

    results = []
    timeframes = config.get("timeframes", ["15m"])
    if isinstance(timeframes, str):
        timeframes = [timeframes]

    for symbol in pairs:
        for timeframe in timeframes:
            df = fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
            if df.empty:
                continue
            df = add_indicators(df)
            setup = score_setup(df)
            if setup["direction"] != "NEUTRAL" and setup["score"] >= 55:
                results.append(
                    {
                        "Symbol": symbol.replace("/USDT", ""),
                        "Direction": setup["direction"],
                        "Score": setup["score"],
                        "Price": setup["price"],
                        "RSI": round(setup["rsi"], 1),
                        "ATR %": setup["atr_pct"],
                        "Timeframe": timeframe,
                        "Reasons": " • ".join(setup["reasons"]),
                    }
                )

    return sorted(results, key=lambda row: row["Score"], reverse=True)


st.title("📈 Bitfunded Signal Scanner")
st.caption("Manual trading signal dashboard for crypto pairs and Bitfunded challenge risk rules.")

with st.sidebar:
    st.header("Controls")
    st.metric("Account Size", f"${config.get('account_size', 0):,}")
    st.metric("Risk / Trade", f"{config.get('risk_per_trade_pct', 0)}%")
    st.metric("Max Open Positions", config.get("max_open_positions", 0))
    st.metric("Min Hold", f"{config.get('min_hold_minutes', 12)} min")

    if st.button("Refresh Signals", type="primary"):
        st.cache_data.clear()
        st.rerun()

st.subheader("Live opportunities")
results = scan_pairs()

if results:
    df_results = pd.DataFrame(results)
    st.dataframe(
        df_results,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "Price": st.column_config.NumberColumn(format="%.4f"),
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

    position_size = calculate_position_size(
        price=chart_df["close"].iloc[-1],
        atr=chart_df["atr"].iloc[-1],
        account_size=config.get("account_size", 10000),
        risk_pct=config.get("risk_per_trade_pct", 0.75),
    )

    st.info(
        f"Suggested position size for {selected_symbol}/USDT: {position_size} contracts or units based on ATR and account risk."
    )
else:
    st.warning("No high-quality setups are active right now. Conditions are weak or the market is quiet.")

st.divider()
st.subheader("Manual trade checklist")
st.markdown(
    """
    - Confirm the market trend aligns with the signal
    - Respect daily loss limit and max loss rule
    - Check the minimum hold time before entry
    - Only execute if the setup still makes sense after price action confirmation
    - Keep a journal of entry, reason, exit, and outcome
    """
)

