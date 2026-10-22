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
        return yaml.safe_load(f) or {}


config = load_config()


@st.cache_resource
def get_exchange():
    return ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


def fetch_ohlcv(symbol, timeframe="15m", limit=220):
    exchange = get_exchange()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df.sort_index()
    except Exception as exc:
        st.error(f"Error fetching {symbol} ({timeframe}): {exc}")
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

    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame):
        out = pd.concat([out, macd], axis=1)

    out["volume_sma"] = out["volume"].rolling(window=20).mean()
    return out


def calculate_position_size(price, atr, account_size, risk_pct):
    if price <= 0 or atr <= 0 or account_size <= 0:
        return 0.0
    risk_amount = account_size * (risk_pct / 100)
    stop_distance = atr * 1.5
    if stop_distance <= 0:
        return 0.0
    size = risk_amount / stop_distance
    return round(float(size), 4)


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

    ema_9 = float(last.get("ema_9", 0) or 0)
    ema_21 = float(last.get("ema_21", 0) or 0)
    ema_50 = float(last.get("ema_50", 0) or 0)
    if ema_9 > ema_21 > ema_50:
        bullish_score += 25
        reasons.append("Bullish EMA stack")
    elif ema_9 < ema_21 < ema_50:
        bearish_score += 25
        reasons.append("Bearish EMA stack")

    rsi = float(last.get("rsi", 0) or 0)
    if 55 <= rsi <= 70:
        bullish_score += 18
        reasons.append("RSI in bullish zone")
    elif 30 <= rsi <= 45:
        bearish_score += 18
        reasons.append("RSI in bearish zone")
    elif rsi > 70:
        bullish_score += 8
        reasons.append("Strong momentum")
    elif rsi < 30:
        bearish_score += 8
        reasons.append("Oversold momentum")

    macd_col = "MACD_12_26_9" if "MACD_12_26_9" in df.columns else None
    signal_col = "MACDs_12_26_9" if "MACDs_12_26_9" in df.columns else None
    if macd_col and signal_col:
        macd_val = float(last.get(macd_col, 0) or 0)
        signal_val = float(last.get(signal_col, 0) or 0)
        if macd_val > signal_val:
            bullish_score += 15
            reasons.append("MACD above signal")
        elif macd_val < signal_val:
            bearish_score += 15
            reasons.append("MACD below signal")

    adx = float(last.get("adx", 0) or 0)
    if adx >= 25:
        if last["close"] > ema_21:
            bullish_score += 12
            reasons.append("Trend is up")
        else:
            bearish_score += 12
            reasons.append("Trend is down")

    volume = float(last.get("volume", 0) or 0)
    volume_sma = float(last.get("volume_sma", 0) or 0)
    if volume_sma > 0:
        if volume > volume_sma * 1.2:
            bullish_score += 10
            reasons.append("Bullish volume expansion")
        elif volume < volume_sma * 0.8:
            bearish_score += 10
            reasons.append("Weak volume")

    atr = float(last.get("atr", 0) or 0)
    price = float(last.get("close", 0) or 0)
    atr_pct = (atr / price) * 100 if price > 0 and atr > 0 else 0.0

    score = max(bullish_score, bearish_score) + abs(bullish_score - bearish_score) * 0.5
    score = min(100.0, round(score, 1))

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
    results = []
    timeframe_list = config.get("timeframes", ["15m"])
    if isinstance(timeframe_list, str):
        timeframe_list = [timeframe_list]

    pairs = config.get("pairs", {})
    for symbol in pairs:
        for timeframe in timeframe_list:
            df = fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
            if df.empty:
                continue
            df = add_indicators(df)
            setup = score_setup(df)
            if setup["direction"] != "NEUTRAL" and setup["score"] >= 55:
                results.append(
                    {
                        "Symbol": symbol.replace(":USDT", ""),
                        "Direction": setup["direction"],
                        "Score": setup["score"],
                        "Price": setup["price"],
                        "RSI": round(setup["rsi"], 1),
                        "ATR %": setup["atr_pct"],
                        "Timeframe": timeframe,
                        "Reasons": " • ".join(setup["reasons"]),
                    }
                )
    return sorted(results, key=lambda item: item["Score"], reverse=True)


st.title("📈 Bitfunded Signal Scanner")
st.caption("Manual signals only • Designed for Bitfunded challenge risk management")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Account Size", f"${config.get('account_size', 0):,}")
with col2:
    st.metric("Risk per Trade", f"{config.get('risk_per_trade_pct', 0)}%")
with col3:
    if st.button("🔄 Refresh", type="primary"):
        st.cache_data.clear()
        st.rerun()

st.divider()

results = scan_pairs()
if results:
    df_results = pd.DataFrame(results)
    st.subheader("Ranked Opportunities")
    st.dataframe(
        df_results,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "Price": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    selected_symbol = st.selectbox("Choose a symbol to inspect", df_results["Symbol"].unique())
    symbol = selected_symbol + "/USDT"
    timeframe = st.selectbox("Timeframe", config.get("timeframes", ["15m"]))
    chart_df = fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
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
        fig.update_layout(title=f"{symbol} {timeframe}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No high-quality setups found right now. Conditions are too weak or the market is quiet.")

st.divider()
st.info(
    "Important: This tool is for manual trading signal review only. Always respect your daily loss limits, max loss caps, and minimum hold rules before executing any trade."
)

