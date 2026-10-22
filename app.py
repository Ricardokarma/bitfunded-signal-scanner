import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
OKX = "https://www.okx.com"
HEADERS = {"User-Agent": "BitfundedScanner/2.0"}
DEFAULT_WATCH = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK", "SUI", "ARB", "OP", "NEAR", "APT", "PEPE"]
MODE_THRESHOLDS = {"Scalp": 50, "Swing": 58, "Challenge": 68}
TF_MAP = {"15m": "15m", "1h": "1H"}


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def to_inst(sym):
    s = str(sym).upper().replace("/USDT", "").replace("-USDT-SWAP", "").replace("-USDT", "").replace("USDT", "")
    return f"{s}-USDT-SWAP"


def display_symbol(inst_id):
    return inst_id.replace("-USDT-SWAP", "").replace("-SWAP", "")


@st.cache_data(ttl=300)
def load_config():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


config = load_config()


def okx_get(path, params=None, timeout=12):
    try:
        r = requests.get(f"{OKX}{path}", params=params or {}, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return None
        data = r.json()
        if str(data.get("code")) != "0":
            return None
        return data.get("data")
    except Exception:
        return None


@st.cache_data(ttl=600)
def list_all_swap_symbols():
    data = okx_get("/api/v5/public/instruments", {"instType": "SWAP"})
    if not data:
        return DEFAULT_WATCH
    out = []
    for x in data:
        if x.get("settleCcy") == "USDT" and x.get("state") == "live":
            out.append(display_symbol(x["instId"]))
    return sorted(set(out)) or DEFAULT_WATCH


@st.cache_data(ttl=120)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8, headers=HEADERS)
        row = r.json().get("data", [{}])[0]
        return {"value": safe_float(row.get("value")), "label": row.get("value_classification", "N/A")}
    except Exception:
        return {"value": None, "label": "N/A"}


@st.cache_data(ttl=90)
def fetch_btc_dominance():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=8, headers=HEADERS)
        d = r.json().get("data", {})
        return {
            "btc_dom": safe_float(d.get("market_cap_percentage", {}).get("btc")),
            "mcap_change_24h": safe_float(d.get("market_cap_change_percentage_24h_usd")),
        }
    except Exception:
        return {"btc_dom": None, "mcap_change_24h": None}


@st.cache_data(ttl=90)
def fetch_long_short(ccy="BTC"):
    data = okx_get("/api/v5/rubik/stat/contracts/long-short-account-ratio", {"ccy": ccy, "period": "1H"})
    if not data:
        return None
    try:
        return safe_float(data[0][1])
    except Exception:
        return None


@st.cache_data(ttl=45)
def fetch_liq_heatmap(symbol):
    """MarginPad free buckets + clusters."""
    sym = display_symbol(to_inst(symbol))
    out = {"buckets": [], "clusters": []}
    try:
        r = requests.get(
            "https://marginpad.io/api/v1/liquidations/recent",
            params={"symbol": sym, "minutes": 720},
            timeout=12,
            headers=HEADERS,
        )
        if r.status_code == 200:
            out["buckets"] = r.json().get("buckets") or []
    except Exception:
        pass
    try:
        r = requests.get(
            "https://marginpad.io/api/v1/clusters",
            params={"symbol": sym},
            timeout=12,
            headers=HEADERS,
        )
        if r.status_code == 200:
            out["clusters"] = r.json().get("clusters") or []
    except Exception:
        pass
    return out


@st.cache_data(ttl=40)
def fetch_okx_liqs(symbol):
    uly = f"{display_symbol(to_inst(symbol))}-USDT"
    data = okx_get(
        "/api/v5/public/liquidation-orders",
        {"instType": "SWAP", "uly": uly, "state": "filled", "limit": "30"},
    )
    return data or []


@st.cache_data(ttl=20)
def fetch_ticker(inst_id):
    data = okx_get("/api/v5/market/ticker", {"instId": inst_id})
    if not data:
        return {"last": 0.0, "change": 0.0, "quoteVolume": 0.0}
    row = data[0]
    last = safe_float(row.get("last"))
    open24 = safe_float(row.get("open24h"))
    change = ((last - open24) / open24 * 100) if open24 else 0.0
    return {"last": last, "change": change, "quoteVolume": safe_float(row.get("volCcy24h"))}


@st.cache_data(ttl=25)
def fetch_ohlcv(inst_id, timeframe="15m", limit=120):
    bar = TF_MAP.get(timeframe, timeframe)
    data = okx_get("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": str(limit)})
    if not data:
        return pd.DataFrame()
    rows = []
    for item in data:
        rows.append({
            "timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True),
            "open": safe_float(item[1]),
            "high": safe_float(item[2]),
            "low": safe_float(item[3]),
            "close": safe_float(item[4]),
            "volume": safe_float(item[5]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("timestamp").set_index("timestamp")


@st.cache_data(ttl=40)
def fetch_funding(inst_id):
    data = okx_get("/api/v5/public/funding-rate", {"instId": inst_id})
    if not data:
        return 0.0
    return safe_float(data[0].get("fundingRate")) * 100


@st.cache_data(ttl=60)
def fetch_open_interest(inst_id):
    data = okx_get("/api/v5/public/open-interest", {"instType": "SWAP", "instId": inst_id})
    if not data:
        return 0.0
    return safe_float(data[0].get("oiCcy") or data[0].get("oi"))


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0.0)
    loss = -d.clip(upper=0.0)
    ag = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    al = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def _adx(h, l, c, n=14):
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / n, min_periods=n, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / n, min_periods=n, adjust=False).mean() / atr.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1 / n, min_periods=n, adjust=False).mean().fillna(0.0)


def _macd(s, fast=12, slow=26, signal=9):
    line = _ema(s, fast) - _ema(s, slow)
    return line, _ema(line, signal)


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
    out["volume_sma"] = out["volume"].rolling(20).mean()
    macd_line, sig = _macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = sig
    return out


def score_setup(df, mode):
    empty = {
        "direction": "NEUTRAL", "score": 0.0, "confidence": 0.0, "win_rate": 0.0,
        "reasons": ["No data"], "price": 0.0, "atr": 0.0, "rsi": 50.0, "regime": "RANGE",
        "vol_ratio": 1.0, "whale_flag": False,
    }
    if df.empty or len(df) < 50:
        return empty
    last = df.iloc[-1]
    bullish = bearish = 0.0
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
    vol_ratio = (vol / vol_sma) if vol_sma > 0 else 1.0
    whale_flag = vol_ratio >= 2.0
    wmap = {
        "Scalp": {"trend": 1.1, "mom": 1.35, "vol": 1.3},
        "Swing": {"trend": 1.35, "mom": 1.15, "vol": 1.05},
        "Challenge": {"trend": 1.5, "mom": 1.0, "vol": 1.1},
    }
    w = wmap.get(mode, wmap["Challenge"])
    if ema_9 > ema_21 > ema_50:
        bullish += 24 * w["trend"]; reasons.append("Bullish EMA stack")
    elif ema_9 < ema_21 < ema_50:
        bearish += 24 * w["trend"]; reasons.append("Bearish EMA stack")
    if 55 <= rsi <= 70:
        bullish += 16 * w["mom"]; reasons.append("RSI bullish")
    elif 30 <= rsi <= 45:
        bearish += 16 * w["mom"]; reasons.append("RSI bearish")
    elif rsi > 70:
        bullish += 7 * w["mom"]; reasons.append("Overbought")
    elif rsi < 30:
        bearish += 7 * w["mom"]; reasons.append("Oversold")
    if macd_v > macd_s:
        bullish += 14 * w["mom"]; reasons.append("MACD bullish")
    elif macd_v < macd_s:
        bearish += 14 * w["mom"]; reasons.append("MACD bearish")
    if adx >= 22:
        if price > ema_21:
            bullish += 12 * w["trend"]; reasons.append("ADX up")
        else:
            bearish += 12 * w["trend"]; reasons.append("ADX down")
    if vol_sma > 0:
        if vol_ratio >= 2.0:
            if bullish >= bearish:
                bullish += 14 * w["vol"]
            else:
                bearish += 14 * w["vol"]
            reasons.append(f"Whale vol x{vol_ratio:.1f}")
        elif vol_ratio >= 1.2:
            if bullish >= bearish:
                bullish += 10 * w["vol"]
            else:
                bearish += 10 * w["vol"]
            reasons.append("Volume expansion")
    score = round(min(100.0, max(bullish, bearish) + abs(bullish - bearish) * 0.45), 1)
    min_score = MODE_THRESHOLDS.get(mode, 55)
    if bullish > bearish and score >= min_score:
        direction = "LONG"
    elif bearish > bullish and score >= min_score:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    if ema_9 > ema_21 > ema_50 and adx > 22:
        regime = "TRENDING_UP"
    elif ema_9 < ema_21 < ema_50 and adx > 22:
        regime = "TRENDING_DOWN"
    else:
        regime = "RANGE"
    return {
        "direction": direction, "score": score,
        "confidence": round(min(99.0, 45 + score * 0.52), 1) if direction != "NEUTRAL" else 0.0,
        "win_rate": round(max(38.0, min(76.0, 48 + (score - 50) * 0.5)), 1) if direction != "NEUTRAL" else 0.0,
        "reasons": reasons[:6] if reasons else ["Weak"],
        "price": price, "atr": atr, "rsi": rsi, "regime": regime,
        "vol_ratio": round(vol_ratio, 2), "whale_flag": whale_flag,
    }


def apply_macro(direction, score, funding, fng_value, btc_change, ls_ratio, mode):
    notes, adj = [], 0.0
    if funding is not None:
        if direction == "LONG" and funding > 0.05:
            adj -= 8; notes.append("Funding overcrowded long")
        elif direction == "SHORT" and funding < -0.05:
            adj -= 8; notes.append("Funding overcrowded short")
        elif direction == "LONG" and funding < -0.02:
            adj += 5; notes.append("Funding helps LONG")
        elif direction == "SHORT" and funding > 0.02:
            adj += 5; notes.append("Funding helps SHORT")
    if fng_value is not None:
        if direction == "LONG" and fng_value <= 25:
            adj += 6; notes.append("Extreme Fear")
        elif direction == "SHORT" and fng_value >= 75:
            adj += 6; notes.append("Extreme Greed")
        elif direction == "LONG" and fng_value >= 80:
            adj -= 6; notes.append("Greed vs LONG")
        elif direction == "SHORT" and fng_value <= 20:
            adj -= 6; notes.append("Fear vs SHORT")
    if btc_change is not None and mode == "Challenge":
        if direction == "LONG" and btc_change < -3:
            adj -= 10; notes.append("BTC dumping")
        elif direction == "SHORT" and btc_change > 3:
            adj -= 10; notes.append("BTC pumping")
    if ls_ratio is not None:
        if direction == "SHORT" and ls_ratio > 1.8:
            adj += 4; notes.append("Crowd long-heavy")
        elif direction == "LONG" and ls_ratio < 0.7:
            adj += 4; notes.append("Crowd short-heavy")
    return clamp(round(score + adj, 1), 0, 100), notes


def build_signal(inst_id, mode, macro):
    setups = []
    for tf in ("15m", "1h"):
        df = fetch_ohlcv(inst_id, timeframe=tf, limit=120)
        if df.empty:
            continue
        s = score_setup(add_indicators(df), mode)
        s["timeframe"] = tf
        setups.append(s)
    if not setups:
        return {"direction": "NEUTRAL", "score": 0.0, "base_score": 0.0, "confidence": 0.0, "win_rate": 0.0,
                "reasons": ["No data"], "macro_notes": [], "timeframes": [], "price": 0.0, "atr": 0.0,
                "regime": "RANGE", "vol_ratio": 1.0, "whale_flag": False}
    avg = round(sum(x["score"] for x in setups) / len(setups), 1)
    longs = sum(1 for x in setups if x["direction"] == "LONG")
    shorts = sum(1 for x in setups if x["direction"] == "SHORT")
    min_s = MODE_THRESHOLDS.get(mode, 55)
    if longs >= shorts and avg >= min_s * 0.9:
        direction = "LONG"
    elif shorts > longs and avg >= min_s * 0.9:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    reasons = list(dict.fromkeys([r for s in setups for r in s["reasons"]]))[:6]
    final, macro_notes = apply_macro(
        direction, avg, macro.get("funding"), macro.get("fng"),
        macro.get("btc_change"), macro.get("ls_ratio"), mode,
    )
    if direction != "NEUTRAL" and final < min_s:
        direction = "NEUTRAL"
    return {
        "direction": direction, "score": final, "base_score": avg,
        "confidence": round(min(99.0, 42 + final * 0.55), 1) if direction != "NEUTRAL" else 0.0,
        "win_rate": round(max(38.0, min(76.0, 48 + (final - 50) * 0.5)), 1) if direction != "NEUTRAL" else 0.0,
        "reasons": reasons, "macro_notes": macro_notes,
        "timeframes": [x["timeframe"] for x in setups],
        "price": setups[-1]["price"], "atr": setups[-1]["atr"],
        "regime": setups[-1]["regime"], "vol_ratio": setups[-1]["vol_ratio"],
        "whale_flag": any(x.get("whale_flag") for x in setups),
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


def learning_status(journal_df):
    if journal_df is None or journal_df.empty:
        return "Learning: inativo (0 trades) 鈥� grava resultados em Advanced se quiseres calibrar."
    df = journal_df.copy()
    df["result"] = pd.to_numeric(df["result"], errors="coerce").fillna(0.0)
    n = len(df)
    wr = float((df["result"] > 0).mean()) * 100
    avg = float(df["result"].mean())
    level = "baixo" if n < 10 else ("medio" if n < 30 else "alto")
    return f"Learning: {level} 路 {n} trades 路 win rate {wr:.0f}% 路 avg {avg:+.2f}%"


def apply_learning(score, symbol, mode, direction, journal_df):
    if journal_df is None or journal_df.empty:
        return score
    df = journal_df.copy()
    df["direction"] = df["direction"].astype(str).str.upper()
    df["result"] = pd.to_numeric(df["result"], errors="coerce").fillna(0.0)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    sub = df[df["direction"] == direction]
    mult = 1.0
    if not sub.empty:
        mult *= clamp(1 + float(sub["result"].mean()) / 120.0, 0.85, 1.2)
    sym = symbol.upper()
    sub2 = df[df["symbol"] == sym]
    if not sub2.empty:
        mult *= clamp(1 + float(sub2["result"].mean()) / 120.0, 0.8, 1.25)
    return clamp(round(score * mult, 1), 0, 100)


def calculate_position_size(price, atr, account_size, risk_pct):
    price, atr = safe_float(price), safe_float(atr)
    account_size, risk_pct = safe_float(account_size), safe_float(risk_pct)
    if price <= 0 or atr <= 0 or account_size <= 0:
        return 0.0
    return round((account_size * (risk_pct / 100)) / (atr * 1.5), 4)


@st.cache_data(ttl=40, show_spinner=False)
def scan_symbols(mode, symbols_key, fng_value, btc_change, ls_ratio):
    symbols = [s for s in symbols_key.split("|") if s]
    rows = []
    for sym in symbols:
        inst = to_inst(sym)
        try:
            funding = fetch_funding(inst)
            macro = {"funding": funding, "fng": fng_value, "btc_change": btc_change, "ls_ratio": ls_ratio}
            signal = build_signal(inst, mode, macro)
            if signal["direction"] == "NEUTRAL":
                continue
            ticker = fetch_ticker(inst)
            oi = fetch_open_interest(inst)
            rows.append({
                "Symbol": display_symbol(inst),
                "Inst": inst,
                "Direction": signal["direction"],
                "Score": signal["score"],
                "Base": signal.get("base_score", signal["score"]),
                "Confidence": signal["confidence"],
                "Win Rate": signal["win_rate"],
                "Price": ticker.get("last") or signal.get("price", 0.0),
                "24h %": ticker.get("change", 0.0),
                "Funding %": funding,
                "OI": oi,
                "Vol x": signal.get("vol_ratio", 1.0),
                "Whale": "YES" if signal.get("whale_flag") else "",
                "Reasons": " | ".join(signal["reasons"]),
                "Macro": " | ".join(signal.get("macro_notes") or []) or "-",
                "Regime": signal.get("regime", "RANGE"),
                "ATR": signal.get("atr", 0.0),
            })
        except Exception:
            continue
    return sorted(rows, key=lambda x: x["Score"], reverse=True)


def plot_liq_heatmap(buckets, price, symbol):
    if not buckets:
        return None
    df = pd.DataFrame(buckets)
    if df.empty or "price" not in df.columns:
        return None
    df["long"] = df.get("long", 0).fillna(0)
    df["short"] = df.get("short", 0).fillna(0)
    # keep near price
    if price > 0:
        df = df[(df["price"] > price * 0.92) & (df["price"] < price * 1.08)]
    if df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df["price"], x=df["long"], orientation="h", name="Longs liq", marker_color="#ff6b6b"))
    fig.add_trace(go.Bar(y=df["price"], x=df["short"], orientation="h", name="Shorts liq", marker_color="#51cf66"))
    if price > 0:
        fig.add_hline(y=price, line_dash="dash", line_color="white", annotation_text="Price")
    fig.update_layout(
        title=f"Liquidation heatmap (MarginPad) 路 {symbol}",
        barmode="overlay", height=360, template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="USD liquidated", yaxis_title="Price",
    )
    return fig


# ================= UI =================
st.title("Bitfunded Signal Scanner Pro")
st.caption("OKX 路 Fear&Greed 路 Funding/OI 路 Whale volume 路 Liquidation heatmap 路 Favoritos + pesquisa")

all_symbols = list_all_swap_symbols()

if "account_size" not in st.session_state:
    st.session_state.account_size = float(config.get("account_size", 10000))
if "risk_pct" not in st.session_state:
    st.session_state.risk_pct = float(config.get("risk_per_trade_pct", 0.75))
if "mode" not in st.session_state:
    st.session_state.mode = "Challenge"
if "favorites" not in st.session_state:
    st.session_state.favorites = [s for s in DEFAULT_WATCH if s in all_symbols] or DEFAULT_WATCH[:10]

c1, c2, c3, c4 = st.columns([1.1, 1, 1.1, 1])
with c1:
    st.session_state.account_size = st.number_input("Account ($)", 1000.0, 500000.0, float(st.session_state.account_size), 1000.0)
with c2:
    st.session_state.risk_pct = st.number_input("Risk %", 0.25, 2.0, float(st.session_state.risk_pct), 0.25)
with c3:
    st.session_state.mode = st.selectbox("Mode", ["Scalp", "Swing", "Challenge"], index=["Scalp", "Swing", "Challenge"].index(st.session_state.mode))
with c4:
    st.write(""); st.write("")
    if st.button("Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Favorites + search
st.markdown("**Favoritos & pesquisa** (todos os perps USDT OKX ~ Bitfunded crypto pairs)")
fc1, fc2 = st.columns([2, 2])
with fc1:
    st.session_state.favorites = st.multiselect(
        "Favoritos (scan principal)",
        options=all_symbols,
        default=[f for f in st.session_state.favorites if f in all_symbols][:20],
    )
with fc2:
    extra = st.multiselect("Pesquisar / adicionar ao scan", options=all_symbols, default=[])

scan_list = sorted(set(st.session_state.favorites + extra))
if not scan_list:
    scan_list = DEFAULT_WATCH[:12]
    st.warning("Seleciona pelo menos 1 favorito.")

fng = fetch_fear_greed()
glob = fetch_btc_dominance()
ls_btc = fetch_long_short("BTC")
btc_tick = fetch_ticker("BTC-USDT-SWAP")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Account", f"${st.session_state.account_size:,.0f}")
m2.metric("Risk", f"{st.session_state.risk_pct}%")
m3.metric("Fear & Greed", f"{int(fng['value']) if fng['value'] else '鈥�'} ({fng['label']})")
m4.metric("BTC Dom", f"{glob['btc_dom']:.1f}%" if glob['btc_dom'] else "鈥�")
m5.metric("Mkt 24h", f"{glob['mcap_change_24h']:+.2f}%" if glob['mcap_change_24h'] is not None else "鈥�")

if btc_tick.get("last", 0) > 0:
    st.success(f"OKX online 路 BTC ${btc_tick['last']:,.1f} ({btc_tick['change']:+.2f}%) 路 L/S {ls_btc if ls_btc else '鈥�'}")
else:
    st.error("API offline 鈥� tenta Refresh")

journal_df = load_journal()
st.caption(learning_status(journal_df))

with st.spinner(f"Scan {len(scan_list)} pares 路 {st.session_state.mode}..."):
    results = scan_symbols(
        st.session_state.mode, "|".join(scan_list),
        fng.get("value"), btc_tick.get("change"), ls_btc,
    )

for r in results:
    r["Score"] = apply_learning(r["Score"], r["Symbol"], st.session_state.mode, r["Direction"], journal_df)

results = sorted(results, key=lambda x: x["Score"], reverse=True)
min_s = MODE_THRESHOLDS[st.session_state.mode]
results = [r for r in results if r["Score"] >= min_s]
longs = [r for r in results if r["Direction"] == "LONG"]
shorts = [r for r in results if r["Direction"] == "SHORT"]

st.subheader(f"Sugest玫es LONG ({len(longs)})")
if longs:
    st.dataframe(pd.DataFrame(longs)[["Symbol", "Score", "Confidence", "Price", "24h %", "Funding %", "Whale", "Macro", "Reasons"]],
                 use_container_width=True, hide_index=True)
else:
    st.caption("Sem LONG fortes neste momento.")

st.subheader(f"Sugest玫es SHORT ({len(shorts)})")
if shorts:
    st.dataframe(pd.DataFrame(shorts)[["Symbol", "Score", "Confidence", "Price", "24h %", "Funding %", "Whale", "Macro", "Reasons"]],
                 use_container_width=True, hide_index=True)
else:
    st.caption("Sem SHORT fortes neste momento.")

if results:
    pick = st.selectbox("Detalhe + heatmap liquida莽玫es", [r["Symbol"] for r in results])
    row = next(r for r in results if r["Symbol"] == pick)
    inst = row["Inst"]
    tf = st.radio("TF gr谩fico", ["15m", "1h"], horizontal=True)
    chart_df = fetch_ohlcv(inst, timeframe=tf, limit=120)
    if not chart_df.empty:
        chart_df = add_indicators(chart_df)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df["open"], high=chart_df["high"],
                                     low=chart_df["low"], close=chart_df["close"], name=pick))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_9"], mode="lines", name="EMA9", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_21"], mode="lines", name="EMA21", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_50"], mode="lines", name="EMA50", line=dict(width=1)))
        fig.update_layout(title=f"{pick} {row['Direction']} 路 {tf}", xaxis_rangeslider_visible=False,
                          height=400, template="plotly_dark", margin=dict(l=8, r=8, t=40, b=8))
        st.plotly_chart(fig, use_container_width=True)

    # Liquidation heatmap
    liq = fetch_liq_heatmap(pick)
    heat = plot_liq_heatmap(liq.get("buckets") or [], safe_float(row["Price"]), pick)
    if heat:
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.caption("Heatmap liquida莽玫es indispon铆vel para este par neste momento.")

    # nearest clusters
    clusters = liq.get("clusters") or []
    price = safe_float(row["Price"])
    if clusters and price > 0:
        near = sorted(clusters, key=lambda c: abs(safe_float(c.get("price")) - price))[:8]
        st.markdown("**Clusters de liquida莽茫o pr贸ximos**")
        st.dataframe(pd.DataFrame([{
            "Price": c.get("price"),
            "Side": c.get("side"),
            "Est. notional": round(safe_float(c.get("est_notional")), 1),
            "Dist %": round((safe_float(c.get("price")) - price) / price * 100, 2),
        } for c in near]), use_container_width=True, hide_index=True)

    okx_liqs = fetch_okx_liqs(pick)
    if okx_liqs:
        details = []
        for block in okx_liqs[:1]:
            for d in (block.get("details") or [])[:12]:
                details.append({
                    "side": d.get("posSide") or d.get("side"),
                    "price": d.get("bkPx"),
                    "size": d.get("sz"),
                    "time": d.get("time") or d.get("ts"),
                })
        if details:
            st.markdown("**脷ltimas liquida莽玫es OKX**")
            st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)

    size = calculate_position_size(row["Price"], row["ATR"] or row["Price"] * 0.008,
                                   st.session_state.account_size, st.session_state.risk_pct)
    st.info(f"**{row['Direction']} {pick}** 路 Score {row['Score']} 路 Conf {row['Confidence']}% 路 size 鈮� {size} 路 {row.get('Macro','')}")

# Hidden optional journal (collapsed)
with st.expander("Advanced 路 log manual opcional (learning)", expanded=False):
    st.caption("O learning s贸 calibra se registares resultados. Sem API da Bitfunded n茫o d谩 para ser 100% autom谩tico.")
    with st.form("quick_log"):
        a, b, c = st.columns(3)
        with a:
            js = st.text_input("Symbol", "BTC")
        with b:
            jd = st.selectbox("Dir", ["LONG", "SHORT"])
        with c:
            jr = st.number_input("Result %", value=0.0, format="%.2f")
        if st.form_submit_button("Guardar"):
            df = load_journal()
            df = pd.concat([df, pd.DataFrame([{
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": js, "direction": jd, "entry": 0, "exit": 0,
                "result": jr, "notes": "", "mode": st.session_state.mode,
            }])], ignore_index=True)
            df.to_csv(JOURNAL_PATH, index=False)
            st.success("OK")
            st.rerun()

st.caption("Bitfunded 茅 crypto-only 路 pares = perps USDT (OKX). Heatmap: MarginPad (gr谩tis). Dados de mercado: OKX.")
