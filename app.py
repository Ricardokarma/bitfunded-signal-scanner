import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml

st.set_page_config(page_title="Bitfunded Scanner", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

JOURNAL_PATH = "trades_journal.csv"
OKX = "https://www.okx.com"
HEADERS = {"User-Agent": "BitfundedScanner/3.1"}
DEFAULT_WATCH = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK", "SUI", "ARB", "OP", "NEAR", "APT", "PEPE", "WIF", "DOT", "ATOM", "FIL", "INJ"]
MODE_THRESHOLDS = {"Scalp": 48, "Swing": 56, "Challenge": 65}
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


@st.cache_data(ttl=900)
def list_instruments():
    data = okx_get("/api/v5/public/instruments", {"instType": "SWAP"})
    out = {}
    if not data:
        return out
    for x in data:
        if x.get("settleCcy") == "USDT" and x.get("state") == "live":
            sym = display_symbol(x["instId"])
            out[sym] = {
                "instId": x["instId"],
                "ctVal": safe_float(x.get("ctVal"), 1),
                "ctMult": safe_float(x.get("ctMult"), 1),
                "lotSz": safe_float(x.get("lotSz"), 0.01),
                "minSz": safe_float(x.get("minSz"), 0.01),
                "tickSz": safe_float(x.get("tickSz"), 0.01),
            }
    return out


def round_lot(qty, lot_sz):
    if lot_sz <= 0:
        return round(qty, 4)
    steps = int(qty / lot_sz)
    return round(steps * lot_sz, 8)


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


@st.cache_data(ttl=45)
def fetch_liq_heatmap(symbol):
    out = {"buckets": [], "clusters": []}
    try:
        r = requests.get("https://marginpad.io/api/v1/liquidations/recent", params={"symbol": symbol, "minutes": 720}, timeout=12, headers=HEADERS)
        if r.status_code == 200:
            out["buckets"] = r.json().get("buckets") or []
    except Exception:
        pass
    try:
        r = requests.get("https://marginpad.io/api/v1/clusters", params={"symbol": symbol}, timeout=12, headers=HEADERS)
        if r.status_code == 200:
            out["clusters"] = r.json().get("clusters") or []
    except Exception:
        pass
    return out


@st.cache_data(ttl=20)
def fetch_ticker(inst_id):
    data = okx_get("/api/v5/market/ticker", {"instId": inst_id})
    if not data:
        return {"last": 0.0, "change": 0.0}
    row = data[0]
    last = safe_float(row.get("last"))
    open24 = safe_float(row.get("open24h"))
    change = ((last - open24) / open24 * 100) if open24 else 0.0
    return {"last": last, "change": change}


@st.cache_data(ttl=25)
def fetch_ohlcv(inst_id, timeframe="15m", limit=120):
    bar = TF_MAP.get(timeframe, timeframe)
    data = okx_get("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": str(limit)})
    if not data:
        return pd.DataFrame()
    rows = [{
        "timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True),
        "open": safe_float(item[1]), "high": safe_float(item[2]),
        "low": safe_float(item[3]), "close": safe_float(item[4]),
        "volume": safe_float(item[5]),
    } for item in data]
    df = pd.DataFrame(rows)
    return df.sort_values("timestamp").set_index("timestamp") if not df.empty else df


@st.cache_data(ttl=40)
def fetch_funding(inst_id):
    data = okx_get("/api/v5/public/funding-rate", {"instId": inst_id})
    if not data:
        return 0.0
    return safe_float(data[0].get("fundingRate")) * 100


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
    empty = {"direction": "NEUTRAL", "score": 0.0, "confidence": 0.0, "price": 0.0, "atr": 0.0, "rsi": 50.0, "reasons": []}
    if df.empty or len(df) < 50:
        return empty
    last = df.iloc[-1]
    bullish = bearish = 0.0
    reasons = []
    ema_9, ema_21, ema_50 = safe_float(last.get("ema_9")), safe_float(last.get("ema_21")), safe_float(last.get("ema_50"))
    rsi = safe_float(last.get("rsi"), 50)
    adx = safe_float(last.get("adx"), 0)
    price = safe_float(last.get("close"), 0)
    atr = safe_float(last.get("atr"), 0)
    macd_v, macd_s = safe_float(last.get("macd")), safe_float(last.get("macd_signal"))
    vol = safe_float(last.get("volume"))
    vol_sma = safe_float(last.get("volume_sma"))
    vol_ratio = (vol / vol_sma) if vol_sma > 0 else 1.0
    w = {"Scalp": 1.2, "Swing": 1.1, "Challenge": 1.0}.get(mode, 1.0)

    if ema_9 > ema_21 > ema_50:
        bullish += 24 * w; reasons.append("EMA up")
    elif ema_9 < ema_21 < ema_50:
        bearish += 24 * w; reasons.append("EMA down")
    if 55 <= rsi <= 70:
        bullish += 16; reasons.append("RSI bull")
    elif 30 <= rsi <= 45:
        bearish += 16; reasons.append("RSI bear")
    if macd_v > macd_s:
        bullish += 14
    else:
        bearish += 14
    if adx >= 22:
        bullish += 12 if price > ema_21 else 0
        bearish += 12 if price <= ema_21 else 0
    if vol_ratio >= 1.3:
        if bullish >= bearish:
            bullish += 10
        else:
            bearish += 10

    score = round(min(100.0, max(bullish, bearish) + abs(bullish - bearish) * 0.4), 1)
    min_score = MODE_THRESHOLDS.get(mode, 55)

    rev = 0.0
    if rsi <= 32:
        rev += 30; reasons.append("RSI oversold")
    if rsi <= 25:
        rev += 12
    try:
        recent_low = float(df["low"].tail(20).min())
        if price > 0 and (price - recent_low) / price <= 0.012:
            rev += 14; reasons.append("Near low")
    except Exception:
        pass
    if macd_v > macd_s and rsi < 45:
        rev += 12; reasons.append("MACD turn")
    rev = round(min(100.0, rev), 1)

    if bullish > bearish and score >= min_score:
        direction = "LONG"
    elif bearish > bullish and score >= min_score:
        direction = "SHORT"
    elif rev >= max(48, min_score - 8):
        direction = "LONG"
        score = max(score, rev)
        reasons.append("REVERSAL")
    else:
        direction = "NEUTRAL"

    conf = round(min(92.0, 38 + score * 0.5), 1) if direction != "NEUTRAL" else 0.0
    return {"direction": direction, "score": score, "confidence": conf, "price": price, "atr": atr, "rsi": rsi, "reasons": reasons[:5]}


def trade_levels(price, atr, direction):
    price, atr = safe_float(price), safe_float(atr)
    if price <= 0:
        return {}
    if atr <= 0:
        atr = price * 0.008
    if direction == "LONG":
        sl, tp1, tp2 = price - 1.5 * atr, price + 1.5 * atr, price + 3.0 * atr
    elif direction == "SHORT":
        sl, tp1, tp2 = price + 1.5 * atr, price - 1.5 * atr, price - 3.0 * atr
    else:
        return {}
    risk = abs(price - sl)
    return {
        "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2,
        "rr1": round(abs(tp1 - price) / risk, 2) if risk else 0,
        "rr2": round(abs(tp2 - price) / risk, 2) if risk else 0,
        "sl_pct": round(risk / price * 100, 2),
        "tp1_pct": round(abs(tp1 - price) / price * 100, 2),
        "tp2_pct": round(abs(tp2 - price) / price * 100, 2),
    }


def institutional_size(price, sl, account, risk_pct, inst_meta):
    """OKX linear contracts from risk budget."""
    price = safe_float(price)
    sl = safe_float(sl)
    account = safe_float(account)
    risk_pct = safe_float(risk_pct)
    stop = abs(price - sl)
    risk_usd = account * (risk_pct / 100.0)
    if stop <= 0 or risk_usd <= 0 or price <= 0:
        return {"contracts": 0.0, "notional": 0.0, "risk_usd": risk_usd, "ok": False, "msg": "invalid stop"}

    ct_val = safe_float(inst_meta.get("ctVal"), 1.0)
    ct_mult = safe_float(inst_meta.get("ctMult"), 1.0)
    lot = safe_float(inst_meta.get("lotSz"), 0.01)
    min_sz = safe_float(inst_meta.get("minSz"), lot)

    # USDT loss per 1 contract ≈ ctVal * ctMult * stop_distance
    loss_per_contract = ct_val * ct_mult * stop
    if loss_per_contract <= 0:
        return {"contracts": 0.0, "notional": 0.0, "risk_usd": risk_usd, "ok": False, "msg": "bad ctVal"}

    raw = risk_usd / loss_per_contract
    contracts = round_lot(raw, lot)
    if contracts < min_sz:
        contracts = min_sz if risk_usd >= loss_per_contract * min_sz * 0.5 else 0.0
        if contracts == 0:
            return {"contracts": 0.0, "notional": 0.0, "risk_usd": risk_usd, "ok": False, "msg": "risk too small for minSz"}

    notional = contracts * ct_val * ct_mult * price
    return {
        "contracts": contracts,
        "notional": round(notional, 2),
        "risk_usd": round(risk_usd, 2),
        "loss_per_contract": round(loss_per_contract, 4),
        "ok": True,
        "msg": "ok",
        "ctVal": ct_val,
        "lotSz": lot,
    }


def apply_macro(direction, score, funding, fng, btc_chg, mode):
    adj, notes = 0.0, []
    if funding is not None:
        if direction == "LONG" and funding > 0.05:
            adj -= 6; notes.append("funding crowded")
        elif direction == "SHORT" and funding < -0.05:
            adj -= 6; notes.append("funding crowded")
        elif direction == "SHORT" and funding > 0.02:
            adj += 4
        elif direction == "LONG" and funding < -0.02:
            adj += 4
    if fng is not None:
        if direction == "LONG" and fng <= 30:
            adj += 5; notes.append("fear bounce")
        elif direction == "SHORT" and fng >= 75:
            adj += 5; notes.append("greed fade")
    if btc_chg is not None and mode == "Challenge":
        if direction == "LONG" and btc_chg < -3:
            adj -= 4
        elif direction == "SHORT" and btc_chg > 3:
            adj -= 8
    return clamp(round(score + adj, 1), 0, 100), notes


def build_signal(inst_id, mode, macro):
    setups = []
    for tf in ("15m", "1h"):
        df = fetch_ohlcv(inst_id, timeframe=tf, limit=120)
        if df.empty:
            continue
        s = score_setup(add_indicators(df), mode)
        setups.append(s)
    if not setups:
        return None
    avg = round(sum(x["score"] for x in setups) / len(setups), 1)
    longs = sum(1 for x in setups if x["direction"] == "LONG")
    shorts = sum(1 for x in setups if x["direction"] == "SHORT")
    min_s = MODE_THRESHOLDS.get(mode, 55)
    best = max(setups, key=lambda x: x["score"])
    if longs >= shorts and avg >= min_s * 0.85:
        direction = "LONG"
    elif shorts > longs and avg >= min_s * 0.9:
        direction = "SHORT"
    elif best["direction"] == "LONG" and best["score"] >= min_s - 5:
        direction = "LONG"
        avg = best["score"]
    else:
        return None
    final, notes = apply_macro(direction, avg, macro.get("funding"), macro.get("fng"), macro.get("btc_change"), mode)
    if final < min_s - (8 if direction == "LONG" else 0):
        return None
    reasons = list(dict.fromkeys([r for s in setups for r in s["reasons"]]))[:5]
    price = setups[-1]["price"]
    atr = setups[-1]["atr"]
    conf = round(min(92.0, 38 + final * 0.5), 1)
    return {
        "direction": direction, "score": final, "confidence": conf,
        "price": price, "atr": atr, "reasons": reasons, "macro": notes,
        "rsi": setups[-1]["rsi"],
    }


def load_journal():
    """Always safe — never breaks the UI."""
    try:
        from pathlib import Path as _P
        path = _P("trades_journal.csv")
        if path.is_file():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def learning_line(df):
    if df is None or df.empty:
        return "Learning: off (0 trades)"
    r = pd.to_numeric(df.get("result"), errors="coerce").fillna(0.0)
    n = len(r)
    wr = float((r > 0).mean()) * 100 if n else 0
    return f"Learning: {n} trades | WR {wr:.0f}% | avg {float(r.mean()):+.2f}%"


@st.cache_data(ttl=40, show_spinner=False)
def scan_symbols(mode, symbols_key, fng_value, btc_change):
    symbols = [s for s in symbols_key.split("|") if s]
    rows = []
    for sym in symbols:
        inst = to_inst(sym)
        try:
            funding = fetch_funding(inst)
            sig = build_signal(inst, mode, {"funding": funding, "fng": fng_value, "btc_change": btc_change})
            if not sig:
                continue
            ticker = fetch_ticker(inst)
            price = ticker.get("last") or sig["price"]
            lv = trade_levels(price, sig["atr"] or price * 0.008, sig["direction"])
            rows.append({
                "Symbol": display_symbol(inst),
                "Direction": sig["direction"],
                "Score": sig["score"],
                "Conf": sig["confidence"],
                "Price": price,
                "Change": ticker.get("change", 0.0),
                "Entry": lv.get("entry"),
                "SL": lv.get("sl"),
                "TP1": lv.get("tp1"),
                "TP2": lv.get("tp2"),
                "RR1": lv.get("rr1"),
                "RR2": lv.get("rr2"),
                "SL_pct": lv.get("sl_pct"),
                "ATR": sig["atr"],
                "Reasons": " | ".join(sig["reasons"]),
            })
        except Exception:
            continue
    return sorted(rows, key=lambda x: x["Score"], reverse=True)


def fmt_px(x):
    x = safe_float(x)
    if x == 0:
        return "-"
    if x >= 1000:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


# ===================== UI =====================
st.markdown("## Bitfunded Signal Scanner")
st.caption("OKX contracts | Challenge risk | Top Longs/Shorts | Entry SL TP1 TP2 | Heatmap")

instruments = list_instruments()
all_symbols = sorted(instruments.keys()) or DEFAULT_WATCH

for k, v in {
    "account_size": float(config.get("account_size", 10000)),
    "risk_pct": float(config.get("risk_per_trade_pct", 0.75)),
    "profit_target": 10.0,
    "daily_loss": float(config.get("daily_loss_limit_pct", 4.0)),
    "max_dd": float(config.get("max_loss_limit_pct", 6.0)),
    "max_positions": int(config.get("max_open_positions", 2)),
    "mode": "Challenge",
    "min_score_ui": 60,
    "favorites": [s for s in DEFAULT_WATCH if s in all_symbols][:15],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

tab_scan, tab_sniper, tab_risk = st.tabs(["Top setups", "Sniper (1 token)", "Challenge & risk"])

with tab_risk:
    st.subheader("Challenge Management & Risk")
    r1, r2 = st.columns(2)
    with r1:
        st.session_state.account_size = st.number_input("CHALLENGE SIZE ($)", 1000.0, 500000.0, float(st.session_state.account_size), 1000.0)
        st.session_state.profit_target = st.number_input("PROFIT TARGET (%)", 1.0, 30.0, float(st.session_state.profit_target), 0.5)
        st.session_state.risk_pct = st.number_input("RISK PER TRADE (%)", 0.25, 2.0, float(st.session_state.risk_pct), 0.25)
    with r2:
        st.session_state.daily_loss = st.number_input("DAILY LOSS LIMIT (%)", 1.0, 10.0, float(st.session_state.daily_loss), 0.5)
        st.session_state.max_dd = st.number_input("MAX DRAWDOWN (%)", 2.0, 20.0, float(st.session_state.max_dd), 0.5)
        st.session_state.max_positions = st.number_input("MAX OPEN POSITIONS", 1, 10, int(st.session_state.max_positions), 1)
    st.session_state.mode = st.selectbox("DEFAULT SCAN MODE", ["Scalp", "Swing", "Challenge"], index=["Scalp", "Swing", "Challenge"].index(st.session_state.mode))
    tgt = st.session_state.account_size * st.session_state.profit_target / 100
    dstop = st.session_state.account_size * st.session_state.daily_loss / 100
    mdd = st.session_state.account_size * st.session_state.max_dd / 100
    st.success(f"Target ${tgt:,.0f} | Daily stop ${dstop:,.0f} | Max DD ${mdd:,.0f} | Max positions {st.session_state.max_positions}")
    st.caption("Never risk more than the daily loss limit across open trades.")

with tab_scan:
    st.session_state.favorites = st.multiselect(
        "Watchlist",
        options=all_symbols,
        default=[f for f in st.session_state.favorites if f in all_symbols][:20],
    )
    extra = st.multiselect("Search more tokens", options=all_symbols, default=[])
    scan_list = sorted(set(st.session_state.favorites + extra)) or DEFAULT_WATCH[:12]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state.mode = st.selectbox("Mode", ["Scalp", "Swing", "Challenge"], index=["Scalp", "Swing", "Challenge"].index(st.session_state.mode), key="mode_scan")
    with c2:
        st.session_state.min_score_ui = st.slider("Min score", 40, 90, int(st.session_state.min_score_ui), 1)
    with c3:
        st.write(""); st.write("")
        if st.button("Refresh scan", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with c4:
        st.write(""); st.write("")
        if st.button("Clear cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    fng = fetch_fear_greed()
    glob = fetch_btc_dominance()
    btc = fetch_ticker("BTC-USDT-SWAP")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Account", f"${st.session_state.account_size:,.0f}")
    m2.metric("Risk/trade", f"{st.session_state.risk_pct}%")
    m3.metric("Fear&Greed", f"{int(fng['value']) if fng['value'] else '-'} ({fng['label']})")
    m4.metric("BTC", f"${btc['last']:,.0f}" if btc.get("last") else "-", delta=f"{btc.get('change', 0):+.1f}%")
    if btc.get("last"):
        dom = f"{glob['btc_dom']:.1f}%" if glob.get("btc_dom") else "-"
        st.success(f"OKX online | BTC {btc['change']:+.2f}% | Dom {dom}")
    st.caption("Learning: optional (see Advanced)")

    with st.spinner(f"Scanning {len(scan_list)} pairs..."):
        results = scan_symbols(st.session_state.mode, "|".join(scan_list), fng.get("value"), btc.get("change"))

    min_s = max(MODE_THRESHOLDS[st.session_state.mode], int(st.session_state.min_score_ui))
    results = [r for r in results if r["Score"] >= min_s]
    longs = [r for r in results if r["Direction"] == "LONG"][:10]
    shorts = [r for r in results if r["Direction"] == "SHORT"][:10]

    def render_card(r):
        meta = instruments.get(r["Symbol"], {"ctVal": 1, "ctMult": 1, "lotSz": 0.01, "minSz": 0.01})
        sizing = institutional_size(r["Entry"], r["SL"], st.session_state.account_size, st.session_state.risk_pct, meta)
        contracts = sizing["contracts"] if sizing["ok"] else 0
        st.markdown(
            f"**{r['Symbol']} ({r['Conf']:.0f}%) · {r['Direction']}**  \n"
            f"Entry: `{fmt_px(r['Entry'])}` | SL: `{fmt_px(r['SL'])}` | TP1: `{fmt_px(r['TP1'])}` | TP2: `{fmt_px(r['TP2'])}`  \n"
            f"Score {r['Score']:.0f} | 24h {r['Change']:+.1f}% | "
            f"contracts **{contracts}** | notional ${sizing.get('notional', 0):,.0f} | "
            f"risk ${sizing.get('risk_usd', 0):.0f} | R:R {r.get('RR1')}/{r.get('RR2')}  \n"
            f"{r['Reasons']}"
        )
        st.divider()

    st.markdown(f"### Top Longs ({len(longs)}) | min score {min_s}")
    if longs:
        for r in longs:
            render_card(r)
    else:
        st.caption("No strong LONGs. In a dump use Scalp or wait for RSI oversold / REVERSAL.")

    st.markdown(f"### Top Shorts ({len(shorts)}) | min score {min_s}")
    if shorts:
        for r in shorts:
            render_card(r)
    else:
        st.caption("No STRONGs matching this filter.")

    st.info(f"Challenge max open positions: **{st.session_state.max_positions}**. Do not open more than this at once.")

    pool = longs + shorts
    if pool:
        pick = st.selectbox("Detail + heatmap", [r["Symbol"] for r in pool])
        row = next(r for r in pool if r["Symbol"] == pick)
        inst = to_inst(pick)
        tf = st.radio("TF", ["15m", "1h"], horizontal=True, key="tf_detail")
        cdf = fetch_ohlcv(inst, timeframe=tf, limit=120)
        if not cdf.empty:
            cdf = add_indicators(cdf)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=cdf.index, open=cdf["open"], high=cdf["high"], low=cdf["low"], close=cdf["close"], name=pick))
            fig.add_trace(go.Scatter(x=cdf.index, y=cdf["ema_9"], mode="lines", name="EMA9", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=cdf.index, y=cdf["ema_21"], mode="lines", name="EMA21", line=dict(width=1)))
            for y, name, col in [(row.get("Entry"), "Entry", "#fff"), (row.get("SL"), "SL", "#ff6b6b"), (row.get("TP1"), "TP1", "#51cf66"), (row.get("TP2"), "TP2", "#2f9e44")]:
                if y:
                    fig.add_hline(y=y, line_dash="dot", line_color=col, annotation_text=name)
            fig.update_layout(title=f"{pick} {row['Direction']}", xaxis_rangeslider_visible=False, height=400, template="plotly_dark", margin=dict(l=8, r=8, t=40, b=8))
            st.plotly_chart(fig, use_container_width=True)

        liq = fetch_liq_heatmap(pick)
        bdf = pd.DataFrame(liq.get("buckets") or [])
        price = safe_float(row["Price"])
        if not bdf.empty and price > 0 and "price" in bdf.columns:
            bdf = bdf[(bdf["price"] > price * 0.92) & (bdf["price"] < price * 1.08)]
            if not bdf.empty:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(y=bdf["price"], x=bdf.get("long", 0), orientation="h", name="Longs liq", marker_color="#ff6b6b"))
                fig2.add_trace(go.Bar(y=bdf["price"], x=bdf.get("short", 0), orientation="h", name="Shorts liq", marker_color="#51cf66"))
                fig2.add_hline(y=price, line_dash="dash", line_color="white", annotation_text="Price")
                fig2.update_layout(title=f"Liquidation heatmap | {pick}", barmode="overlay", height=320, template="plotly_dark", margin=dict(l=8, r=8, t=40, b=8))
                st.plotly_chart(fig2, use_container_width=True)

with tab_sniper:
    st.subheader("Institutional setup | 1 token")
    tok = st.selectbox("SELECT TOKEN", options=all_symbols, index=all_symbols.index("BTC") if "BTC" in all_symbols else 0)
    direction = st.radio("DIRECTION", ["LONG", "SHORT"], horizontal=True)
    inst = to_inst(tok)
    tick = fetch_ticker(inst)
    price = tick.get("last") or 0.0
    st.metric("CURRENT MARKET PRICE", f"${fmt_px(price)}")
    if st.button("CALCULATE INSTITUTIONAL SETUP", type="primary"):
        df = fetch_ohlcv(inst, "15m", 120)
        atr = 0.0
        if not df.empty:
            atr = safe_float(add_indicators(df).iloc[-1].get("atr"))
        lv = trade_levels(price, atr, direction)
        meta = instruments.get(tok, {"ctVal": 1, "ctMult": 1, "lotSz": 0.01, "minSz": 0.01})
        sizing = institutional_size(lv.get("entry", price), lv.get("sl", price), st.session_state.account_size, st.session_state.risk_pct, meta)
        if lv:
            a, b, c, d = st.columns(4)
            a.metric("Entry", fmt_px(lv["entry"]))
            b.metric("Stop Loss", fmt_px(lv["sl"]), delta=f"{lv['sl_pct']}%")
            c.metric("TP1 (partial)", fmt_px(lv["tp1"]), delta=f"{lv['tp1_pct']}%")
            d.metric("TP2 (runner)", fmt_px(lv["tp2"]), delta=f"{lv['tp2_pct']}%")
            st.markdown(
                f"""
**{direction} {tok}**
- Contracts (OKX): **{sizing.get('contracts', 0)}** (ctVal={meta.get('ctVal')}, lot={meta.get('lotSz')})
- Notional: **${sizing.get('notional', 0):,.2f}**
- Risk: **${sizing.get('risk_usd', 0):.2f}** ({st.session_state.risk_pct}% of ${st.session_state.account_size:,.0f})
- R:R TP1 **{lv['rr1']}** | TP2 **{lv['rr2']}**
- Plan: close ~50% at TP1, move SL to entry, runner to TP2
- Challenge: daily loss {st.session_state.daily_loss}% | max DD {st.session_state.max_dd}% | max pos {st.session_state.max_positions}
"""
            )
            if not sizing.get("ok"):
                st.warning(f"Size note: {sizing.get('msg')}")

with st.expander("Advanced | manual trade log (learning)", expanded=False):
    with st.form("log"):
        a, b, c = st.columns(3)
        js = a.text_input("Symbol", "BTC")
        jd = b.selectbox("Dir", ["LONG", "SHORT"])
        jr = c.number_input("Result %", value=0.0, format="%.2f")
        if st.form_submit_button("Save"):
            df = load_journal()
            row = {"timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "symbol": js, "direction": jd, "result": jr, "mode": st.session_state.mode}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(Path(JOURNAL_PATH), index=False)
            st.success("Saved")
            st.rerun()

st.caption("Bitfunded crypto perps | OKX contract sizing | MarginPad heatmap | not financial advice")
