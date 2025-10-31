# main.py — GlobaLedge
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# optional OpenAI client (not required for dashboard)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ===============================================================
# CONFIG
# ===============================================================
API_BASE = "https://api.frankfurter.app"
st.set_page_config(page_title="💱 GlobaLedge — Currency & Stocks Dashboard", layout="wide")

# ===============================================================
# HELPERS — Currency Functions
# ===============================================================
@st.cache_data(ttl=3600)
def fetch_symbols() -> list:
    try:
        r = requests.get(f"{API_BASE}/currencies", timeout=10)
        r.raise_for_status()
        data = r.json()
        return sorted(list(data.keys()))
    except Exception:
        return sorted(["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "INR", "CNY", "NZD"])


@st.cache_data(ttl=1800)
def fetch_latest(base: str = "USD") -> dict:
    try:
        resp = requests.get(f"{API_BASE}/latest", params={"from": base}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {"date": data.get("date"), "base": data.get("base", base), "rates": data.get("rates", {})}
    except Exception as e:
        st.warning(f"⚠️ Live rate fetch failed: {e}")
        return {"date": None, "base": base, "rates": {}}


@st.cache_data(ttl=3600)
def fetch_history(base: str, targets: list, start: date, end: date):
    if not targets:
        return pd.DataFrame(columns=["date", "currency", "rate"])
    try:
        symbols = ",".join(targets)
        url = f"{API_BASE}/{start.strftime('%Y-%m-%d')}..{end.strftime('%Y-%m-%d')}"
        r = requests.get(url, params={"from": base, "to": symbols}, timeout=15)
        r.raise_for_status()
        data = r.json()
        rates = data.get("rates", {})
        df = pd.DataFrame(rates).T.reset_index().melt(id_vars="index", var_name="currency", value_name="rate")
        df.rename(columns={"index": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["currency", "date"])
    except Exception as e:
        st.error(f"❌ Failed to fetch historical data: {e}")
        return pd.DataFrame(columns=["date", "currency", "rate"])


def convert_currency(from_currency, to_currency, amount):
    try:
        if from_currency == to_currency:
            return amount
        r = requests.get(
            f"{API_BASE}/latest",
            params={"amount": amount, "from": from_currency, "to": to_currency},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        return data["rates"].get(to_currency)
    except Exception:
        return None


# ===============================================================
# STOCK HELPERS — No .NS/.BO auto-append
# ===============================================================
def safe_download_single(ticker: str, period="1mo", interval="1d"):
    """Download data exactly for the given ticker (no suffix manipulation)."""
    ticker_clean = ticker.strip().upper()
    if not ticker_clean:
        return pd.DataFrame()
    try:
        df = yf.download(ticker_clean, period=period, interval=interval, progress=False, auto_adjust=True)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def build_price_dataframe_for_chart(tickers: list, period="1mo", interval="1d"):
    """Return tidy DataFrame (Date, Ticker, Close) for multi-ticker plotting."""
    if not tickers:
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])
    try:
        df = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(columns=["Date", "Ticker", "Close"])

        # Flatten multiindex columns if multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        df = df.reset_index()

        # Pick Close columns
        close_cols = [c for c in df.columns if "Close" in c]
        if not close_cols:
            return pd.DataFrame(columns=["Date", "Ticker", "Close"])

        df_melt = df.melt(id_vars="Date", value_vars=close_cols, var_name="Ticker", value_name="Close")
        df_melt["Ticker"] = df_melt["Ticker"].str.replace("Close_", "", regex=False).str.replace("_Close", "", regex=False)
        return df_melt[["Date", "Ticker", "Close"]]
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])


def fetch_latest_for_portfolio(ticker: str):
    """Return (last_price, currency) for given ticker."""
    df = safe_download_single(ticker, period="1d", interval="1d")
    if df.empty or "Close" not in df.columns:
        return (None, None)
    price = float(df["Close"].iloc[-1])
    try:
        info = yf.Ticker(ticker).info
        currency = info.get("currency", "USD")
    except Exception:
        currency = "USD"
    return (price, currency)


# ===============================================================
# SIDEBAR SETTINGS
# ===============================================================
st.sidebar.title("⚙️ Settings")

refresh_interval = st.sidebar.slider("Auto-refresh interval (sec)", 15, 600, 60, step=15)
st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")

all_currencies = fetch_symbols()
base_currency = st.sidebar.selectbox("🌍 Base Currency", all_currencies, index=all_currencies.index("USD"))

# ===============================================================
# TABS
# ===============================================================
tabs = st.tabs(["💱 Currency Dashboard", "📈 Stocks Tracker"])

# ===============================================================
# TAB 1: CURRENCY DASHBOARD
# ===============================================================
with tabs[0]:
    st.header("💱 Currency Dashboard")
    st.caption("Live exchange rates, converter, and charts — powered by Frankfurter API")

    latest = fetch_latest(base_currency)
    rates = latest["rates"]

    if not rates:
        st.error("❌ No exchange rates available.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("💰 Currency Converter")
        amount = st.number_input("Amount", min_value=0.0, value=100.0, format="%.4f")
        from_curr = st.selectbox("From", all_currencies, index=all_currencies.index(base_currency))
        to_curr = st.selectbox("To", all_currencies, index=all_currencies.index("EUR") if "EUR" in all_currencies else 0)

        if st.button("Convert"):
            result = convert_currency(from_curr, to_curr, amount)
            if result is not None:
                one_unit = convert_currency(from_curr, to_curr, 1.0)
                st.metric(
                    label=f"{amount:.2f} {from_curr} → {to_curr}",
                    value=f"{result:.4f}",
                    delta=f"1 {from_curr} = {one_unit:.6f} {to_curr}" if one_unit else "",
                )
            else:
                st.error("Conversion failed. Try again.")

    with col2:
        st.subheader("📊 Compare Exchange Rates")
        compare_targets = st.multiselect(
            "Select currencies to compare",
            all_currencies,
            default=[c for c in ["EUR", "GBP", "JPY", "INR"] if c in all_currencies],
        )
        if compare_targets:
            df_comp = pd.DataFrame([{"Currency": c, "Rate": rates.get(c)} for c in compare_targets])
            st.dataframe(df_comp.style.format({"Rate": lambda x: f"{x:.6f}" if x else "N/A"}))
        else:
            st.info("Select one or more currencies to compare.")

    st.divider()
    st.subheader("📈 Historical Exchange Rates")

    chart_col1, chart_col2 = st.columns([1, 2])
    with chart_col1:
        target_for_chart = st.multiselect("Currencies to chart (max 5)", all_currencies, default=["EUR", "GBP"], max_selections=5)
        view_all = st.checkbox("📅 View all available history (since 1999)")
        days = st.slider("Days back", 10, 365, 90)
        end_date = date.today() - timedelta(days=1)
        start_date = date(1999, 1, 1) if view_all else (end_date - timedelta(days=days))

    with chart_col2:
        if target_for_chart:
            df_hist = fetch_history(base_currency, target_for_chart, start_date, end_date)
            if not df_hist.empty:
                fig = px.line(df_hist, x="date", y="rate", color="currency",
                              title=f"Exchange Rates — Base: {base_currency}", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available.")
        else:
            st.info("Select currencies to view trend.")

# ===============================================================
# TAB 2: STOCKS TRACKER
# ===============================================================
with tabs[1]:
    st.header("📈 Stocks Portfolio & Price Calculator")
    st.caption("Live stock data powered by Yahoo Finance (yfinance)")

    # --- Calculator ---
    st.subheader("🧮 Stock Price Calculator")
    calc_ticker = st.text_input("Stock symbol (e.g. RELIANCE.NS, ^NSEI, AAPL)", "RELIANCE.NS").upper()
    calc_shares = st.number_input("Number of Shares", min_value=1, value=10)

    if st.button("Calculate Price"):
        df = safe_download_single(calc_ticker, period="1d", interval="1d")
        if not df.empty and "Close" in df.columns:
            price_native = float(df["Close"].iloc[-1])
            info = yf.Ticker(calc_ticker).info
            stock_currency = info.get("currency", "USD")
            rate_to_target = convert_currency(stock_currency, base_currency, 1) if stock_currency != base_currency else 1
            total_value = price_native * rate_to_target * calc_shares
            st.success(f"💰 {calc_shares} × {calc_ticker} = {total_value:,.2f} {base_currency}")
        else:
            st.error("No price data found.")

    st.divider()
    st.markdown("### 💼 Portfolio Setup")

    tickers = [t.strip().upper() for t in st.text_input(
        "Enter stock symbols (comma-separated)", "RELIANCE.NS, ^NSEI, GOOGL"
    ).split(",") if t.strip()]

    portfolio = {
        t: st.number_input(f"{t} shares", min_value=0, value=10, step=1, key=f"shares_{t}")
        for t in tickers
    }

    st.divider()

    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.markdown("#### 📊 Stock Data Options")
        period = st.selectbox("Data period", ["5d", "1mo", "3mo", "6mo", "1y", "2y"], index=2)
        interval = st.selectbox("Data interval", ["1d", "1wk", "1mo"], index=0)

    with right_col:
        st.markdown(f"#### 📈 Stock Prices — Base: {base_currency}")
        if tickers:
            df_all = build_price_dataframe_for_chart(tickers, period=period, interval=interval)
            if not df_all.empty:
                fig = px.line(df_all, x="Date", y="Close", color="Ticker",
                              title=f"Stock Price Trends (Base: {base_currency})", markers=True)
                fig.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data found for tickers.")
        else:
            st.info("Enter at least one ticker.")

    st.divider()
    if tickers:
        latest_prices = []
        for t in tickers:
            price, currency = fetch_latest_for_portfolio(t)
            if price is None:
                continue
            shares = portfolio.get(t, 0)
            rate_to_base = convert_currency(currency, base_currency, 1) if currency != base_currency else 1
            value_converted = price * shares * rate_to_base
            latest_prices.append({"Ticker": t, "Last Price": price, "Currency": currency,
                                  "Shares": shares, f"Value ({base_currency})": value_converted})
        if latest_prices:
            df_summary = pd.DataFrame(latest_prices)
            total_value = df_summary[f"Value ({base_currency})"].sum()
            df_summary["% of Portfolio"] = (df_summary[f"Value ({base_currency})"] / total_value * 100).round(2)
            st.metric(f"Total Portfolio Value ({base_currency})", f"{total_value:,.2f} {base_currency}")

            c1, c2 = st.columns([2, 1])
            with c1:
                st.dataframe(df_summary.style.format({
                    f"Value ({base_currency})": "{:,.2f}", "% of Portfolio": "{:.2f}%"
                }), use_container_width=True)
            with c2:
                fig_pie = px.pie(df_summary, names="Ticker", values=f"Value ({base_currency})", hole=0.5)
                fig_pie.update_traces(textinfo="label+percent", textposition="inside")
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- AI Portfolio Assistant (optional) ---
            st.divider()
            st.markdown("## 🤖 AI Portfolio Assistant")
            st.caption("Ask questions or get summaries about your holdings (optional).")

            openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
            if OpenAI is None:
                st.info("OpenAI package not installed — AI assistant disabled. (pip install openai)")
            elif not openai_api_key:
                st.info("No OpenAI key in Streamlit secrets — AI assistant disabled.")
            else:
                client = OpenAI(api_key=openai_api_key)
                if "messages" not in st.session_state:
                    st.session_state["messages"] = [
                        {"role": "system", "content": "You are a helpful financial assistant."},
                        {"role": "assistant", "content": "Hi! I can analyze your portfolio. Ask me anything about diversification or performance."}
                    ]

                user_msg = st.chat_input("Ask about your portfolio...")
                if user_msg:
                    st.session_state["messages"].append({"role": "user", "content": user_msg})
                    with st.spinner("Analyzing..."):
                        portfolio_text = df_summary.to_string(index=False)
                        total_value_text = f"{total_value:,.2f} {base_currency}"
                        ai_prompt = f"Portfolio Data:\n{portfolio_text}\n\nTotal Value: {total_value_text}\nBase currency: {base_currency}\nQuestion: {user_msg}"
                        messages_for_model = st.session_state["messages"] + [{"role": "user", "content": ai_prompt}]
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=messages_for_model,
                                max_tokens=400,
                            )
                            reply = response.choices[0].message.content
                            st.session_state["messages"].append({"role": "assistant", "content": reply})
                        except Exception as e:
                            st.error(f"AI assistant failed: {e}")

                for msg in st.session_state.get("messages", [])[1:]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

st.markdown("---")
st.warning("FOOTER & DISCLAMERS")
st.caption("Made by Sabeeir Sharrma and Aaryan Bayala | github.com/sabeeirsharrma/GlobaLedge")
st.caption("Pwoered by: Streamlit, Frankfurter API (currencies), Yahoo Finance (stocks) & OpenAI (AI Chatbot)")
st.caption("This application is licensed under the GNU AGPL License - Any redistribution requires disclosure of original source (github.com/sabeeirsharrma/GlobaLedge).")
st.caption("Feedback? Report issues at: github.com/sabeeirsharrma/GlobaLedge/issues")
st.caption("Note: This app is for educational and informational purposes only. Do your own research before making financial decisions.")
st.caption("Disclaimer: Stock data may be delayed by up to 15 minutes. Always verify with official sources.")
st.caption("Data sources are for informational purposes only. Not financial advice.")
