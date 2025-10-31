# 🌍 GlobaLedge

**GlobaLedge** is an all-in-one **financial dashboard** that brings together **live currency exchange rates**, **stock tracking**, and an **AI-powered portfolio assistant** — giving you a clear, intelligent view of your global financial edge. - Made by [Sabeeir Sharrma](https://github.com/SabeeirSharrma) and [Aaryan Bayala](https://github.com/Aaryan792) under [ANS SOLUTIONS](https://github.com/ANS-Solutions/)

Also available at [ANS Solutions github](https://github.com/ANS-Solutions/GlobaLedge)

https://img.shields.io/github/license/SabeeirSharrma/GlobaLedge

[Discord](https://discord.gg/Z3tvfdvqkG)


---

## 🚀 Features

### 💱 Currency Dashboard
- View **live exchange rates** powered by the [Frankfurter API](https://www.frankfurter.app/).  
- Convert between **any two global currencies** in real-time.  
- Compare multiple currencies against your **base currency**.  
- Visualize **historical exchange trends** with interactive charts.  
- “**View all time history**” option for long-term analysis.  

### 📈 Stocks Tracker
- Track **live stock prices** using [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`.  
- Configure your **personal portfolio** by specifying shares per stock.  
- View portfolio value in your **selected base currency** (auto-converted).  
- Display **interactive price trends** and **portfolio distribution pie charts**.  
- Supports **Indian stocks** (e.g., `RELIANCE.NS`, `TCS.NS`) and global tickers (`AAPL`, `TSLA`, etc.).  

### 🤖 AI Portfolio Assistant
- Built-in **AI financial advisor** using the [OpenAI API](https://platform.openai.com/).  
- Ask questions about diversification, exposure, or portfolio health.  
- Provides contextual responses using your **actual holdings**.  
- Chat interface powered by `st.chat_message()` with memory persistence.  

### ⚙️ Smart UI & Customization
- Adjustable **auto-refresh interval** (15s–10min).  
- Unified **base currency selector** for both stocks and currencies.  
- Compact, dual-column layout for charts and data tables.  
- Designed with **Streamlit’s modern layout system** for a clean, responsive interface.  

---

## 🧩 Tech Stack

| Component | Description |
|------------|--------------|
| **Python 3.10+** | Core language |
| **Streamlit** | Web UI framework |
| **yfinance** | Stock market data |
| **Frankfurter API** | Real-time & historical forex data |
| **OpenAI API** | AI assistant |
| **Plotly** | Interactive data visualization |
| **Pandas** | Data handling |
| **streamlit-autorefresh** | Live refresh timer |

---

## 🛠️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/SabeeirSharrma/GlobaLedge.git
cd GlobaLedge
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key
**In `.streamlit/secrets.toml`**
```bash
OPENAI_API_KEY = "your_openai_api_key_here"
```

### 4. Run the app
**Do `streamlit run main.py` in app folder and a new page should open up in your browser**

## Env. Variables

| Key              | Description                                                               |
| ---------------- | ------------------------------------------------------------------------- |
| `OPENAI_API_KEY` | API key for the AI assistant                                              |

## 🧠 AI Assistant Capabilities

+  **You can ask:**

+  **“Summarize my portfolio.”**

+  **“Which stock contributes the most to my total value?”**

+  **“How diversified is my portfolio?”**

+  **“What’s my exposure to tech stocks?”**

+  **The assistant uses your live portfolio data for context-aware insights.**

## 🌐 Supported Markets

**Global currencies (USD, EUR, GBP, INR, JPY, etc.)**

**Stocks from:**

-  **NYSE / NASDAQ**

-  **NSE / BSE (India)**

-  **LSE, TSE, and others supported by Yahoo Finance**

## 📸 Credits

**Created by:**

-  **🧑‍💻 Sabeeir Sharrma**
-  **🧑‍💻 Aaryan Bayala**

## 🪙 License

This project is licensed under the **GNU AGPL** License
