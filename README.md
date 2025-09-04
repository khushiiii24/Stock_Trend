<h1 align="center">📈 Stock Trend Predictor</h1>

<p align="center">
Predict tomorrow's stock trend (📈 Up / 📉 Down) using Machine Learning, Technical Indicators, and Real-Time Market Data.
</p>

---

## 🚀 Overview  
The **Stock Trend Predictor** is a Python-based project built in **PyCharm** that uses real stock data from **Yahoo Finance** to forecast whether the next day’s stock price will go **up** or **down**.  
It calculates key **technical indicators** like:
- **SMA5** (5-day Simple Moving Average)
- **SMA10** (10-day Simple Moving Average)
- **EMA20** (20-day Exponential Moving Average)
- **Volatility**

These features are fed into a **trained XGBoost model**, which outputs the prediction.  
The results are displayed in a beautiful, interactive **Streamlit dashboard** with:
- 📊 Price charts  
- 🧠 Model predictions  
- 🔍 Market sentiment visualization  
- 📰 Top news headlines (optional)  

---

## ✨ Features  

✅ **Real-Time Data Fetching** – using `yfinance`  
✅ **Technical Indicators** – SMA5, SMA10, EMA20, Volatility  
✅ **ML Model** – Trained with XGBoost for trend prediction  
✅ **Interactive Dashboard** – Built with Streamlit  
✅ **Visual Insights** – Candlestick charts & trend markers  
✅ **Easy Deployment** – Heroku / Streamlit Cloud / Render  

---

## 🛠️ Tech Stack  

| Category | Tools / Libraries |
|---------|-----------------|
| **IDE** | PyCharm 🟦 |
| **Language** | Python 🐍 |
| **Framework** | Streamlit 🎨 |
| **ML** | XGBoost 🤖 |
| **Data** | Yahoo Finance API 📡 |
| **Visualization** | Matplotlib 📊, Plotly 📈 |

---

## ⚡ Integration  
The project uses **Yahoo Finance API** via the `yfinance` Python library to fetch historical and live stock data.  
This adds **real-world relevance** and allows users to analyze **any stock ticker** dynamically.

---

## 📸 Preview  

<img width="2756" height="1547" alt="image" src="https://github.com/user-attachments/assets/9202e154-2f21-4a31-915b-d1cf92edc0d4" />


---

## 🏃‍♂️ Quick Start  

```bash
# 1️⃣ Clone the repository
git clone https://github.com/khushiiii24/Stock_Trend.git
cd Stock_Trend

# 2️⃣ Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Streamlit app
streamlit run dashboard.py
