 =====================================================
# NSE STOCK PRICE PREDICTOR - STREAMLIT APP
#
# Install:  pip install -r requirements_app.txt
# Run:      streamlit run nse_app.py
# Opens in your browser at http://localhost:8501
# =====================================================

import re

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="NSE Stock Predictor", page_icon="📈", layout="centered")

FEATURES = ["Return", "Return_1", "Return_2", "SMA5_gap", "SMA20_gap",
            "Volatility", "HL_range", "Volume_ratio", "RSI"]


# ---------------------------------------------------------------
# The ML pipeline (cached for 1 hour so repeat lookups are instant)
# ---------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def run_pipeline(symbol):
    # 1. Data loading
    ticker = symbol.upper().strip()
    if not ticker.endswith(".NS"):
        ticker += ".NS"
    df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the NSE symbol and try again.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Data exploration + quality (before cleaning)
    raw_rows = len(df)
    explore = df[["Open", "High", "Low", "Close", "Volume"]].describe().round(2)
    quality = {
        "Rows downloaded": raw_rows,
        "Missing values": int(df.isnull().sum().sum()),
        "Duplicate dates": int(df.index.duplicated().sum()),
        "Zero-volume days": int((df["Volume"] == 0).sum()),
    }
    df = df[~df.index.duplicated()]
    df = df[df["Volume"] > 0]
    df = df.ffill().dropna()
    quality["Rows after cleaning"] = len(df)
    if len(df) < 250:
        raise ValueError(f"Not enough history for '{ticker}' (need about a year of trading data).")

    # 3. Features
    df["Return"] = df["Close"].pct_change()
    df["Return_1"] = df["Return"].shift(1)
    df["Return_2"] = df["Return"].shift(2)
    df["SMA5_gap"] = df["Close"] / df["Close"].rolling(5).mean() - 1
    df["SMA20_gap"] = df["Close"] / df["Close"].rolling(20).mean() - 1
    df["Volatility"] = df["Return"].rolling(10).std()
    df["HL_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Volume_ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()
    change = df["Close"].diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    df["Target"] = df["Close"].shift(-1) / df["Close"] - 1   # tomorrow's % change

    today_features = df[FEATURES].iloc[[-1]]
    data = df.dropna()
    X, y = data[FEATURES], data["Target"]

    # 4. Time-based split (no shuffling)
    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 5. Feature selection (training data only)
    selector = SelectKBest(f_regression, k=6).fit(X_train, y_train)
    chosen = [f for f, keep in zip(FEATURES, selector.get_support()) if keep]
    X_train, X_test = X_train[chosen], X_test[chosen]

    # 6. Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 7. Three models + evaluation
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }
    today_close = data["Close"].iloc[split:].values
    actual = today_close * (1 + y_test.values)

    def score(pred):
        return [mean_absolute_error(actual, pred),
                float(np.sqrt(mean_squared_error(actual, pred))),
                r2_score(actual, pred)]

    rows = {"Baseline (tomorrow = today)": score(today_close)}
    chart = pd.DataFrame({"Actual": actual}, index=data.index[split:])
    best_name, best_rmse = None, float("inf")
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = today_close * (1 + model.predict(X_test_s))
        rows[name] = score(pred)
        chart[name] = pred
        if rows[name][1] < best_rmse:
            best_name, best_rmse = name, rows[name][1]
    results = pd.DataFrame(rows, index=["MAE (₹)", "RMSE (₹)", "R²"]).T.round(3)

    # 8. Retrain best model on all data, predict next trading day
    final_scaler = StandardScaler().fit(X[chosen])
    final_model = models[best_name].fit(final_scaler.transform(X[chosen]), y)
    pred_return = float(final_model.predict(final_scaler.transform(today_features[chosen]))[0])

    return {
        "ticker": ticker,
        "last_close": float(df["Close"].iloc[-1]),
        "last_date": str(df.index[-1].date()),
        "pred_return": pred_return,
        "best_name": best_name,
        "chosen": chosen,
        "results": results,
        "chart": chart.tail(60),
        "prices": df["Close"].tail(250).rename("Close price (₹)"),
        "explore": explore,
        "quality": quality,
    }


# ---------------------------------------------------------------
# The app screen
# ---------------------------------------------------------------
st.title("📈 NSE Stock Predictor")
st.caption("Predicts the next trading day's closing price with 3 machine learning models. "
           "For learning only - not financial advice.")

with st.form("stock_form"):
    typed = st.text_input("ENTER THE STOCK:", value="TCS", max_chars=20,
                          help="NSE symbol, for example TCS, INFY, RELIANCE, SBIN, HDFCBANK")
    go = st.form_submit_button("Predict", type="primary", width="stretch")

if go:
    sym = typed.strip()
    if re.fullmatch(r"[A-Za-z0-9&\-]{1,20}(\.NS)?", sym, flags=re.IGNORECASE):
        st.session_state["symbol"] = sym
    else:
        st.session_state.pop("symbol", None)
        st.error("That doesn't look like an NSE symbol. Try something like TCS or INFY.")

if "symbol" in st.session_state:
    try:
        with st.spinner("Downloading data and training 3 models... (about 10-30 seconds the first time)"):
            r = run_pipeline(st.session_state["symbol"])
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Something went wrong while getting data or training: {e}")
        st.stop()

    pred_price = r["last_close"] * (1 + r["pred_return"])
    st.subheader(r["ticker"])

    c1, c2 = st.columns(2)
    c1.metric("Last close", f"₹{r['last_close']:,.2f}", help=f"Closing price on {r['last_date']}")
    c2.metric("Predicted next close", f"₹{pred_price:,.2f}", f"{r['pred_return'] * 100:+.2f}%")
    st.caption(f"Model used: **{r['best_name']}**  ·  Last close date: {r['last_date']}")

    st.markdown("**Closing price, last year**")
    st.line_chart(r["prices"])

    st.markdown("**How the models did on days they had never seen**")
    st.dataframe(r["results"], width="stretch")
    base_rmse = r["results"].loc["Baseline (tomorrow = today)", "RMSE (₹)"]
    best_rmse = r["results"].loc[r["best_name"], "RMSE (₹)"]
    if best_rmse < base_rmse:
        st.success(f"{r['best_name']} beat the 'tomorrow = today' baseline on this stock "
                   f"(RMSE ₹{best_rmse:.2f} vs ₹{base_rmse:.2f}).")
    else:
        st.info("No model clearly beat the simple 'tomorrow = today' guess on this stock. "
                "That is common with daily prices, so treat the prediction with extra caution.")

    st.markdown("**Actual vs predicted, last 60 test days**")
    st.line_chart(r["chart"])

    with st.expander("Data exploration"):
        st.dataframe(r["explore"], width="stretch")
    with st.expander("Data quality"):
        st.table(pd.DataFrame(r["quality"], index=["Count"]).T)
    with st.expander("Features chosen by feature selection"):
        st.write(", ".join(r["chosen"]))

    st.warning("This is an educational demo. Predictions are estimates, not financial advice.")
