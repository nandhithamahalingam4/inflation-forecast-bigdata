import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Load data
df = pd.read_csv("final_data.csv")

# Feature engineering
df['price_change'] = df['price_index'].pct_change()
df['salary_growth'] = df['avg_salary'].pct_change()
df['sentiment_shift'] = df['sentiment'].diff()
df.fillna(0, inplace=True)

# Features
features = ['price_index','sentiment','avg_salary',
            'price_change','salary_growth','sentiment_shift']

X = df[features]
y = df['cpi']

# Train model
model = GradientBoostingRegressor()
model.fit(X, y)

# -------------------------------
# UI
# -------------------------------
st.title("📊 Inflation Forecast Dashboard")

# Charts
st.subheader("CPI Trend")
st.line_chart(df[['cpi']])

st.subheader("Price Index Trend")
st.line_chart(df[['price_index']])

# User Inputs
st.subheader("🔮 Predict Inflation")

price = st.slider("Price Index", 90, 150, 110)
sentiment = st.slider("Sentiment", -1.0, 1.0, -0.5)
salary = st.slider("Avg Salary", 20000, 40000, 28000)

# Derived features
price_change = 0.05
salary_growth = 0.04
sentiment_shift = -0.1

# Prediction
prediction = model.predict([[price, sentiment, salary,
                             price_change, salary_growth, sentiment_shift]])

st.success(f"Predicted CPI: {prediction[0]:.2f}")

# Backtesting view
st.subheader("📈 Backtesting")
df['predicted_cpi'] = model.predict(X)
st.line_chart(df[['cpi','predicted_cpi']])