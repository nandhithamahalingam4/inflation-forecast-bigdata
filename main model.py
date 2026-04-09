# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("final_data.csv")

# -------------------------------
# ✅ STEP 1: FEATURE ENGINEERING
# -------------------------------
df['price_change'] = df['price_index'].pct_change()
df['salary_growth'] = df['avg_salary'].pct_change()
df['sentiment_shift'] = df['sentiment'].diff()

# Fill missing values
df.fillna(0, inplace=True)

# -------------------------------
# ✅ STEP 2: MODEL
# -------------------------------
features = ['price_index','sentiment','avg_salary',
            'price_change','salary_growth','sentiment_shift']

X = df[features]
y = df['cpi']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Advanced model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# -------------------------------
# ✅ STEP 3: EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# -------------------------------
# ✅ STEP 4: BACKTESTING
# -------------------------------
df['predicted_cpi'] = model.predict(X)

print("\nBacktesting Results:")
print(df[['date','cpi','predicted_cpi']])

# Save model output
df.to_csv("output.csv", index=False)

# -------------------------------
# 🔮 FUTURE PREDICTION
# -------------------------------
future_input = [[112, -0.7, 28000, 0.05, 0.04, -0.1]]
future_pred = model.predict(future_input)

print("\nFuture CPI Prediction:", future_pred[0])