
 📊 Inflation Forecasting using Big Data

1.Overview
This project predicts national inflation (CPI) using alternative data sources such as:
- 🛒 E-commerce product prices
- 💬 Social media sentiment
- 💼 Wage signals (job data)

Traditional CPI data is delayed, but this model provides "faster and more accurate predictions".



2. Features
- Real-time price index generation
- Sentiment analysis integration
- Wage growth indicators
- Machine Learning model (Gradient Boosting)
- Backtesting & evaluation
- Interactive dashboard (Streamlit)


3. Project Structure
 ├── final_data.csv # Input dataset
 ├── main model.py # Model training & evaluation
 ├── dashboard.py # Streamlit dashboard
 ├── output.csv # Predictions
 ├── requirements.txt # Dependencies
 └── README.md # Project documentation

5. Run the Project
Run Model

python main model.py

Run Dashboard

streamlit run dashboard.py

6.Model Details

Feature Engineering:
Price Change
Salary Growth
Sentiment Shift

Model Used:
Gradient Boosting Regressor

7. Evaluation Metrics
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)

8. Output
CPI Forecast
Backtesting comparison (Actual vs Predicted)

