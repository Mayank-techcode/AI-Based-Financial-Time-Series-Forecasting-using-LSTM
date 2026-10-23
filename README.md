# AI-Based Financial Time Series Forecasting using LSTM

Predicts the **next-day closing price** of **Reliance** stock using a **Deep Learning (LSTM – Long Short-Term Memory)** model.

Stock prices are time series data: each value depends on previous values, and patterns unfold over time. LSTM networks are built for this kind of sequential learning, so the model uses the **previous 10 days' closing prices** to predict the 11th day's close.

> For a detailed walkthrough of every concept used here (scaling, sliding window, LSTM gates, metrics), see [NOTES.md](NOTES.md).

---

## Dataset

Historical Reliance stock data with the columns `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
Only the **Close** price is used as the feature.

```
Date         Open     High     Low      Close    Volume
2024-01-01   2505.4   2520.8   2490.2   2510.5   5000000
```

---

## Workflow

```
Raw Data → Sort by Date → MinMax Scaling → Sliding Window (10 days)
        → Reshape to 3D → LSTM Model → Training → Prediction
        → Inverse Scaling → Evaluation → Next-Day Forecast
```

- **Input shape:** `(samples, 10, 1)`
- **Output:** A single value, the predicted close price

---

## Model Architecture

```python
model.add(LSTM(units=50, return_sequences=True, input_shape=(TIME_STEPS, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25, activation="relu"))
model.add(Dense(units=1))
```

## Training

| Parameter        | Value                |
|------------------|----------------------|
| Time steps       | 10                   |
| Epochs           | 60                   |
| Batch size       | 16                   |
| Validation split | 0.2                  |
| Loss             | `mean_squared_error` |

## Evaluation

Predictions are converted back to rupees with `scaler.inverse_transform` and evaluated using **MAE**, **MSE** and **RMSE**.

## Next-Day Prediction

```python
last_10_days = scaled_close[-TIME_STEPS:].reshape(1, TIME_STEPS, 1)
next_day_price = scaler.inverse_transform(model.predict(last_10_days))
```

```
Predicted Next Day Close Price: ₹2635.75
```

---

## Tech Stack

Python · TensorFlow/Keras · Pandas · NumPy · scikit-learn · matplotlib

## Getting Started

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install numpy pandas scikit-learn tensorflow matplotlib
python main.py
```

## Disclaimer

For **educational purposes only**. Predictions should not be used as financial advice or for real trading decisions.
