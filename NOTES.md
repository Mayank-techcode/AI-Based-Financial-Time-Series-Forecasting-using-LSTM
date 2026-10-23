# Project Notes: Concepts Behind the LSTM Stock Forecaster

Detailed explanations of the concepts used in this project. Back to [README](README.md).

---

## 1. Why LSTM?

Stock market data is time series data:

- Each value depends on previous values
- Patterns exist over time
- Sequential learning is required

Traditional ML models struggle here. LSTM is a special type of RNN designed to remember long-term dependencies and avoid the vanishing gradient problem.

---

## 2. Dataset Columns

| Column | Description |
|--------|-------------|
| Date   | Trading date |
| Open   | Price at market opening |
| High   | Highest price of the day |
| Low    | Lowest price of the day |
| Close  | Final price of the day |
| Volume | Total shares traded |

In this project we select only the **Close** price, since it represents the final agreed value of the stock for that trading day.

```python
close_data = data[["Close"]].values
```

```
Input  : Previous closing prices
Output : Future closing price
```

---

## 3. Data Preprocessing

**Sorting** ensures chronological order:

```python
data = data.sort_values("Date")
```

**Scaling** uses `MinMaxScaler(feature_range=(0, 1))`:

```
Scaled = (Value - Min) / (Max - Min)

Example: Min = 2500, Max = 2700, Value = 2600
Scaled = (2600 - 2500) / 200 = 0.5
```

---

## 4. Sliding Window Sequence Generation

An LSTM learns from a sequence of previous values, not a single value.

```python
TIME_STEPS = 10
```

The previous 10 days' closing prices are used to predict the 11th day's close.

```
Input X : Day 1 ... Day 10  → Output y : Day 11
Input X : Day 2 ... Day 11  → Output y : Day 12
```

This is called the **sliding window technique**.

---

## 5. LSTM Input Shape

LSTM expects 3D input: `(samples, time_steps, features)`

```python
X = X.reshape(X.shape[0], X.shape[1], 1)
```

- `samples`: total number of training examples
- `time_steps`: 10 previous days
- `features`: 1, because only the Close price is used

Example: `X shape = (100, 10, 1)` means 100 samples, each with 10 days, each day with 1 value.

---

## 6. LSTM Gates

LSTM remembers important past information and forgets unnecessary information using three main gates:

| Gate | Role | Example |
|------|------|---------|
| Forget Gate | Removes useless memory | Old market trend is no longer useful, so forget it |
| Input Gate  | Adds new useful memory | Today's price movement is important, so store it |
| Output Gate | Produces the prediction | Use important memory to predict tomorrow's close price |

This is why LSTM is better than a simple RNN for time series data.

---

## 7. Layer-wise Explanation

| Layer | Purpose |
|-------|---------|
| LSTM Layer 1 | Learns sequence patterns from the previous 10 days; `return_sequences=True` passes the full output sequence to the next LSTM layer |
| Dropout (0.2) | Randomly ignores 20% of neurons during training to prevent overfitting |
| LSTM Layer 2 | Learns deeper time-based patterns; returns the final output only |
| Dense (25, ReLU) | Fully connected layer that learns complex relationships |
| Dense (1) | Outputs a single value: the predicted close price |

---

## 8. Training and Loss

```python
model.fit(
    X_train,
    y_train,
    epochs=60,
    batch_size=16,
    validation_split=0.2
)
```

- **Epoch:** one complete pass over the training data. `epochs=60` means the model can learn from the data up to 60 times.
- **Batch size:** `batch_size=16` means the model processes 16 samples at a time.
- **Validation split:** 20% of the training data is held out to check performance during training.
- **Loss:** the error, i.e. the difference between actual and predicted values. This project uses `loss="mean_squared_error"`. Lower loss means a better model.

If loss decreases over epochs, the model is learning properly:

```
Epoch 1   Loss = 0.050
Epoch 10  Loss = 0.020
Epoch 30  Loss = 0.006
```

---

## 9. Predictions and Inverse Scaling

The model outputs scaled values (e.g. `0.65`), which are not real prices. Inverse scaling converts them back into rupees.

```python
predicted_scaled = model.predict(X_test)

predicted_prices = scaler.inverse_transform(predicted_scaled)
actual_prices    = scaler.inverse_transform(y_test.reshape(-1, 1))
```

Manual formula:

```
Original Value = Scaled Value × (Max - Min) + Min

Example: Min = 2500, Max = 2700, Scaled Prediction = 0.75
Original Price = 0.75 × 200 + 2500 = ₹2650
```

---

## 10. Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** | Average of absolute errors | Same unit as price. `MAE = 25` means the average error is around ₹25. |
| **MSE** | Average of squared errors | Penalizes large errors more. Error 20 → squared error 400. |
| **RMSE** | Square root of MSE | Same unit as price. `√400 = 20`. |

```python
mse  = mean_squared_error(actual_prices, predicted_prices)
mae  = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
```

---

## 11. Next-Day Prediction

```python
last_10_days = scaled_close[-TIME_STEPS:]
last_10_days = last_10_days.reshape(1, TIME_STEPS, 1)

next_day_scaled = model.predict(last_10_days)
next_day_price  = scaler.inverse_transform(next_day_scaled)
```

1. Take the last 10 days' closing prices
2. Give them to the LSTM model
3. Model predicts the next day's close price
4. Convert the prediction back to the original price

```
Predicted Next Day Close Price: ₹2635.75
```
