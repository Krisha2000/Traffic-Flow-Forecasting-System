# Traffic Flow Prediction Using Time Series Analysis


##  Abstract

Urban centers across the globe are increasingly burdened by traffic congestion due to rapid urbanization. This project addresses the problem by leveraging **time series forecasting methodologies** to model and predict urban traffic volumes using historical data.  

Both **classical statistical models** and **advanced deep learning architectures** are explored to capture temporal dynamics such as daily and weekly seasonality, enabling informed and intelligent traffic management systems.

---

##  Dataset Overview

- **Dataset:** Traffic Dataset (Four Junctions)  
- **Total Observations:** 48,120 hourly records  
- **Focus:** Junction-1  

### Features
- **DateTime:** Hourly timestamp  
- **Vehicles:** Number of vehicles recorded *(Target Variable)*  

### Preprocessing Steps
- Filtered data to isolate **Junction-1**
- Resampled to **daily frequency** for trend analysis
- Checked stationarity using the **Augmented Dickey-Fuller (ADF) test**
- Applied **first-order differencing** to achieve stationarity
- Feature Engineering:
  - Hour  
  - Day of Week  
  - Month  
  - Year  

These features help capture cyclic and seasonal patterns in traffic flow.

---

##  Technologies and Libraries

The project was implemented using **Python** with the following libraries:

- **Data Processing:** `pandas`, `numpy`, `datetime`
- **Feature Engineering:** `holidays`
- **Visualization:** `matplotlib.pyplot`
- **Statistical Modeling:** `statsmodels`  
  - ARIMA, SARIMA  
  - ADF test  
  - ACF / PACF plots
- **Machine Learning Utilities:** `scikit-learn`  
  - MinMaxScaler  
  - Train-Test Split  
  - Evaluation Metrics
- **Deep Learning:** `torch (PyTorch)`  
  - RNN  
  - LSTM  
  - GRU  

---

##  Models Implemented

The project compares **statistical time series models** with **deep learning models using quantile regression**.

---

### 1️. Statistical Models

#### ARMA (Autoregressive Moving Average)
- **Order:** (4, 4) *(selected using AIC)*

#### ARIMA (Autoregressive Integrated Moving Average)
- **Order:** (4, 1, 4)
- Captures non-seasonal trends
- Struggled with short-term fluctuations

#### SARIMA (Seasonal ARIMA)
- **Order:** (1, 1, 2) × (0, 1, 1, 7)
- Explicitly models **weekly seasonality (period = 7)**

---

### 2️. Deep Learning Models (Quantile Regression)

Deep learning models were trained using **Quantile Loss (Pinball Loss)** to predict conditional quantiles:

- **0.05** (lower bound)  
- **0.50** (median)  
- **0.95** (upper bound)  

This enables the generation of **prediction intervals**.

#### Implemented Models
- **RNN (Recurrent Neural Network):** Basic recurrent architecture  
- **LSTM (Long Short-Term Memory):** Captures long-term dependencies and mitigates vanishing gradients  
- **GRU (Gated Recurrent Unit):** Computationally efficient with strong performance  

---

##  Evaluation Metrics

Model performance was evaluated using:

- **RMSE (Root Mean Squared Error):** Measures prediction error magnitude  
- **MAPE (Mean Absolute Percentage Error):** Measures average relative error  
- **PICP (Prediction Interval Coverage Probability):** Percentage of true values within predicted intervals  
- **MPIW (Mean Prediction Interval Width):** Measures sharpness of prediction intervals  

---

##  Results

| Model   | RMSE | MAPE (%) | PICP (%) | MPIW |
|--------|------|----------|----------|------|
| ARMA   | 18.76 | 24.51 | 10.50 | 10.88 |
| ARIMA  | 22.49 | 30.39 | 1.66  | 10.87 |
| SARIMA | 7.93  | 9.67  | 36.46 | 8.28  |
| RNN    | 3.92  | 11.38 | 84.00 | 11.03 |
| LSTM   | 3.58  | 11.49 | 88.00 | 10.52 |
| GRU    | **3.57** | 11.82 | **90.00** | **10.24** |

---

##  Key Findings

- **Statistical vs Deep Learning:**  
  Deep learning models significantly outperform statistical models in both **point prediction accuracy (RMSE)** and **uncertainty estimation (PICP)**.

- **Best Model:**  
  The **GRU** model achieved:
  - Lowest RMSE: **3.57**
  - Highest PICP: **90.00%**
  - Narrowest prediction interval: **MPIW = 10.24**

- **Seasonality Handling:**  
  - **SARIMA** was the best-performing statistical model due to its ability to model weekly seasonality.
  - **ARIMA** failed to capture seasonal patterns effectively.

---

##  Conclusion

This project demonstrates that while traditional statistical methods like **SARIMA** can model seasonality in traffic data, **deep learning architectures**—particularly **GRU and LSTM**—offer superior predictive performance for complex urban traffic patterns.  

The use of **quantile regression** further enhances these models by providing reliable **prediction intervals**, making them highly suitable for **risk-aware and real-world traffic management systems**.

---

##  Future Scope

- Incorporating **exogenous variables** (weather, holidays, events)
- Extending models to **multi-junction forecasting**
- Real-time deployment with **streaming data pipelines**
- Exploring **attention-based architectures** and **transformers**

---
