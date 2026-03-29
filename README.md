## **Hybrid Deep Learning for Stock Market Prediction: A Comparative Study**

This repository contains the full implementation, experiments, and analysis for the dissertation **“Hybrid Deep Learning for Stock Market Prediction: A Comparative Study of Modern Forecasting Architectures.”** It evaluates how standalone **LSTM** and hybrid **CNN‑LSTM** models perform in forecasting S&P 500 log‑prices and predicting **bull/bear market regimes** using walk‑forward expanding window validation.

---

## **📂 Project Structure**
- `data/` — S&P 500 OHLCV data (2006–2026), engineered features, SMA‑200 regime labels  
- `preprocessing/` — Cleaning, scaling, log‑transformations, sliding‑window sequence generation  
- `features/` — Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, volatility, volume‑based indicators)  
- `models/` — ARIMA baseline, LSTM architecture, CNN‑LSTM hybrid architecture  
- `training/` — Scripts for regression and classification training  
- `walk_forward/` — Custom expanding‑window validation implementation  
- `results/` — Regression metrics, classification metrics, confusion matrices, ROC curves, bar charts  
- `notebooks/` — Full experimental workflow and analysis  
- `figures/` — Dissertation‑ready plots and diagrams  

---

## **🔍 Project Overview**
This study investigates two predictive tasks:

### **1. Regression (Exploratory Phase)**  
Forecasting log‑transformed S&P 500 closing prices using:  
- **ARIMA** (statistical baseline)  
- **LSTM**  
- **CNN‑LSTM hybrid**

### **2. Classification (Core Phase)**  
Predicting **bull vs. bear regimes** using SMA‑200 directional labeling.  
Models evaluated:  
- **LSTM**  
- **CNN‑LSTM**

The hybrid model is tested to determine whether CNN‑based local feature extraction improves temporal modeling and directional accuracy.

---

## **📊 Evaluation Metrics**
### **Regression**
- RMSE  
- MAE  
- MAPE  

### **Classification**
- Accuracy  
- Precision  
- Recall  
- Specificity  
- F1‑Score  
- MCC  
- ROC‑AUC  

Walk‑forward expanding window validation is used to simulate realistic market conditions and avoid data leakage.

---

## **🎯 Key Objectives**
- Compare standalone LSTM vs. hybrid CNN‑LSTM  
- Assess numerical vs. directional prediction performance  
- Evaluate model stability across multiple walk‑forward folds  
- Determine whether hybrid architectures justify their computational cost  

---

## **📈 Outputs**
- Predicted vs. actual price plots  
- Confusion matrices  
- ROC curves  
- Bar charts comparing all metrics  
- Walk‑forward fold performance summaries  
- Full discussion of findings and implications  

---

## **📝 Citation**
If referencing this work, please cite the dissertation:

**Mohamed Mohieldin Mohamed Elkemany (2026).  
Hybrid Deep Learning for Stock Market Prediction: A Comparative Study of Modern Forecasting Architectures.  
University of East London.**
