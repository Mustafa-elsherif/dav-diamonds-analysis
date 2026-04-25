# 💎 Diamond Market Price Analysis

### Data Analytics & Visualization Project

---

## 📌 Project Overview

This project analyzes diamond pricing in the retail market using a real-world dataset of **53,940 diamonds**.

The goal is to understand **what truly drives diamond prices**, uncover **hidden patterns**, and identify **mispricing opportunities** using data analytics and machine learning.

---

## 🌐 Live Dashboard

👉 https://dav-diamonds.streamlit.app/

An interactive dashboard built with **Streamlit** allows users to explore the dataset dynamically, filter diamonds, and visualize insights in real time.

---

## 🎯 Problem Statement

Diamond pricing lacks transparency in the retail market. Buyers and small retailers often cannot determine whether a diamond is fairly priced based on its characteristics, leading to overpayment or missed value opportunities.

---

## ❓ Key Questions

* What features most influence diamond price?
* Does higher quality (cut, color, clarity) always mean higher price?
* Are there overpriced or underpriced diamonds?
* Can we segment the market into meaningful groups?

---

## 📊 Dataset Information

* **Source:** Kaggle / ggplot2
* **Size:** 53,940 rows × 10 features

### Features

| Feature | Description                    |
| ------- | ------------------------------ |
| carat   | Weight of the diamond          |
| cut     | Quality of cut (Fair → Ideal)  |
| color   | Color grade (D → J)            |
| clarity | Clarity level (IF → I1)        |
| depth   | Total depth percentage         |
| table   | Top width percentage           |
| price   | Price in USD (Target Variable) |
| x, y, z | Diamond dimensions in mm       |

---

## 🧹 Data Cleaning

* Removed **146 duplicate rows**
* Removed **19 invalid records** (zero dimensions)
* Removed **3 extreme outliers**
* Final dataset: **53,772 rows**

---

## 📈 Key Performance Indicators (KPIs)

1. **Average Price per Carat**
2. **Price Range by Cut Quality**
3. **Price Range by Color**
4. **Price Range by Clarity**
5. **Anomaly Rate (Outliers)**

---

## 🔍 Key Insights

* **Carat is the strongest price driver** (correlation = 0.92)
* Price increases **non-linearly** with size
* Better quality (cut/color/clarity) does **NOT always mean higher price**
* Around **6.54% of diamonds are anomalously priced**

---

## ⚠️ Core Insight

### Carat Size Confounding Effect

High-quality diamonds tend to be **smaller**, while lower-quality diamonds are often **larger**.

Since **carat strongly drives price**, larger low-quality diamonds can appear more expensive than smaller high-quality ones.

---

## 📊 Market Segmentation (Clustering)

Using **K-Means Clustering**:

| Segment         | Avg Carat | Avg Price |
| --------------- | --------- | --------- |
| Budget          | 0.40      | $1,055    |
| Mid-Range       | 0.89      | $3,697    |
| Upper Mid-Range | 1.27      | $7,803    |
| Luxury          | 1.87      | $14,387   |

* Optimal clusters: **4**
* Silhouette Score: **0.5631**

---

## 🤖 Machine Learning Model

### Polynomial Ridge Regression

* Predicts diamond price from physical and quality features
* Uses polynomial features (degree = 2)
* Applies Ridge regularization to prevent overfitting

### Performance

* **R² Score:** 0.9858
* **RMSE:** 0.1203

✅ Model explains **98.58% of price variation**

---

## 📊 Visualizations

The project includes:

* Price distribution
* Price vs carat (scatter plot)
* Price by cut, color, and clarity
* Correlation heatmap
* Clustering visualization
* Regression evaluation plots

---

## 🖥️ Dashboard Features

* Interactive filtering (carat, cut, color, clarity)
* Real-time visualizations
* Market segmentation view
* Price analysis tools

---

## 🛠️ Tech Stack

* **Python** (Pandas, NumPy)
* **Visualization** (Matplotlib, Seaborn)
* **Machine Learning** (Scikit-learn)
* **Dashboard** (Streamlit)

---

## 📁 Project Structure

```
DAV-DIAMONDS-ANALYSIS/
│
├── dashboard/        # Streamlit application
├── data/             # Dataset files
├── notebooks/        # Analysis notebooks
├── requirements.txt  # Project dependencies
└── README.md         # Documentation
```

---

## 🚀 Conclusion

* Diamond pricing is **dominated by carat (size)**
* Quality features are important but often **masked by size**
* The market contains **detectable inefficiencies**
* Machine learning can **accurately predict diamond prices**

---

## 👤 Author

**Mustafa Nabil**
CET242 – Data Analytics & Visualization
Spring 2026

---
