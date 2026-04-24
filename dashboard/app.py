# ============================================================
# DIAMOND PRICE ANALYSIS — STREAMLIT DASHBOARD
# CET242 – Data Analytics and Visualization
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Diamond Price Analysis",
    page_icon="💎",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

import os
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'diamonds.csv')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.drop_duplicates()
    df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
    df = df[(df['y'] < 20) & (df['z'] < 20)].copy()

    cut_order     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_order   = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    df['cut_encoded']     = df['cut'].map({v: i+1 for i, v in enumerate(cut_order)})
    df['color_encoded']   = df['color'].map({v: i+1 for i, v in enumerate(color_order)})
    df['clarity_encoded'] = df['clarity'].map({v: i+1 for i, v in enumerate(clarity_order)})

    return df

df = load_data()

# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(_df):
    # Clustering
    scaler = StandardScaler()
    X_clust = scaler.fit_transform(_df[['carat', 'price']])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_clust)

    # Regression
    features = ['carat', 'depth', 'table', 'x', 'y', 'z',
                'cut_encoded', 'color_encoded', 'clarity_encoded']
    X = _df[features].values
    y = np.log(_df['price'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=1.0))
    ])
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))

    return clusters, model, r2

clusters, reg_model, r2 = train_models(df)
df['cluster'] = clusters
cluster_names = {0: 'Budget', 1: 'Mid-Range', 2: 'Upper Mid-Range', 3: 'Luxury'}
df['cluster_label'] = df['cluster'].map(cluster_names)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💎 Diamond Analysis")
page = st.sidebar.radio("Navigate", ["📊 Overview", "💎 Market Segments", "🔮 Price Predictor"])

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "📊 Overview":
    st.title("💎 Diamond Market Price Analysis")
    st.markdown("**CET242 – Data Analytics and Visualization | Spring 2026**")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Diamonds", f"{len(df):,}")
    col2.metric("Avg Price", f"${df['price'].mean():,.0f}")
    col3.metric("Avg Price/Carat", f"${df['price'].sum()/df['carat'].sum():,.0f}")
    col4.metric("Price Range", f"$326 – $18,823")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='price', nbins=50,
                           title='Price Distribution',
                           color_discrete_sequence=['steelblue'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df.sample(5000), x='carat', y='price',
                         title='Price vs Carat',
                         opacity=0.3,
                         color_discrete_sequence=['coral'])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("KPI Summary")
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    kpi_df = df.groupby('cut')['price'].agg(['mean','min','max']).loc[cut_order].round(2)
    kpi_df.columns = ['Avg Price', 'Min Price', 'Max Price']
    st.dataframe(kpi_df, use_container_width=True)

# ============================================================
# PAGE 2 — CLUSTERING
# ============================================================

elif page == "💎 Market Segments":
    st.title("💎 Diamond Market Segments")
    st.markdown("K-Means Clustering — 4 natural market segments based on carat and price")
    st.divider()

    fig = px.scatter(df.sample(10000), x='carat', y='price',
                     color='cluster_label',
                     title='Diamond Market Segments (K-Means)',
                     opacity=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summary")
    summary = df.groupby('cluster_label')[['carat', 'price']].mean().round(2)
    summary.columns = ['Avg Carat', 'Avg Price ($)']
    st.dataframe(summary, use_container_width=True)

# ============================================================
# PAGE 3 — PRICE PREDICTOR
# ============================================================

elif page == "🔮 Price Predictor":
    st.title("🔮 Diamond Price Predictor")
    st.markdown(f"**Model:** Polynomial Regression + Ridge | **R² = {r2:.4f}**")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        carat   = st.slider("Carat", 0.2, 5.0, 1.0, 0.01)
        depth   = st.slider("Depth %", 43.0, 79.0, 61.7, 0.1)
        table   = st.slider("Table %", 43.0, 95.0, 57.0, 0.1)

    with col2:
        cut     = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
        color   = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
        clarity = st.selectbox("Clarity", ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])

    with col3:
        x = st.slider("X (mm)", 0.0, 11.0, 6.4, 0.01)
        y = st.slider("Y (mm)", 0.0, 11.0, 6.4, 0.01)
        z = st.slider("Z (mm)", 0.0, 7.0, 4.0, 0.01)

    cut_enc     = ['Fair','Good','Very Good','Premium','Ideal'].index(cut) + 1
    color_enc   = ['J','I','H','G','F','E','D'].index(color) + 1
    clarity_enc = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'].index(clarity) + 1

    features = np.array([[carat, depth, table, x, y, z, cut_enc, color_enc, clarity_enc]])
    pred_log = reg_model.predict(features)[0]
    pred_price = np.exp(pred_log)

    st.divider()
    st.metric("💰 Predicted Price", f"${pred_price:,.0f}")