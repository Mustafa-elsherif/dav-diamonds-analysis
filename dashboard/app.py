# ============================================================
# DIAMOND PRICE ANALYSIS — STREAMLIT DASHBOARD
# CET242 – Data Analytics and Visualization | Spring 2026
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

st.markdown("""
<style>
    * { font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] { background-color: #0f0f14; }
    [data-testid="stSidebar"] { background-color: #16161f; border-right: 1px solid #2a2a3a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; font-size: 0.9rem; }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div { color: #a78bfa !important; font-weight: 600; }
    h1, h2, h3, p, div { color: #e2e8f0; }
    .stDivider { border-color: #2a2a3a !important; }

    .kpi-card {
        background: #1a1a27;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 18px 20px;
        border-top: 3px solid #a78bfa;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 6px;
    }
    .chart-caption {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
        padding: 0 4px;
    }
    .page-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 2px;
    }
    .page-sub {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def train_models(_df):
    scaler_clust = StandardScaler()
    X_clust = scaler_clust.fit_transform(_df[['carat', 'price']])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_clust)
    temp_df = _df.copy()
    temp_df['cluster'] = clusters
    cluster_avg = temp_df.groupby('cluster')['price'].mean().sort_values()
    cluster_names = {
        cluster_avg.index[0]: 'Budget',
        cluster_avg.index[1]: 'Mid-Range',
        cluster_avg.index[2]: 'Upper Mid-Range',
        cluster_avg.index[3]: 'Luxury'
    }
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
    return clusters, cluster_names, model, r2

clusters, cluster_names, reg_model, r2 = train_models(df)
df['cluster'] = clusters
df['cluster_label'] = df['cluster'].map(cluster_names)

cut_order     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order   = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_order = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']

DARK_BG    = '#0f0f14'
CARD_BG    = '#1a1a27'
BORDER     = '#2a2a3a'
TEXT       = '#e2e8f0'
MUTED      = '#64748b'
ACCENT     = '#a78bfa'
ACCENT2    = '#818cf8'
COLORS     = ['#a78bfa', '#818cf8', '#60a5fa', '#34d399', '#f472b6']

def dark_chart(fig):
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family='Inter, sans-serif', size=11, color=TEXT),
        title_font=dict(size=12, color='#f1f5f9'),
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor=BORDER,
            font=dict(color=TEXT)
        )
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED)
    )
    fig.update_yaxes(
        gridcolor='#1f1f2e',
        linecolor=BORDER,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED)
    )
    return fig

def kpi(label, value):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## Diamond Analysis")
st.sidebar.markdown('<p style="color:#64748b;font-size:0.8rem;">CET242 · Spring 2026</p>', unsafe_allow_html=True)
st.sidebar.divider()
page = st.sidebar.radio("", ["Overview", "Price Analysis", "Market Segments", "Price Predictor"])
st.sidebar.divider()
st.sidebar.markdown('<p style="color:#64748b;font-size:0.78rem;">53,772 diamonds · 10 features</p>', unsafe_allow_html=True)

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown('<p class="page-title">Diamond Market Price Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Diamonds Dataset · 53,772 records · 10 features</p>', unsafe_allow_html=True)
    st.divider()

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, label, value in zip(
        [col1, col2, col3, col4, col5],
        ["Total Diamonds", "Average Price", "Price per Carat", "Min Price", "Max Price"],
        [f"{len(df):,}", f"${df['price'].mean():,.0f}", f"${df['price'].sum()/df['carat'].sum():,.0f}",
         f"${df['price'].min():,}", f"${df['price'].max():,}"]
    ):
        col.markdown(kpi(label, value), unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='price', nbins=50, title='Price Distribution')
        fig.update_traces(marker_color=ACCENT, marker_line_width=0)
        fig.update_layout(xaxis_title='Price ($)', yaxis_title='Count')
        st.plotly_chart(dark_chart(fig), use_container_width=True)
        st.markdown('<p class="chart-caption">Most diamonds priced under $2,500. Right-skewed — few very expensive diamonds pull the average up.</p>', unsafe_allow_html=True)

    with col2:
        fig = px.scatter(df.sample(5000, random_state=42), x='carat', y='price',
                         title='Price vs Carat Weight')
        fig.update_traces(marker=dict(color=ACCENT, size=3, opacity=0.4))
        fig.update_layout(xaxis_title='Carat', yaxis_title='Price ($)')
        st.plotly_chart(dark_chart(fig), use_container_width=True)
        st.markdown('<p class="chart-caption">Strong positive relationship (correlation = 0.92). Price grows exponentially with carat.</p>', unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        cut_counts = df['cut'].value_counts().loc[cut_order].reset_index()
        fig = px.bar(cut_counts, x='cut', y='count', title='Count by Cut')
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(xaxis_title='', yaxis_title='Count')
        st.plotly_chart(dark_chart(fig), use_container_width=True)

    with col2:
        color_counts = df['color'].value_counts().loc[color_order].reset_index()
        fig = px.bar(color_counts, x='color', y='count', title='Count by Color')
        fig.update_traces(marker_color=ACCENT2)
        fig.update_layout(xaxis_title='', yaxis_title='Count')
        st.plotly_chart(dark_chart(fig), use_container_width=True)

    with col3:
        clarity_counts = df['clarity'].value_counts().loc[clarity_order].reset_index()
        fig = px.bar(clarity_counts, x='clarity', y='count', title='Count by Clarity')
        fig.update_traces(marker_color='#60a5fa')
        fig.update_layout(xaxis_title='', yaxis_title='Count')
        st.plotly_chart(dark_chart(fig), use_container_width=True)

# ============================================================
# PAGE 2 — PRICE ANALYSIS
# ============================================================

elif page == "Price Analysis":
    st.markdown('<p class="page-title">Price Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">How each feature affects diamond price</p>', unsafe_allow_html=True)
    st.divider()

    numeric_cols = df[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']].corr()
    price_corr = numeric_cols['price'].drop('price').sort_values(ascending=True)
    fig = px.bar(x=price_corr.values, y=price_corr.index,
                 orientation='h', title='Feature Correlation with Price')
    fig.update_traces(marker_color=[ACCENT if v > 0 else '#f472b6' for v in price_corr.values])
    fig.update_layout(xaxis_title='Correlation Coefficient', yaxis_title='')
    st.plotly_chart(dark_chart(fig), use_container_width=True)
    st.markdown('<p class="chart-caption">Carat (0.92) is the dominant price driver. Physical dimensions x, y, z are also strong because they correlate directly with carat. Depth and table have negligible effect.</p>', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        avg = df.groupby('cut').agg(avg_price=('price','mean'), avg_carat=('carat','mean')).loc[cut_order].reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=avg['cut'], y=avg['avg_price'], name='Avg Price ($)', marker_color=ACCENT))
        fig.add_trace(go.Scatter(x=avg['cut'], y=avg['avg_carat']*5000, name='Avg Carat ×5000',
                                 mode='lines+markers', line=dict(color='#f472b6', width=2), marker=dict(size=7)))
        fig.update_layout(title='Cut — Avg Price vs Avg Carat', xaxis_title='Cut',
                         yaxis_title='Avg Price ($)', legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(dark_chart(fig), use_container_width=True)
        st.markdown('<p class="chart-caption">Ideal cut has lower avg price ($3,462) than Premium ($4,578) because Ideal diamonds are smaller (0.703 ct vs 1.044 ct for Fair). Carat confounds the quality premium.</p>', unsafe_allow_html=True)

    with col2:
        avg = df.groupby('color').agg(avg_price=('price','mean'), avg_carat=('carat','mean')).loc[color_order].reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=avg['color'], y=avg['avg_price'], name='Avg Price ($)', marker_color=ACCENT2))
        fig.add_trace(go.Scatter(x=avg['color'], y=avg['avg_carat']*5000, name='Avg Carat ×5000',
                                 mode='lines+markers', line=dict(color='#f472b6', width=2), marker=dict(size=7)))
        fig.update_layout(title='Color — Avg Price vs Avg Carat', xaxis_title='Color (D=best)',
                         yaxis_title='Avg Price ($)', legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(dark_chart(fig), use_container_width=True)
        st.markdown('<p class="chart-caption">D color (best) avg price = $3,172 vs J (worst) = $5,326. J color diamonds are 77% larger — carat drives the difference, not color quality.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        avg = df.groupby('clarity').agg(avg_price=('price','mean'), avg_carat=('carat','mean')).loc[clarity_order].reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=avg['clarity'], y=avg['avg_price'], name='Avg Price ($)', marker_color='#60a5fa'))
        fig.add_trace(go.Scatter(x=avg['clarity'], y=avg['avg_carat']*5000, name='Avg Carat ×5000',
                                 mode='lines+markers', line=dict(color='#f472b6', width=2), marker=dict(size=7)))
        fig.update_layout(title='Clarity — Avg Price vs Avg Carat', xaxis_title='Clarity (IF=best)',
                         yaxis_title='Avg Price ($)', legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(dark_chart(fig), use_container_width=True)
        st.markdown('<p class="chart-caption">IF clarity (best) has the lowest avg price ($2,870). I1 diamonds are 154% larger on average. VS1-VS2 offers the best quality-to-price balance.</p>', unsafe_allow_html=True)

    with col2:
        root_data = pd.DataFrame({
            'Grade': ['Cut', 'Color', 'Clarity'],
            'Best Grade': [0.703, 0.658, 0.506],
            'Worst Grade': [1.044, 1.163, 1.284],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Best Grade', x=root_data['Grade'],
                             y=root_data['Best Grade'], marker_color=ACCENT))
        fig.add_trace(go.Bar(name='Worst Grade', x=root_data['Grade'],
                             y=root_data['Worst Grade'], marker_color='#f472b6'))
        fig.update_layout(title='Root Cause — Avg Carat by Quality Extreme',
                         barmode='group', xaxis_title='', yaxis_title='Average Carat')
        st.plotly_chart(dark_chart(fig), use_container_width=True)
        st.markdown('<p class="chart-caption">Worst quality grades are consistently larger in carat. This is the root cause of the counterintuitive pricing pattern across all quality dimensions.</p>', unsafe_allow_html=True)

# ============================================================
# PAGE 3 — MARKET SEGMENTS
# ============================================================

elif page == "Market Segments":
    st.markdown('<p class="page-title">Market Segments</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">K-Means Clustering · 4 natural segments based on carat and price</p>', unsafe_allow_html=True)
    st.divider()

    fig = px.scatter(df.sample(10000, random_state=42),
                     x='carat', y='price', color='cluster_label',
                     title='Diamond Market Segments — K-Means (k=4)',
                     category_orders={'cluster_label': ['Budget', 'Mid-Range', 'Upper Mid-Range', 'Luxury']},
                     color_discrete_sequence=COLORS, opacity=0.5)
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(xaxis_title='Carat', yaxis_title='Price ($)', legend_title='Segment')
    st.plotly_chart(dark_chart(fig), use_container_width=True)
    st.markdown('<p class="chart-caption">Segments are naturally separated along the carat axis — confirming carat as the dominant factor that divides the diamond market into price tiers.</p>', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        summary = df.groupby('cluster_label').agg(
            Count=('price', 'count'),
            Avg_Carat=('carat', 'mean'),
            Avg_Price=('price', 'mean'),
            Min_Price=('price', 'min'),
            Max_Price=('price', 'max')
        ).round(2)
        summary = summary.loc[['Budget', 'Mid-Range', 'Upper Mid-Range', 'Luxury']]
        summary.columns = ['Count', 'Avg Carat', 'Avg Price ($)', 'Min ($)', 'Max ($)']
        st.dataframe(summary, use_container_width=True)

    with col2:
        seg_counts = df['cluster_label'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig = px.pie(seg_counts, values='Count', names='Segment',
                     title='Segment Size Distribution',
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='inside', textinfo='percent+label',
                         textfont=dict(color='white'))
        st.plotly_chart(dark_chart(fig), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        cut_seg = df.groupby(['cluster_label', 'cut']).size().reset_index(name='count')
        fig = px.bar(cut_seg, x='cluster_label', y='count', color='cut',
                     title='Cut Distribution per Segment', barmode='stack',
                     category_orders={
                         'cluster_label': ['Budget', 'Mid-Range', 'Upper Mid-Range', 'Luxury'],
                         'cut': cut_order
                     })
        fig.update_layout(xaxis_title='Segment', yaxis_title='Count', legend_title='Cut')
        st.plotly_chart(dark_chart(fig), use_container_width=True)

    with col2:
        clarity_seg = df.groupby(['cluster_label', 'clarity']).size().reset_index(name='count')
        fig = px.bar(clarity_seg, x='cluster_label', y='count', color='clarity',
                     title='Clarity Distribution per Segment', barmode='stack',
                     category_orders={
                         'cluster_label': ['Budget', 'Mid-Range', 'Upper Mid-Range', 'Luxury'],
                         'clarity': clarity_order
                     })
        fig.update_layout(xaxis_title='Segment', yaxis_title='Count', legend_title='Clarity')
        st.plotly_chart(dark_chart(fig), use_container_width=True)

# ============================================================
# PAGE 4 — PRICE PREDICTOR
# ============================================================

elif page == "Price Predictor":
    st.markdown('<p class="page-title">Price Predictor</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-sub">Polynomial Regression + Ridge · R² = {r2:.4f} · {r2*100:.2f}% of price variance explained</p>', unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Physical Features**")
        carat = st.slider("Carat", 0.2, 5.0, 1.0, 0.01)
        depth = st.slider("Depth %", 43.0, 79.0, 61.7, 0.1)
        table = st.slider("Table %", 43.0, 95.0, 57.0, 0.1)

    with col2:
        st.markdown("**Quality Grades**")
        cut     = st.selectbox("Cut",     cut_order, index=4)
        color   = st.selectbox("Color",   color_order, index=3)
        clarity = st.selectbox("Clarity", clarity_order, index=3)

    with col3:
        st.markdown("**Dimensions (mm)**")
        x = st.slider("X — Length", 0.0, 11.0, 6.4, 0.01)
        y = st.slider("Y — Width",  0.0, 11.0, 6.4, 0.01)
        z = st.slider("Z — Depth",  0.0, 7.0,  4.0, 0.01)

    cut_enc     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'].index(cut) + 1
    color_enc   = ['J', 'I', 'H', 'G', 'F', 'E', 'D'].index(color) + 1
    clarity_enc = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'].index(clarity) + 1

    features = np.array([[carat, depth, table, x, y, z, cut_enc, color_enc, clarity_enc]])
    pred_price = np.exp(reg_model.predict(features)[0])

    if pred_price < 2000:
        segment = "Budget"
    elif pred_price < 6000:
        segment = "Mid-Range"
    elif pred_price < 11000:
        segment = "Upper Mid-Range"
    else:
        segment = "Luxury"

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    for col, label, value in zip(
        [col1, col2, col3, col4],
        ["Predicted Price", "Market Segment", "Model R²", "Carat Weight"],
        [f"${pred_price:,.0f}", segment, f"{r2:.4f}", f"{carat} ct"]
    ):
        col.markdown(kpi(label, value), unsafe_allow_html=True)