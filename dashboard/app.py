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

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .insight-box {
        background-color: #f0f4ff;
        border-left: 4px solid #4a6cf7;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .guide-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 4px;
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

# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(_df):
    # Clustering
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
    return clusters, cluster_names, model, r2

clusters, cluster_names, reg_model, r2 = train_models(df)
df['cluster'] = clusters
df['cluster_label'] = df['cluster'].map(cluster_names)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💎 Diamond Analysis")
st.sidebar.markdown("**CET242 – DA&V | Spring 2026**")
st.sidebar.markdown("**Team:** Mustafa Elsherif & Rawda Attia")
st.sidebar.divider()

page = st.sidebar.radio("📂 Navigate", [
    "📊 Overview",
    "🔍 Deep Analysis",
    "💎 Market Segments",
    "🔮 Price Predictor"
])

st.sidebar.divider()
st.sidebar.markdown("""
### 📖 How to Use
- **Overview** → Start here to understand the dataset and KPIs
- **Deep Analysis** → Explore how each feature affects price
- **Market Segments** → See natural diamond market groups
- **Price Predictor** → Enter diamond specs to predict price
""")

st.sidebar.divider()
st.sidebar.markdown("""
### 📌 Diamond Grading Guide
| Feature | Best | Worst |
|---------|------|-------|
| Cut | Ideal | Fair |
| Color | D | J |
| Clarity | IF | I1 |
""")

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "📊 Overview":
    st.title("💎 Diamond Market Price Analysis")
    st.markdown("**CET242 – Data Analytics and Visualization | Spring 2026**")
    st.markdown("""
    > This dashboard analyzes **53,772 diamonds** to understand what drives diamond prices
    > in the retail gemstone market. Use the sidebar to navigate between sections.
    """)
    st.divider()

    # --- KPIs ---
    st.subheader("📌 Key Performance Indicators (KPIs)")
    st.markdown('<div class="guide-box">These 5 KPIs summarize the most important metrics in the dataset. They give you a quick snapshot of the market before diving into deeper analysis.</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💎 Total Diamonds", f"{len(df):,}")
    col2.metric("💰 Avg Price", f"${df['price'].mean():,.0f}")
    col3.metric("⚖️ Avg Price/Carat", f"${df['price'].sum()/df['carat'].sum():,.0f}")
    col4.metric("📉 Min Price", f"${df['price'].min():,}")
    col5.metric("📈 Max Price", f"${df['price'].max():,}")

    st.divider()

    # --- Price Distribution ---
    st.subheader("💵 Price Distribution")
    st.markdown('<div class="guide-box">These two charts show how diamond prices are distributed across the dataset. The left chart shows the raw distribution, the right shows the log-transformed version which reveals the true shape.</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='price', nbins=50,
                           title='Price Distribution (Original)',
                           color_discrete_sequence=['steelblue'])
        fig.update_layout(xaxis_title='Price ($)', yaxis_title='Count',
                         bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">📊 <b>Reading this chart:</b> Most diamonds are priced under $2,500. The distribution is heavily right-skewed — a small number of very expensive diamonds pull the average up.</div>', unsafe_allow_html=True)

    with col2:
        fig = px.histogram(df, x=np.log(df['price']), nbins=50,
                           title='Log(Price) Distribution',
                           color_discrete_sequence=['coral'])
        fig.update_layout(xaxis_title='Log Price', yaxis_title='Count',
                         bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">📊 <b>Why log scale?</b> Log transformation converts the skewed distribution into a near-normal bell curve, which makes statistical analysis more accurate and reveals patterns hidden in the original scale.</div>', unsafe_allow_html=True)

    st.divider()

    # --- Carat Distribution ---
    st.subheader("⚖️ Carat Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='carat', nbins=50,
                           title='Carat Distribution',
                           color_discrete_sequence=['mediumpurple'])
        fig.update_layout(xaxis_title='Carat', yaxis_title='Count', bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">📊 <b>Reading this chart:</b> Most diamonds are under 1 carat. Notice the spikes at 0.5, 1.0, 1.5, and 2.0 carats — buyers prefer "round number" sizes, creating natural demand clusters.</div>', unsafe_allow_html=True)

    with col2:
        fig = px.scatter(df.sample(5000, random_state=42),
                         x='carat', y='price',
                         title='Price vs Carat',
                         opacity=0.3,
                         color_discrete_sequence=['coral'])
        fig.update_layout(xaxis_title='Carat', yaxis_title='Price ($)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">📊 <b>Key finding:</b> Strong positive relationship between carat and price (correlation = 0.92). However, the relationship is non-linear — price grows exponentially with carat, not linearly.</div>', unsafe_allow_html=True)

    st.divider()

    # --- Category Distributions ---
    st.subheader("🏷️ Category Distributions")
    st.markdown('<div class="guide-box">These charts show how many diamonds exist in each quality category. Understanding the supply distribution helps explain pricing patterns.</div>', unsafe_allow_html=True)
    st.write("")

    cut_order     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_order   = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity_order = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']

    col1, col2, col3 = st.columns(3)
    with col1:
        cut_counts = df['cut'].value_counts().loc[cut_order].reset_index()
        fig = px.bar(cut_counts, x='cut', y='count',
                     title='Diamond Count by Cut',
                     color_discrete_sequence=['steelblue'])
        fig.update_layout(xaxis_title='Cut Quality', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">Ideal cut is the most common (~40%). Fair cut is the rarest — only ~1.5% of diamonds.</div>', unsafe_allow_html=True)

    with col2:
        color_counts = df['color'].value_counts().loc[color_order].reset_index()
        fig = px.bar(color_counts, x='color', y='count',
                     title='Diamond Count by Color',
                     color_discrete_sequence=['mediumseagreen'])
        fig.update_layout(xaxis_title='Color Grade', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">G color is the most common. D (best colorless) and J (most yellow) are the rarest grades.</div>', unsafe_allow_html=True)

    with col3:
        clarity_counts = df['clarity'].value_counts().loc[clarity_order].reset_index()
        fig = px.bar(clarity_counts, x='clarity', y='count',
                     title='Diamond Count by Clarity',
                     color_discrete_sequence=['tomato'])
        fig.update_layout(xaxis_title='Clarity Grade', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">SI1 and VS2 are the most common clarity grades. IF (internally flawless) is the rarest — less than 1.5% of diamonds.</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 2 — DEEP ANALYSIS
# ============================================================

elif page == "🔍 Deep Analysis":
    st.title("🔍 Deep Analysis")
    st.markdown("""
    > This section explores **how each feature affects diamond price**.
    > Each chart is followed by an explanation of what it means and why it matters.
    """)
    st.divider()

    cut_order     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_order   = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity_order = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']

    # --- Correlation ---
    st.subheader("🔗 Feature Correlation with Price")
    st.markdown('<div class="guide-box">Correlation measures how strongly each feature moves with price. Values close to 1.0 mean strong positive relationship, close to -1.0 mean strong negative, close to 0 means no relationship.</div>', unsafe_allow_html=True)
    st.write("")

    numeric_cols = df[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']].corr()
    price_corr = numeric_cols['price'].drop('price').sort_values(ascending=True)

    fig = px.bar(x=price_corr.values, y=price_corr.index,
                 orientation='h',
                 title='Feature Correlation with Price',
                 color=price_corr.values,
                 color_continuous_scale='RdBu',
                 range_color=[-1, 1])
    fig.update_layout(xaxis_title='Correlation', yaxis_title='Feature',
                     coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📊 <b>Key findings:</b><br>
    - <b>carat (0.92)</b> — Strongest predictor by far. Bigger diamond = much higher price<br>
    - <b>x, y, z (0.88-0.89)</b> — Physical dimensions are highly correlated with carat, so they also correlate strongly with price<br>
    - <b>depth (-0.01)</b> — Almost zero correlation. Depth percentage alone tells us almost nothing about price<br>
    - <b>table (0.13)</b> — Very weak. Table size has minimal effect on price
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Price by Cut ---
    st.subheader("✂️ Price vs Cut Quality")
    st.markdown('<div class="guide-box">Cut quality ranges from Fair (lowest) to Ideal (best). This section tests whether better cut = higher price.</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        avg_cut = df.groupby('cut')['price'].mean().loc[cut_order].reset_index()
        fig = px.bar(avg_cut, x='cut', y='price',
                     title='Avg Price by Cut Quality',
                     color='cut',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(xaxis_title='Cut', yaxis_title='Avg Price ($)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_carat_cut = df.groupby('cut')['carat'].mean().loc[cut_order].reset_index()
        fig = px.bar(avg_carat_cut, x='cut', y='carat',
                     title='Avg Carat by Cut Quality',
                     color='cut',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(xaxis_title='Cut', yaxis_title='Avg Carat', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📊 <b>Surprising finding — Hypothesis H2 REJECTED:</b><br>
    Ideal cut has a <b>lower</b> average price ($3,462) than Premium ($4,578).<br>
    <b>Why?</b> Look at the right chart — Ideal cut diamonds are <b>smaller on average (0.703 carat)</b> than Fair cut (1.044 carat).<br>
    Since carat dominates price, smaller high-quality diamonds end up cheaper than larger low-quality ones.<br>
    This is called the <b>Carat Size Confounding Effect</b>.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Price by Color ---
    st.subheader("🎨 Price vs Color Grade")
    st.markdown('<div class="guide-box">Color grades D (colorless/best) → J (light yellow/worst). Better color should logically mean higher price — but the data tells a different story.</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        avg_color = df.groupby('color')['price'].mean().loc[color_order].reset_index()
        fig = px.bar(avg_color, x='color', y='price',
                     title='Avg Price by Color Grade',
                     color='color',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(xaxis_title='Color (D=best → J=worst)',
                         yaxis_title='Avg Price ($)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_carat_color = df.groupby('color')['carat'].mean().loc[color_order].reset_index()
        fig = px.bar(avg_carat_color, x='color', y='carat',
                     title='Avg Carat by Color Grade',
                     color='color',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(xaxis_title='Color (D=best → J=worst)',
                         yaxis_title='Avg Carat', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📊 <b>Surprising finding — Hypothesis H3 REJECTED:</b><br>
    D color (best) avg price = $3,172 vs J color (worst) avg price = $5,326.<br>
    <b>Why?</b> J color diamonds are significantly larger on average (1.163 carat vs 0.658 carat for D).<br>
    <b>Practical advice for buyers:</b> D-F color grades offer the best color value — you pay less because they come in smaller sizes, not because they are lower quality.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Price by Clarity ---
    st.subheader("🔬 Price vs Clarity Grade")
    st.markdown('<div class="guide-box">Clarity measures internal flaws. IF (Internally Flawless) is the best, I1 (Included) is the worst. Again, logic says better clarity = higher price.</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        avg_clarity = df.groupby('clarity')['price'].mean().loc[clarity_order].reset_index()
        fig = px.bar(avg_clarity, x='clarity', y='price',
                     title='Avg Price by Clarity Grade',
                     color='clarity',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(xaxis_title='Clarity (IF=best → I1=worst)',
                         yaxis_title='Avg Price ($)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_carat_clarity = df.groupby('clarity')['carat'].mean().loc[clarity_order].reset_index()
        fig = px.bar(avg_carat_clarity, x='clarity', y='carat',
                     title='Avg Carat by Clarity Grade',
                     color='clarity',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(xaxis_title='Clarity (IF=best → I1=worst)',
                         yaxis_title='Avg Carat', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📊 <b>Surprising finding — Hypothesis H4 REJECTED:</b><br>
    IF clarity (best) avg price = $2,870 — the LOWEST of all grades.<br>
    I1 (worst clarity) avg carat = 1.284 vs IF avg carat = 0.506.<br>
    <b>Practical advice:</b> For large diamonds, VS1-VS2 clarity offers the best quality-to-price balance.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Root Cause Summary ---
    st.subheader("🧠 Root Cause Summary")
    root_data = {
        'Category': ['Cut: Ideal vs Fair', 'Color: D vs J', 'Clarity: IF vs I1'],
        'Best Grade Avg Carat': [0.703, 0.658, 0.506],
        'Worst Grade Avg Carat': [1.044, 1.163, 1.284],
        'Carat Difference': ['+49%', '+77%', '+154%']
    }
    st.dataframe(pd.DataFrame(root_data), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box">
    🧠 <b>Root Cause: Carat Size Confounding Effect</b><br>
    In every quality dimension (cut, color, clarity), the worst grade diamonds are significantly larger in carat.<br>
    Since carat has a 0.92 correlation with price, larger diamonds always appear more expensive on average —
    even when their quality is lower. This masks the true premium of quality grades.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 3 — MARKET SEGMENTS
# ============================================================

elif page == "💎 Market Segments":
    st.title("💎 Diamond Market Segments")
    st.markdown("""
    > Using **K-Means Clustering**, we grouped all 53,772 diamonds into 4 natural market segments
    > based on their carat weight and price. This reveals the natural structure of the diamond market.
    """)
    st.divider()

    st.markdown("""
    <div class="guide-box">
    <b>How to read this page:</b><br>
    - Each dot represents a diamond, colored by its market segment<br>
    - The X-axis shows carat weight, Y-axis shows price<br>
    - Segments are determined automatically by the K-Means algorithm — not manually defined
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    # Cluster scatter
    fig = px.scatter(df.sample(10000, random_state=42),
                     x='carat', y='price',
                     color='cluster_label',
                     title='Diamond Market Segments (K-Means, k=4) — Sample of 10,000',
                     opacity=0.4,
                     category_orders={'cluster_label': ['Budget', 'Mid-Range', 'Upper Mid-Range', 'Luxury']},
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(xaxis_title='Carat Weight', yaxis_title='Price ($)',
                     legend_title='Market Segment')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📊 <b>Reading this chart:</b><br>
    The 4 segments are clearly separated along the carat axis — confirming that carat is the dominant
    factor that naturally divides the diamond market into price tiers.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Segment summary
    st.subheader("📋 Segment Summary")
    summary = df.groupby('cluster_label').agg(
        Count=('price', 'count'),
        Avg_Carat=('carat', 'mean'),
        Avg_Price=('price', 'mean'),
        Min_Price=('price', 'min'),
        Max_Price=('price', 'max')
    ).round(2)
    summary = summary.loc[['Budget', 'Mid-Range', 'Upper Mid-Range', 'Luxury']]
    summary.columns = ['Count', 'Avg Carat', 'Avg Price ($)', 'Min Price ($)', 'Max Price ($)']
    st.dataframe(summary, use_container_width=True)

    st.divider()

    # Segment breakdown charts
    st.subheader("🏷️ Segment Breakdown by Quality")
    st.markdown('<div class="guide-box">These charts show the quality distribution within each market segment — helping understand what kind of diamonds dominate each price tier.</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        cut_seg = df.groupby(['cluster_label', 'cut']).size().reset_index(name='count')
        fig = px.bar(cut_seg, x='cluster_label', y='count', color='cut',
                     title='Cut Distribution per Segment',
                     category_orders={
                         'cluster_label': ['Budget', 'Mid-Range', 'Upper Mid-Range', 'Luxury'],
                         'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
                     },
                     barmode='group')
        fig.update_layout(xaxis_title='Segment', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        clarity_seg = df.groupby(['cluster_label', 'clarity']).size().reset_index(name='count')
        fig = px.bar(clarity_seg, x='cluster_label', y='count', color='clarity',
                     title='Clarity Distribution per Segment',
                     category_orders={
                         'cluster_label': ['Budget', 'Mid-Range', 'Upper Mid-Range', 'Luxury'],
                         'clarity': ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
                     },
                     barmode='group')
        fig.update_layout(xaxis_title='Segment', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📌 Segment Insights")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="insight-box">
        💚 <b>Budget</b><br>
        Small diamonds under 0.5 carat.<br>
        Entry-level market.<br>
        Price range: $326 – ~$2,000
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">
        🔵 <b>Mid-Range</b><br>
        Around 1 carat diamonds.<br>
        Most popular engagement ring size.<br>
        Price range: ~$2,000 – ~$6,000
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="insight-box">
        🟠 <b>Upper Mid-Range</b><br>
        Around 1.5-2 carat diamonds.<br>
        Premium retail segment.<br>
        Price range: ~$6,000 – ~$11,000
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="insight-box">
        🔴 <b>Luxury</b><br>
        2+ carat diamonds.<br>
        High-end collectors market.<br>
        Price range: ~$11,000 – $18,823
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 4 — PRICE PREDICTOR
# ============================================================

elif page == "🔮 Price Predictor":
    st.title("🔮 Diamond Price Predictor")
    st.markdown(f"**Model:** Polynomial Regression (degree=2) + Ridge Regularization | **R² = {r2:.4f}**")
    st.divider()

    st.markdown("""
    <div class="guide-box">
    <b>How to use this tool:</b><br>
    1. Adjust the sliders and dropdowns on the left to match your diamond's specifications<br>
    2. The predicted price updates automatically at the bottom<br>
    3. Use the grading guides below each input to understand what each value means
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⚖️ Physical Features")
        carat = st.slider("Carat Weight", 0.2, 5.0, 1.0, 0.01,
                          help="Diamond weight. 1 carat = 0.2 grams. Most popular sizes: 0.5, 1.0, 1.5, 2.0")
        st.caption("💡 Carat has the strongest effect on price (correlation = 0.92)")

        depth = st.slider("Depth %", 43.0, 79.0, 61.7, 0.1,
                          help="Total depth percentage = z / mean(x,y) × 100. Ideal range: 59-62.5%")
        st.caption("💡 Ideal depth: 59–62.5% for round diamonds")

        table = st.slider("Table %", 43.0, 95.0, 57.0, 0.1,
                          help="Width of the top flat surface as % of diameter. Ideal range: 53-57%")
        st.caption("💡 Ideal table: 53–57% for maximum brilliance")

    with col2:
        st.subheader("🏷️ Quality Grades")
        cut = st.selectbox("Cut Quality",
                           ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                           index=4,
                           help="How well the diamond was cut and shaped")
        st.caption("""
        - **Fair** → Poor light reflection
        - **Good** → Below average
        - **Very Good** → Above average
        - **Premium** → Excellent proportions
        - **Ideal** → Maximum brilliance ⭐
        """)

        color = st.selectbox("Color Grade",
                             ['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                             index=3,
                             help="D=colorless (best), J=light yellow (worst)")
        st.caption("""
        - **D-F** → Colorless (best, rarest)
        - **G-H** → Near colorless (most popular)
        - **I-J** → Slightly yellow (visible to naked eye)
        """)

        clarity = st.selectbox("Clarity Grade",
                               ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'],
                               index=3,
                               help="IF=no inclusions (best), I1=visible inclusions (worst)")
        st.caption("""
        - **IF** → Internally Flawless (rarest)
        - **VVS1-VVS2** → Tiny inclusions
        - **VS1-VS2** → Minor inclusions (best value ⭐)
        - **SI1-SI2** → Noticeable inclusions
        - **I1** → Visible to naked eye
        """)

    with col3:
        st.subheader("📐 Dimensions (mm)")
        x = st.slider("X — Length (mm)", 0.0, 11.0, 6.4, 0.01,
                      help="Length of the diamond in millimeters")
        y = st.slider("Y — Width (mm)", 0.0, 11.0, 6.4, 0.01,
                      help="Width of the diamond in millimeters")
        z = st.slider("Z — Depth (mm)", 0.0, 7.0, 4.0, 0.01,
                      help="Depth of the diamond in millimeters")
        st.caption("💡 For a 1-carat round diamond: x≈6.4mm, y≈6.4mm, z≈3.9mm")

        st.write("")
        st.markdown("""
        <div class="guide-box">
        <b>Typical dimensions by carat:</b><br>
        0.5ct → ~5.1mm diameter<br>
        1.0ct → ~6.4mm diameter<br>
        1.5ct → ~7.3mm diameter<br>
        2.0ct → ~8.1mm diameter
        </div>
        """, unsafe_allow_html=True)

    # Encode and predict
    cut_enc     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'].index(cut) + 1
    color_enc   = ['J', 'I', 'H', 'G', 'F', 'E', 'D'].index(color) + 1
    clarity_enc = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'].index(clarity) + 1

    features = np.array([[carat, depth, table, x, y, z, cut_enc, color_enc, clarity_enc]])
    pred_price = np.exp(reg_model.predict(features)[0])

    # Determine segment
    if pred_price < 2000:
        segment = "💚 Budget"
    elif pred_price < 6000:
        segment = "🔵 Mid-Range"
    elif pred_price < 11000:
        segment = "🟠 Upper Mid-Range"
    else:
        segment = "🔴 Luxury"

    st.divider()
    st.subheader("💰 Prediction Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Predicted Price", f"${pred_price:,.0f}")
    col2.metric("📦 Market Segment", segment)
    col3.metric("📊 Model Accuracy", f"R² = {r2:.4f}")

    st.divider()
    st.subheader("📌 Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
        <b>How the model works:</b><br>
        1. Takes your 9 diamond features as input<br>
        2. Applies polynomial transformation (degree=2) to capture non-linear relationships<br>
        3. Uses Ridge regression to predict log(price)<br>
        4. Converts back from log scale to get actual price in $
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.success(f"**R² = {r2:.4f}** → The model explains **{r2*100:.2f}%** of price variance ✅")
        st.info("**What is R²?** It measures how well the model fits the data. R² = 1.0 means perfect prediction. R² = 0.9858 is excellent — only ~1.4% of price variance is unexplained.")