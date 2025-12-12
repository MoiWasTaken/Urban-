# app.py - UrbanMart Retail Insights Dashboard
# Professional Investor-Ready Dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="UrbanMart Retail Insights",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS FOR PROFESSIONAL STYLING
# ========================================
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
    }
    h2 {
        color: #374151;
        font-weight: 600;
        margin-top: 30px;
    }
    h3 {
        color: #4b5563;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# DATA LOADING & PROCESSING
# ========================================
@st.cache_data
def load_and_process_data():
    """Load and process the UrbanMart sales data"""
    try:
        # Load CSV
        df = pd.read_csv('urbanmart_sales.csv')
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Create derived fields
        df['line_revenue'] = df['quantity'] * df['unit_price'] - df['discount_applied']
        
        # Extract month and day of week
        df['month'] = df['date'].dt.to_period('M').astype(str)
        df['day_of_week'] = df['date'].dt.day_name()
        
        # Category profit margins
        margin_map = {
            'Beverages': 0.30,
            'Snacks': 0.25,
            'Personal Care': 0.40,
            'Dairy': 0.20,
            'Bakery': 0.20
        }
        
        df['profit_margin'] = df['product_category'].map(margin_map)
        df['estimated_profit'] = df['line_revenue'] * df['profit_margin']
        
        return df
    
    except FileNotFoundError:
        st.error("❌ Error: urbanmart_sales.csv not found. Please ensure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Load data
df_original = load_and_process_data()

# ========================================
# SIDEBAR - FILTERS
# ========================================
st.sidebar.image("https://via.placeholder.com/300x80/1f2937/ffffff?text=UrbanMart", use_container_width=True)
st.sidebar.title("🎯 Dashboard Filters")
st.sidebar.markdown("---")

# Date Range Filter
st.sidebar.subheader("📅 Date Range")
min_date = df_original['date'].min().date()
max_date = df_original['date'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Handle single date selection
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

# Store Location Filter
st.sidebar.subheader("🏪 Store Location")
all_locations = sorted(df_original['store_location'].unique().tolist())
selected_locations = st.sidebar.multiselect(
    "Select Locations",
    options=all_locations,
    default=all_locations
)

# Channel Filter
st.sidebar.subheader("🛒 Sales Channel")
selected_channel = st.sidebar.selectbox(
    "Select Channel",
    options=['All', 'In-store', 'Online']
)

# Product Category Filter
st.sidebar.subheader("📦 Product Category")
all_categories = sorted(df_original['product_category'].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=all_categories,
    default=all_categories
)

# Customer Segment Filter
st.sidebar.subheader("👥 Customer Segment")
all_segments = sorted(df_original['customer_segment'].unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Select Segments",
    options=all_segments,
    default=all_segments
)

# Discount Range Filter
st.sidebar.subheader("💰 Discount Range")
min_discount = float(df_original['discount_applied'].min())
max_discount = float(df_original['discount_applied'].max())
discount_range = st.sidebar.slider(
    "Discount Applied ($)",
    min_value=min_discount,
    max_value=max_discount,
    value=(min_discount, max_discount)
)

# Quantity Range Filter
st.sidebar.subheader("📊 Quantity Range")
min_qty = int(df_original['quantity'].min())
max_qty = int(df_original['quantity'].max())
quantity_range = st.sidebar.slider(
    "Quantity Purchased",
    min_value=min_qty,
    max_value=max_qty,
    value=(min_qty, max_qty)
)

# High-Value Customer Filter
st.sidebar.subheader("⭐ High-Value Customers")
high_value_threshold = st.sidebar.number_input(
    "Minimum Customer Revenue ($)",
    min_value=0,
    max_value=1000,
    value=100,
    step=10
)
show_high_value_only = st.sidebar.checkbox("Show Only High-Value Customers")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use filters to drill down into specific segments and time periods.")

# ========================================
# APPLY FILTERS
# ========================================
df = df_original.copy()

# Apply date filter
df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]

# Apply location filter
if selected_locations:
    df = df[df['store_location'].isin(selected_locations)]

# Apply channel filter
if selected_channel != 'All':
    df = df[df['channel'] == selected_channel]

# Apply category filter
if selected_categories:
    df = df[df['product_category'].isin(selected_categories)]

# Apply segment filter
if selected_segments:
    df = df[df['customer_segment'].isin(selected_segments)]

# Apply discount filter
df = df[(df['discount_applied'] >= discount_range[0]) & (df['discount_applied'] <= discount_range[1])]

# Apply quantity filter
df = df[(df['quantity'] >= quantity_range[0]) & (df['quantity'] <= quantity_range[1])]

# Apply high-value customer filter
if show_high_value_only:
    customer_revenue = df.groupby('customer_id')['line_revenue'].sum()
    high_value_customers = customer_revenue[customer_revenue >= high_value_threshold].index
    df = df[df['customer_id'].isin(high_value_customers)]

# ========================================
# MAIN DASHBOARD
# ========================================
st.title("🏪 UrbanMart Retail Insights Dashboard")
st.markdown("### Professional Analytics for Strategic Decision Making")
st.markdown("---")

# Check if data is available after filtering
if len(df) == 0:
    st.warning("⚠️ No data available for the selected filters. Please adjust your filter criteria.")
    st.stop()

# ========================================
# SECTION A - EXECUTIVE KPIs
# ========================================
st.header("📊 Executive KPIs")

# Calculate KPIs
total_revenue = df['line_revenue'].sum()
total_transactions = df['transaction_id'].nunique()
total_bills = df['bill_id'].nunique()
avg_order_value = total_revenue / total_bills if total_bills > 0 else 0

# Calculate monthly growth
monthly_revenue = df.groupby('month')['line_revenue'].sum().sort_index()
if len(monthly_revenue) >= 2:
    current_month_rev = monthly_revenue.iloc[-1]
    previous_month_rev = monthly_revenue.iloc[-2]
    monthly_growth = ((current_month_rev - previous_month_rev) / previous_month_rev * 100) if previous_month_rev > 0 else 0
else:
    monthly_growth = 0

# Calculate repeat purchase rate
customer_orders = df.groupby('customer_id')['bill_id'].nunique()
repeat_customers = (customer_orders > 1).sum()
total_customers = customer_orders.count()
repeat_rate = (repeat_customers / total_customers * 100) if total_customers > 0 else 0

# Display KPIs in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="💵 Total Revenue",
        value=f"${total_revenue:,.2f}",
        delta=None
    )

with col2:
    growth_color = "normal" if monthly_growth >= 0 else "inverse"
    st.metric(
        label="📈 Monthly Growth",
        value=f"{monthly_growth:.1f}%",
        delta=f"{monthly_growth:.1f}%",
        delta_color=growth_color
    )

with col3:
    st.metric(
        label="🛍️ Avg Order Value",
        value=f"${avg_order_value:.2f}",
        delta=None
    )

with col4:
    st.metric(
        label="🔄 Repeat Purchase Rate",
        value=f"{repeat_rate:.1f}%",
        delta=None
    )

st.markdown("---")

# ========================================
# SECTION B - STORE PERFORMANCE
# ========================================
st.header("🏪 Store Performance Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    # Revenue by Store - Horizontal Bar Chart
    revenue_by_store = df.groupby('store_location')['line_revenue'].sum().sort_values(ascending=True)
    
    fig_store = go.Figure(go.Bar(
        x=revenue_by_store.values,
        y=revenue_by_store.index,
        orientation='h',
        marker=dict(
            color=revenue_by_store.values,
            colorscale='Blues',
            showscale=False
        ),
        text=[f'${x:,.0f}' for x in revenue_by_store.values],
        textposition='outside'
    ))
    
    fig_store.update_layout(
        title="Revenue by Store Location",
        xaxis_title="Revenue ($)",
        yaxis_title="Store Location",
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_store, use_container_width=True)

with col2:
    # Store Leaderboard Table
    st.subheader("🏆 Store Leaderboard")
    
    store_stats = df.groupby('store_location').agg({
        'line_revenue': 'sum',
        'bill_id': 'nunique',
        'transaction_id': 'count'
    }).round(2)
    
    store_stats.columns = ['Revenue', 'Orders', 'Items']
    store_stats['AOV'] = (store_stats['Revenue'] / store_stats['Orders']).round(2)
    store_stats = store_stats.sort_values('Revenue', ascending=False)
    store_stats['Revenue'] = store_stats['Revenue'].apply(lambda x: f'${x:,.2f}')
    store_stats['AOV'] = store_stats['AOV'].apply(lambda x: f'${x:,.2f}')
    
    st.dataframe(store_stats, use_container_width=True)

# AOV by Store
st.subheader("📊 Average Order Value by Store")
aov_by_store = df.groupby('store_location').agg({
    'line_revenue': 'sum',
    'bill_id': 'nunique'
})
aov_by_store['AOV'] = aov_by_store['line_revenue'] / aov_by_store['bill_id']
aov_by_store = aov_by_store.sort_values('AOV', ascending=False)

fig_aov = px.bar(
    aov_by_store,
    x=aov_by_store.index,
    y='AOV',
    color='AOV',
    color_continuous_scale='Blues',
    text='AOV'
)

fig_aov.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
fig_aov.update_layout(
    xaxis_title="Store Location",
    yaxis_title="Average Order Value ($)",
    showlegend=False,
    height=400,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

st.plotly_chart(fig_aov, use_container_width=True)

st.markdown("---")

# ========================================
# SECTION C - CUSTOMER INSIGHTS
# ========================================
st.header("👥 Customer Insights")

col1, col2 = st.columns([1, 2])

with col1:
    # Revenue by Customer Segment - Pie Chart
    revenue_by_segment = df.groupby('customer_segment')['line_revenue'].sum()
    
    fig_segment = go.Figure(data=[go.Pie(
        labels=revenue_by_segment.index,
        values=revenue_by_segment.values,
        hole=0.4,
        marker=dict(colors=['#10b981', '#34d399', '#6ee7b7']),
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig_segment.update_layout(
        title="Revenue Distribution by Customer Segment",
        height=400,
        showlegend=True,
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_segment, use_container_width=True)
    
    # Loyal Customer KPI
    loyal_customers = df[df['customer_segment'] == 'Loyal']['customer_id'].nunique()
    st.metric(
        label="⭐ Loyal Customers",
        value=f"{loyal_customers:,}",
        delta=None
    )

with col2:
    # Top 10 Customers Table
    st.subheader("🌟 Top 10 High-Value Customers")
    
    customer_stats = df.groupby('customer_id').agg({
        'line_revenue': 'sum',
        'bill_id': 'nunique',
        'customer_segment': 'first'
    }).round(2)
    
    customer_stats.columns = ['Total Revenue', 'Order Count', 'Segment']
    customer_stats = customer_stats.sort_values('Total Revenue', ascending=False).head(10)
    customer_stats['Total Revenue'] = customer_stats['Total Revenue'].apply(lambda x: f'${x:,.2f}')
    customer_stats.index.name = 'Customer ID'
    
    st.dataframe(customer_stats, use_container_width=True)

st.markdown("---")

# ========================================
# SECTION D - CATEGORY & PROFITABILITY
# ========================================
st.header("📦 Category & Profitability Analysis")

col1, col2 = st.columns(2)

with col1:
    # Revenue by Category
    revenue_by_category = df.groupby('product_category')['line_revenue'].sum().sort_values(ascending=False)
    
    fig_category = px.bar(
        revenue_by_category,
        x=revenue_by_category.index,
        y=revenue_by_category.values,
        color=revenue_by_category.values,
        color_continuous_scale='Purples',
        text=revenue_by_category.values
    )
    
    fig_category.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_category.update_layout(
        title="Revenue by Product Category",
        xaxis_title="Category",
        yaxis_title="Revenue ($)",
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_category, use_container_width=True)

with col2:
    # Profit by Category
    profit_by_category = df.groupby('product_category')['estimated_profit'].sum().sort_values(ascending=False)
    
    fig_profit = px.bar(
        profit_by_category,
        x=profit_by_category.index,
        y=profit_by_category.values,
        color=profit_by_category.values,
        color_continuous_scale='Purples',
        text=profit_by_category.values
    )
    
    fig_profit.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_profit.update_layout(
        title="Estimated Profit by Category",
        xaxis_title="Category",
        yaxis_title="Profit ($)",
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_profit, use_container_width=True)

# Pareto Chart - Top Products (80/20 Rule)
st.subheader("📊 Product Revenue Concentration (Pareto Analysis)")

product_revenue = df.groupby('product_name')['line_revenue'].sum().sort_values(ascending=False)
product_revenue_cumsum = product_revenue.cumsum()
product_revenue_pct = (product_revenue_cumsum / product_revenue.sum() * 100)

fig_pareto = go.Figure()

# Bar chart for revenue
fig_pareto.add_trace(go.Bar(
    x=product_revenue.head(20).index,
    y=product_revenue.head(20).values,
    name='Revenue',
    marker_color='#9333ea',
    yaxis='y'
))

# Line chart for cumulative percentage
fig_pareto.add_trace(go.Scatter(
    x=product_revenue_pct.head(20).index,
    y=product_revenue_pct.head(20).values,
    name='Cumulative %',
    marker_color='#f59e0b',
    yaxis='y2',
    mode='lines+markers'
))

fig_pareto.update_layout(
    title="Top 20 Products - Revenue & Cumulative Percentage",
    xaxis_title="Product Name",
    yaxis=dict(title="Revenue ($)", side='left'),
    yaxis2=dict(title="Cumulative %", side='right', overlaying='y', range=[0, 100]),
    height=500,
    hovermode='x unified',
    plot_bgcolor='white',
    paper_bgcolor='white'
)

st.plotly_chart(fig_pareto, use_container_width=True)

st.markdown("---")

# ========================================
# SECTION E - CHANNEL MIX
# ========================================
st.header("🛒 Sales Channel Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    # Revenue by Channel - Stacked Bar
    channel_revenue = df.groupby(['month', 'channel'])['line_revenue'].sum().reset_index()
    
    fig_channel = px.bar(
        channel_revenue,
        x='month',
        y='line_revenue',
        color='channel',
        title="Revenue by Channel Over Time",
        color_discrete_map={'In-store': '#3b82f6', 'Online': '#10b981'},
        text='line_revenue'
    )
    
    fig_channel.update_traces(texttemplate='$%{text:,.0f}', textposition='inside')
    fig_channel.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_channel, use_container_width=True)

with col2:
    # Channel Mix KPIs
    st.subheader("📊 Channel Distribution")
    
    total_channel_revenue = df.groupby('channel')['line_revenue'].sum()
    online_revenue = total_channel_revenue.get('Online', 0)
    instore_revenue = total_channel_revenue.get('In-store', 0)
    total_rev = total_channel_revenue.sum()
    
    online_pct = (online_revenue / total_rev * 100) if total_rev > 0 else 0
    instore_pct = (instore_revenue / total_rev * 100) if total_rev > 0 else 0
    
    st.metric(
        label="🌐 Online Revenue %",
        value=f"{online_pct:.1f}%",
        delta=None
    )
    
    st.metric(
        label="🏪 In-Store Revenue %",
        value=f"{instore_pct:.1f}%",
        delta=None
    )
    
    st.metric(
        label="💰 Online Revenue",
        value=f"${online_revenue:,.2f}",
        delta=None
    )
    
    st.metric(
        label="💰 In-Store Revenue",
        value=f"${instore_revenue:,.2f}",
        delta=None
    )

st.markdown("---")

# ========================================
# SECTION F - OPERATIONAL INSIGHTS
# ========================================
st.header("⚙️ Operational Insights")

col1, col2 = st.columns(2)

with col1:
    # Heatmap: Revenue by Day of Week x Store
    st.subheader("🔥 Revenue Heatmap: Day × Store")
    
    # Define day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    heatmap_data = df.groupby(['day_of_week', 'store_location'])['line_revenue'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='store_location', values='line_revenue').fillna(0)
    
    # Reindex to ensure correct day order
    heatmap_pivot = heatmap_pivot.reindex(day_order)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Oranges',
        text=heatmap_pivot.values,
        texttemplate='$%{text:,.0f}',
        textfont={"size": 10},
        colorbar=dict(title="Revenue ($)")
    ))
    
    fig_heatmap.update_layout(
        xaxis_title="Store Location",
        yaxis_title="Day of Week",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    # Quantity Distribution
    st.subheader("📦 Quantity Distribution")
    
    fig_qty = px.histogram(
        df,
        x='quantity',
        nbins=20,
        color_discrete_sequence=['#f97316'],
        title="Distribution of Purchase Quantities"
    )
    
    fig_qty.update_layout(
        xaxis_title="Quantity",
        yaxis_title="Frequency",
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_qty, use_container_width=True)

st.markdown("---")

# ========================================
# SECTION G - RAW DATA PREVIEW
# ========================================
st.header("📄 Raw Data Preview")
st.subheader("Filtered Transaction Data (First 30 Rows)")

# Display columns to show
display_columns = [
    'transaction_id', 'date', 'store_location', 'customer_segment',
    'product_category', 'product_name', 'quantity', 'unit_price',
    'discount_applied', 'line_revenue', 'channel', 'payment_method'
]

st.dataframe(
    df[display_columns].head(30).style.format({
        'unit_price': '${:.2f}',
        'discount_applied': '${:.2f}',
        'line_revenue': '${:.2f}'
    }),
    use_container_width=True,
    height=400
)

# Download button for filtered data
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name=f'urbanmart_filtered_data_{datetime.now().strftime("%Y%m%d")}.csv',
    mime='text/csv',
)

st.markdown("---")

# ========================================
# FOOTER
# ========================================
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        <p>🏪 <strong>UrbanMart Retail Insights Dashboard</strong></p>
        <p>Built with Python & Streamlit | Data-Driven Retail Analytics</p>
        <p style='font-size: 12px;'>© 2025 UrbanMart Analytics Team</p>
    </div>
""", unsafe_allow_html=True)