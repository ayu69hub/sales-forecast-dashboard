import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("📊 Sales Forecasting & Demand Intelligence")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_csv("train.csv")

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d/%m/%Y"
)

# Create Year and Month columns
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# =================================================
# PAGE 1 : SALES OVERVIEW
# =================================================
if page == "Sales Overview":

    st.header("📊 Sales Overview Dashboard")

    # -------------------------------
    # Total Sales by Year
    # -------------------------------
    yearly_sales = df.groupby("Year")["Sales"].sum()

    st.subheader("Total Sales by Year")

    fig, ax = plt.subplots(figsize=(8,5))
    yearly_sales.plot(
        kind="bar",
        color="steelblue",
        ax=ax
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")
    ax.set_title("Yearly Sales")

    st.pyplot(fig)

    # -------------------------------
    # Monthly Sales Trend
    # -------------------------------
    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        df.groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        monthly_sales["Month"],
        monthly_sales["Sales"],
        marker="o"
    )

    plt.xticks(rotation=90)

    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.set_title("Monthly Sales Trend")

    st.pyplot(fig)

    # -------------------------------
    # Region Filter
    # -------------------------------
    st.subheader("Sales by Region")

    region = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    region_df = df[df["Region"] == region]

    st.write("Total Sales :", round(region_df["Sales"].sum(),2))

    fig, ax = plt.subplots(figsize=(7,4))

    region_df.groupby("Category")["Sales"].sum().plot(
        kind="bar",
        color="orange",
        ax=ax
    )

    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # -------------------------------
    # Category Filter
    # -------------------------------
    st.subheader("Sales by Category")

    category = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    category_df = df[df["Category"] == category]

    st.write("Total Sales :", round(category_df["Sales"].sum(),2))

    st.dataframe(
        category_df[
            [
                "Order Date",
                "Region",
                "Category",
                "Sub-Category",
                "Sales"
            ]
        ].head(15)
    )

# =================================================
# PAGE 2
# =================================================
# =================================================
# PAGE 2 : FORECAST EXPLORER
# =================================================
elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    # -------------------------------
    # Choose Forecast Type
    # -------------------------------
    forecast_type = st.radio(
        "Choose Forecast Type",
        ["Category", "Region"]
    )

    # -------------------------------
    # Forecast Data
    # -------------------------------
    category_forecast = {
        "Furniture": [31425.168, 9716.003, 6214.6865],
        "Technology": [21819.178, 20182.088, 24370.316],
        "Office Supplies": [29633.475, 25796.031, 25957.260]
    }

    region_forecast = {
        "West": [29677.768, 11175.512, 15125.339],
        "East": [19581.693, 25088.459, 25353.447]
    }

    # -------------------------------
    # Category Forecast
    # -------------------------------
    if forecast_type == "Category":

        selected = st.selectbox(
            "Select Category",
            list(category_forecast.keys())
        )

        forecast = category_forecast[selected]

    # -------------------------------
    # Region Forecast
    # -------------------------------
    else:

        selected = st.selectbox(
            "Select Region",
            list(region_forecast.keys())
        )

        forecast = region_forecast[selected]

    # -------------------------------
    # Forecast Horizon
    # -------------------------------
    months = st.slider(
        "Forecast Horizon (Months)",
        1,
        3,
        3
    )

    forecast = forecast[:months]

    # -------------------------------
    # Forecast Chart
    # -------------------------------
    st.subheader("Forecast Output")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(
        range(1, months+1),
        forecast,
        marker="o",
        linewidth=3,
        color="green"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Predicted Sales")
    ax.set_title(f"{selected} Sales Forecast")

    st.pyplot(fig)

    # -------------------------------
    # Forecast Table
    # -------------------------------
    forecast_df = pd.DataFrame({
        "Month": [f"Month {i}" for i in range(1, months+1)],
        "Forecast Sales": forecast
    })

    st.subheader("Forecast Values")

    st.dataframe(forecast_df)

    # -------------------------------
    # Model Performance
    # -------------------------------
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "XGBoost")

    col2.metric(
        "MAE",
        "13,915.32"
    )

    col3.metric(
        "RMSE",
        "18,893.85"
    )

    st.success("Best Performing Model: XGBoost")

# =================================================
# PAGE 3
# =================================================
# =================================================
# PAGE 3 : ANOMALY REPORT
# =================================================
elif page == "Anomaly Report":

    st.header("⚠️ Anomaly Detection Report")

    from sklearn.ensemble import IsolationForest

    # Weekly Sales
    weekly_sales = (
        df.resample('W', on='Order Date')['Sales']
        .sum()
        .reset_index()
    )

    # Isolation Forest
    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["anomaly"] = iso.fit_predict(
        weekly_sales[["Sales"]]
    )

    anomalies = weekly_sales[
        weekly_sales["anomaly"] == -1
    ]

    st.subheader("Isolation Forest Anomaly Detection")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        weekly_sales["Order Date"],
        weekly_sales["Sales"],
        label="Weekly Sales"
    )

    ax.scatter(
        anomalies["Order Date"],
        anomalies["Sales"],
        color="red",
        s=80,
        label="Anomaly"
    )

    ax.set_title("Isolation Forest Anomaly Detection")

    ax.legend()

    st.pyplot(fig)

    # -------------------------
    # Summary Metrics
    # -------------------------
    st.subheader("Summary")

    col1, col2 = st.columns(2)

    col1.metric(
        "Isolation Forest",
        len(anomalies)
    )

    col2.metric(
        "Z-Score",
        0
    )

    # -------------------------
    # Anomaly Table
    # -------------------------
    st.subheader("Detected Anomalies")

    st.dataframe(
        anomalies[
            ["Order Date","Sales"]
        ]
    )

# =================================================
# PAGE 4
# =================================================
# =================================================
# PAGE 4 : PRODUCT DEMAND SEGMENTS
# =================================================
elif page == "Product Demand Segments":

    st.header("📦 Product Demand Segments")

    st.success("Optimal Number of Clusters (K) = 4")

    cluster_df = pd.DataFrame({

        "Sub-Category":[
            "Accessories",
            "Appliances",
            "Art",
            "Binders",
            "Bookcases",
            "Chairs",
            "Copiers",
            "Envelopes",
            "Fasteners",
            "Furnishings",
            "Labels",
            "Machines",
            "Paper",
            "Phones",
            "Storage",
            "Supplies",
            "Tables"
        ],

        "Cluster":[
            2,1,1,2,1,2,0,1,1,1,1,0,1,2,2,3,2
        ]

    })

    st.subheader("Sub-Category Demand Clusters")

    st.dataframe(cluster_df)

    st.subheader("Cluster Interpretation")

    st.markdown("""
### Cluster 0
High Value Premium Products

### Cluster 1
Low Volume Stable Demand

### Cluster 2
High Volume Stable Demand

### Cluster 3
High Volatility Demand
""")

    st.subheader("Cluster Distribution")

    cluster_count = cluster_df["Cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7,4))

    cluster_count.plot(
        kind="bar",
        color="purple",
        ax=ax
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Sub-Categories")
    ax.set_title("Demand Cluster Distribution")

    st.pyplot(fig)