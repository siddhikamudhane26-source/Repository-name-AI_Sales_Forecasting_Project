import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import mysql.connector

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Sales Forecasting & Customer Intelligence",
    page_icon=":material/analytics:",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("?? AI-Based Sales Forecasting & Customer Intelligence System")

st.markdown(
    "Machine Learning based Sales Forecasting, Customer Segmentation "
    "and Business Intelligence Dashboard"
)


# =========================================================
# FILE PATHS
# =========================================================

DATA_FILE = "data/sales_data.csv"
OUTPUT_DIR = "outputs"

MONTHLY_FILE = os.path.join(
    OUTPUT_DIR,
    "monthly_sales.csv"
)

FORECAST_FILE = os.path.join(
    OUTPUT_DIR,
    "future_sales_forecast.csv"
)

SEGMENT_FILE = os.path.join(
    OUTPUT_DIR,
    "customer_segments.csv"
)

SEGMENT_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "customer_segment_summary.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "model_metrics.csv"
)

TEST_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "forecast_test_results.csv"
)



# =========================================================
# MYSQL DATABASE CONFIGURATION
# =========================================================
# Credentials are read from Streamlit secrets:
# .streamlit/secrets.toml
#
# [mysql]
# host = "127.0.0.1"
# port = 3306
# user = "root"
# password = "YOUR_MYSQL_PASSWORD"
# database = "sales_intelligence_db"
def get_mysql_connection():
    try:
        db = st.secrets["mysql"]

        return mysql.connector.connect(
            host=db.get("host"),
            port=int(db.get("port", 4000)),
            user=db.get("user"),
            password=db["password"],
            database=db.get("database", "sales_intelligence_db"),
            ssl_ca=db.get("ssl_ca"),
            ssl_verify_cert=True,
            ssl_verify_identity=True
        )

    except Exception as e:
        raise ConnectionError(
            "TiDB Cloud connection could not be established. "
            "Check Streamlit Secrets and the database connection. "
            f"Details: {e}"
        )


def load_mysql_data():
    conn = get_mysql_connection()
    try:
        query = """
        SELECT
            Order_ID,
            Order_Date,
            Customer_ID,
            Product_Name AS Product,
            Category,
            Quantity,
            Sales,
            Profit,
            Region_Name AS Region,
            Payment_Mode_Name AS Payment_Mode
        FROM sales_analysis_view
        ORDER BY Order_Date, Order_ID
        """
        df = pd.read_sql(query, conn)
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
        return df
    finally:
        conn.close()


def sync_dataframe_to_mysql(df):
    """
    Synchronize the complete DataFrame into the normalized MySQL schema.
    This keeps the Streamlit CRUD screen and the DBMS tables consistent.
    """
    conn = get_mysql_connection()
    cursor = conn.cursor()

    try:
        clean = df.copy()
        clean["Order_Date"] = pd.to_datetime(
            clean["Order_Date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        for col in ["Quantity", "Sales", "Profit"]:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

        clean = clean.dropna(
            subset=[
                "Order_ID", "Order_Date", "Customer_ID", "Product",
                "Category", "Quantity", "Sales", "Profit",
                "Region", "Payment_Mode"
            ]
        ).drop_duplicates(subset=["Order_ID"])

        # Delete child rows first because of foreign keys.
        cursor.execute("DELETE FROM order_details")
        cursor.execute("DELETE FROM orders")
        cursor.execute("DELETE FROM customers")
        cursor.execute("DELETE FROM products")
        cursor.execute("DELETE FROM regions")
        cursor.execute("DELETE FROM payment_modes")

        # Populate lookup tables.
        regions = clean[["Region"]].drop_duplicates()
        cursor.executemany(
            "INSERT INTO regions (Region_Name) VALUES (%s)",
            [(str(x),) for x in regions["Region"]]
        )

        payment_modes = clean[["Payment_Mode"]].drop_duplicates()
        cursor.executemany(
            "INSERT INTO payment_modes (Payment_Mode_Name) VALUES (%s)",
            [(str(x),) for x in payment_modes["Payment_Mode"]]
        )

        products = clean[["Product", "Category"]].drop_duplicates()
        cursor.executemany(
            "INSERT INTO products (Product_Name, Category) VALUES (%s, %s)",
            [
                (str(row.Product), str(row.Category))
                for row in products.itertuples(index=False)
            ]
        )

        # One customer gets one region in this project schema.
        customers = (
            clean[["Customer_ID", "Region"]]
            .drop_duplicates()
            .sort_values(["Customer_ID", "Region"])
            .drop_duplicates(subset=["Customer_ID"])
        )

        region_map = {}
        cursor.execute("SELECT Region_ID, Region_Name FROM regions")
        for region_id, region_name in cursor.fetchall():
            region_map[str(region_name)] = region_id

        cursor.executemany(
            """
            INSERT INTO customers
                (Customer_ID, Customer_Name, Region_ID)
            VALUES (%s, %s, %s)
            """,
            [
                (
                    str(row.Customer_ID),
                    f"Customer {row.Customer_ID}",
                    region_map[str(row.Region)]
                )
                for row in customers.itertuples(index=False)
            ]
        )

        # Build lookup maps.
        cursor.execute("SELECT Region_ID, Region_Name FROM regions")
        region_map = {str(name): rid for rid, name in cursor.fetchall()}

        cursor.execute(
            "SELECT Payment_Mode_ID, Payment_Mode_Name FROM payment_modes"
        )
        payment_map = {
            str(name): pid for pid, name in cursor.fetchall()
        }

        cursor.execute(
            "SELECT Product_ID, Product_Name, Category FROM products"
        )
        product_map = {
            (str(name), str(category)): pid
            for pid, name, category in cursor.fetchall()
        }

        cursor.executemany(
            """
            INSERT INTO orders
                (Order_ID, Order_Date, Customer_ID, Region_ID, Payment_Mode_ID)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (
                    str(row.Order_ID),
                    row.Order_Date,
                    str(row.Customer_ID),
                    region_map[str(row.Region)],
                    payment_map[str(row.Payment_Mode)]
                )
                for row in clean.itertuples(index=False)
            ]
        )

        cursor.executemany(
            """
            INSERT INTO order_details
                (Order_ID, Product_ID, Quantity, Sales, Profit)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (
                    str(row.Order_ID),
                    product_map[(str(row.Product), str(row.Category))],
                    int(row.Quantity),
                    float(row.Sales),
                    float(row.Profit)
                )
                for row in clean.itertuples(index=False)
            ]
        )

        conn.commit()

    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


# =========================================================
# REQUIRED DATASET COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
    "Order_ID",
    "Order_Date",
    "Customer_ID",
    "Product",
    "Category",
    "Quantity",
    "Sales",
    "Profit",
    "Region",
    "Payment_Mode"
]


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD SALES DATA
# =========================================================

@st.cache_data(ttl=60)
def load_sales_data():

    # MySQL is the primary data source.
    # CSV is kept as a fallback so the dashboard can still open
    # if MySQL is temporarily unavailable.
    try:
        df = load_mysql_data()
        return df
    except Exception as mysql_error:
        try:
            df = pd.read_csv(DATA_FILE)
            df["Order_Date"] = pd.to_datetime(
                df["Order_Date"],
                errors="coerce"
            )
            st.warning(
                "MySQL could not be loaded, so the dashboard is using "
                f"the CSV fallback. Details: {mysql_error}"
            )
            return df
        except Exception as csv_error:
            raise RuntimeError(
                f"MySQL error: {mysql_error}; CSV error: {csv_error}"
            )


# =========================================================
# SAVE DATA
# =========================================================

def save_sales_data(df):

    df_to_save = df.copy()

    df_to_save["Order_Date"] = pd.to_datetime(
        df_to_save["Order_Date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Keep a CSV backup/export.
    df_to_save.to_csv(
        DATA_FILE,
        index=False
    )

    # Keep the normalized MySQL database synchronized.
    sync_dataframe_to_mysql(df_to_save)


# =========================================================
# VALIDATE DATA
# =========================================================

def validate_data(df):

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        return False, (
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:

        return False, "The dataset is empty."

    return True, ""


# =========================================================
# REBUILD ML OUTPUTS
# =========================================================

def rebuild_ml_outputs(df):

    df = df.copy()

    # -----------------------------------------------------
    # Basic cleaning
    # -----------------------------------------------------

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        errors="coerce"
    )

    for column in ["Quantity", "Sales", "Profit"]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Order_ID",
            "Order_Date",
            "Customer_ID",
            "Sales"
        ]
    )

    df = df.drop_duplicates(
        subset=["Order_ID"]
    )

    if len(df) == 0:

        raise ValueError(
            "No valid sales records remain after cleaning."
        )

    # =====================================================
    # 1. MONTHLY SALES
    # =====================================================

    monthly_df = (
        df.assign(
            Month=df["Order_Date"].dt.to_period("M").dt.to_timestamp()
        )
        .groupby("Month", as_index=False)["Sales"]
        .sum()
        .sort_values("Month")
        .reset_index(drop=True)
    )

    monthly_df.to_csv(
        MONTHLY_FILE,
        index=False
    )

    # =====================================================
    # 2. RFM CUSTOMER INTELLIGENCE
    # =====================================================

    reference_date = (
        df["Order_Date"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("Customer_ID")
        .agg(
            Recency=(
                "Order_Date",
                lambda x: (
                    reference_date - x.max()
                ).days
            ),
            Frequency=(
                "Order_ID",
                "nunique"
            ),
            Monetary=(
                "Sales",
                "sum"
            )
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # Customer segmentation
    # -----------------------------------------------------

    if len(rfm) >= 2:

        rfm_features = rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ].copy()

        scaler = StandardScaler()

        rfm_scaled = scaler.fit_transform(
            rfm_features
        )

        # The original notebook uses up to 4 clusters
        # when possible.

        n_clusters = min(
            4,
            len(rfm)
        )

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20
        )

        rfm["Cluster"] = kmeans.fit_predict(
            rfm_scaled
        )

        if n_clusters > 1:

            final_silhouette = silhouette_score(
                rfm_scaled,
                rfm["Cluster"]
            )

        else:

            final_silhouette = np.nan

        # -------------------------------------------------
        # Data-driven segment naming
        # -------------------------------------------------

        cluster_profile = (
            rfm.groupby("Cluster")[
                [
                    "Recency",
                    "Frequency",
                    "Monetary"
                ]
            ]
            .mean()
            .sort_index()
        )

        cluster_profile["ValueScore"] = (
            cluster_profile["Monetary"].rank(
                pct=True
            )
            +
            cluster_profile["Frequency"].rank(
                pct=True
            )
            +
            (
                1
                -
                cluster_profile["Recency"].rank(
                    pct=True
                )
            )
        )

        active_cluster = (
            cluster_profile["ValueScore"]
            .idxmax()
        )

        at_risk_cluster = (
            cluster_profile["Recency"]
            .idxmax()
        )

        remaining = [
            c
            for c in cluster_profile.index
            if c not in [
                active_cluster,
                at_risk_cluster
            ]
        ]

        if remaining:

            regular_cluster = max(
                remaining,
                key=lambda c:
                cluster_profile.loc[
                    c,
                    "Frequency"
                ]
            )

        else:

            regular_cluster = active_cluster

        occasional_candidates = [
            c
            for c in cluster_profile.index
            if c not in [
                active_cluster,
                at_risk_cluster,
                regular_cluster
            ]
        ]

        if occasional_candidates:

            occasional_cluster = (
                occasional_candidates[0]
            )

        else:

            occasional_cluster = (
                regular_cluster
            )

        segment_map = {
            active_cluster:
                "High-Value Customers",

            at_risk_cluster:
                "At-Risk Customers",

            regular_cluster:
                "Regular Customers",

            occasional_cluster:
                "Occasional Customers"
        }

        rfm["Customer_Segment"] = (
            rfm["Cluster"]
            .map(segment_map)
        )

    else:

        # Not enough customers for K-Means.

        rfm["Cluster"] = 0

        rfm["Customer_Segment"] = (
            "Available Customer"
        )

        final_silhouette = np.nan

    # -----------------------------------------------------
    # Customer segment summary
    # -----------------------------------------------------

    segment_summary = (
        rfm.groupby("Customer_Segment")
        .agg(
            Customers=(
                "Customer_ID",
                "count"
            ),
            Avg_Recency=(
                "Recency",
                "mean"
            ),
            Avg_Frequency=(
                "Frequency",
                "mean"
            ),
            Avg_Monetary=(
                "Monetary",
                "mean"
            )
        )
        .reset_index()
    )

    rfm.to_csv(
        SEGMENT_FILE,
        index=False
    )

    segment_summary.to_csv(
        SEGMENT_SUMMARY_FILE,
        index=False
    )

    # =====================================================
    # 3. SALES FORECASTING
    # =====================================================

    forecast_df = monthly_df.copy()

    forecast_df["Year"] = (
        forecast_df["Month"].dt.year
    )

    forecast_df["Month_Number"] = (
        forecast_df["Month"].dt.month
    )

    forecast_df["Time_Index"] = np.arange(
        len(forecast_df)
    )

    # -----------------------------------------------------
    # Lag features
    # -----------------------------------------------------

    for lag in [1, 2, 3]:

        forecast_df[
            f"Lag_{lag}"
        ] = forecast_df["Sales"].shift(lag)

    forecast_df["Rolling_Mean_3"] = (
        forecast_df["Sales"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    feature_columns = [
        "Year",
        "Month_Number",
        "Time_Index",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Rolling_Mean_3"
    ]

    model_df = (
        forecast_df
        .dropna()
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Need enough history for forecasting
    # -----------------------------------------------------

    if len(model_df) < 8:

        raise ValueError(
            "Not enough monthly sales history "
            "for forecasting. Please keep at least "
            "about 12 months of historical sales data."
        )

    # -----------------------------------------------------
    # Chronological train/test split
    # -----------------------------------------------------

    split_index = max(
        5,
        int(len(model_df) * 0.80)
    )

    if split_index >= len(model_df):

        split_index = (
            len(model_df) - 1
        )

    train = (
        model_df
        .iloc[:split_index]
        .copy()
    )

    test = (
        model_df
        .iloc[split_index:]
        .copy()
    )

    X_train = train[
        feature_columns
    ]

    y_train = train["Sales"]

    X_test = test[
        feature_columns
    ]

    y_test = test["Sales"]

    # -----------------------------------------------------
    # Random Forest model
    # -----------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    test["Predicted_Sales"] = (
        model.predict(X_test)
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        test["Predicted_Sales"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test["Predicted_Sales"]
        )
    )

    r2 = (
        r2_score(
            y_test,
            test["Predicted_Sales"]
        )
        if len(test) > 1
        else np.nan
    )

    test.to_csv(
        TEST_RESULTS_FILE,
        index=False
    )

    # =====================================================
    # 4. FINAL MODEL
    # =====================================================

    final_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(
        model_df[feature_columns],
        model_df["Sales"]
    )

    # =====================================================
    # 5. FUTURE 3-MONTH FORECAST
    # =====================================================

    forecast_horizon = 3

    history = monthly_df[
        [
            "Month",
            "Sales"
        ]
    ].copy()

    future_rows = []

    for step in range(
        forecast_horizon
    ):

        next_month = (
            history["Month"].max()
            + pd.offsets.MonthBegin(1)
        )

        lag_1 = (
            history["Sales"].iloc[-1]
        )

        lag_2 = (
            history["Sales"].iloc[-2]
            if len(history) >= 2
            else lag_1
        )

        lag_3 = (
            history["Sales"].iloc[-3]
            if len(history) >= 3
            else lag_2
        )

        rolling_mean_3 = np.mean(
            [
                lag_1,
                lag_2,
                lag_3
            ]
        )

        prediction_row = pd.DataFrame(
            [
                {
                    "Year":
                        next_month.year,

                    "Month_Number":
                        next_month.month,

                    "Time_Index":
                        len(history),

                    "Lag_1":
                        lag_1,

                    "Lag_2":
                        lag_2,

                    "Lag_3":
                        lag_3,

                    "Rolling_Mean_3":
                        rolling_mean_3
                }
            ]
        )

        prediction = float(
            final_model.predict(
                prediction_row[
                    feature_columns
                ]
            )[0]
        )

        prediction = max(
            0,
            prediction
        )

        future_rows.append(
            {
                "Month":
                    next_month,

                "Predicted_Sales":
                    prediction
            }
        )

        history = pd.concat(
            [
                history,

                pd.DataFrame(
                    [
                        {
                            "Month":
                                next_month,

                            "Sales":
                                prediction
                        }
                    ]
                )
            ],
            ignore_index=True
        )

    future_forecast = pd.DataFrame(
        future_rows
    )

    future_forecast.to_csv(
        FORECAST_FILE,
        index=False
    )

    # =====================================================
    # 6. MODEL METRICS
    # =====================================================

    metrics = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
                "R2",
                "Silhouette Score"
            ],

            "Value": [
                mae,
                rmse,
                r2,
                final_silhouette
            ]
        }
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False
    )

    return {
        "monthly": monthly_df,
        "segments": rfm,
        "segment_summary":
            segment_summary,
        "forecast":
            future_forecast,
        "metrics":
            metrics
    }


# =========================================================
# DATA SOURCE
# =========================================================
# MySQL is the primary source. The CSV file remains available
# as a local backup/fallback for the project.
# =========================================================


# =========================================================
# LOAD CURRENT DATA
# =========================================================

try:

    sales = load_sales_data()

except Exception as e:

    st.error(
        f"Could not load sales_data.csv: {e}"
    )

    st.stop()


valid, message = validate_data(
    sales
)

if not valid:

    st.error(message)

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Sales Analysis",
        "Manage Sales Data",
        "Sales Forecast",
        "Customer Intelligence",
        "About Project"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header("?? Business Dashboard")

    sales_column = "Sales"
    customer_column = "Customer_ID"

    total_sales = sales[
        sales_column
    ].sum()

    total_customers = sales[
        customer_column
    ].nunique()

    total_transactions = len(
        sales
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "?? Total Sales",
        f"{total_sales:,.2f}"
    )

    col2.metric(
        "?? Customers",
        f"{total_customers:,}"
    )

    col3.metric(
        "?? Transactions",
        f"{total_transactions:,}"
    )

    st.divider()

    # -----------------------------------------------------
    # Product Sales
    # -----------------------------------------------------

    st.subheader(
        "?? Sales by Product"
    )

    product_sales = (
        sales.groupby("Product")[
            "Sales"
        ]
        .sum()
        .reset_index()
        .sort_values(
            "Sales",
            ascending=False
        )
    )

    fig_product = px.bar(
        product_sales,
        x="Product",
        y="Sales",
        title="Sales by Product"
    )

    st.plotly_chart(
        fig_product,
        width="stretch"
    )

    # -----------------------------------------------------
    # Monthly Sales
    # -----------------------------------------------------

    monthly = (
        sales.assign(
            Month=
            sales["Order_Date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        .groupby(
            "Month",
            as_index=False
        )["Sales"]
        .sum()
        .sort_values("Month")
    )

    st.subheader(
        "?? Monthly Sales Trend"
    )

    fig_monthly = px.line(
        monthly,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(
        fig_monthly,
        width="stretch"
    )


# =========================================================
# SALES ANALYSIS
# =========================================================

elif page == "Sales Analysis":

    st.header(
        "?? Sales Analysis"
    )

    st.subheader(
        "Sales Data"
    )

    st.dataframe(
        sales,
        width="stretch"
    )

    # -----------------------------------------------------
    # Sales by Category
    # -----------------------------------------------------

    st.subheader(
        "Sales by Category"
    )

    category_sales = (
        sales.groupby("Category")[
            "Sales"
        ]
        .sum()
        .reset_index()
    )

    fig_category = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        title="Sales by Category"
    )

    st.plotly_chart(
        fig_category,
        width="stretch"
    )

    # -----------------------------------------------------
    # Sales by Region
    # -----------------------------------------------------

    st.subheader(
        "Sales by Region"
    )

    region_sales = (
        sales.groupby("Region")[
            "Sales"
        ]
        .sum()
        .reset_index()
        .sort_values(
            "Sales",
            ascending=False
        )
    )

    fig_region = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        title="Sales by Region"
    )

    st.plotly_chart(
        fig_region,
        width="stretch"
    )


# =========================================================
# MANAGE SALES DATA
# =========================================================

elif page == "Manage Sales Data":

    st.header(
        " Manage Sales Data"
    )

    st.write(
        "Insert, update or delete sales records. "
        "After every successful change, the ML outputs "
        "are automatically rebuilt."
    )

    st.warning(
        "Changing sales data can change customer segments "
        "and sales forecasts because both are calculated "
        "from the current dataset."
    )

    # =====================================================
    # INSERT
    # =====================================================

    st.subheader(
        "? Insert New Sale"
    )

    with st.form(
        "insert_sale_form",
        clear_on_submit=True
    ):

        col1, col2 = st.columns(2)

        with col1:

            new_order_id = st.text_input(
                "Order ID"
            )

            new_order_date = st.date_input(
                "Order Date"
            )

            new_customer_id = st.text_input(
                "Customer ID"
            )

            new_product = st.text_input(
                "Product"
            )

            new_category = st.text_input(
                "Category"
            )

        with col2:

            new_quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1,
                step=1
            )

            new_sales = st.number_input(
                "Sales",
                min_value=0.0,
                value=0.0,
                step=100.0
            )

            new_profit = st.number_input(
                "Profit",
                value=0.0,
                step=100.0
            )

            new_region = st.text_input(
                "Region"
            )

            new_payment_mode = st.text_input(
                "Payment Mode"
            )

        insert_button = st.form_submit_button(
            "? Add Sale"
        )

    if insert_button:

        if not new_order_id.strip():

            st.error(
                "Please enter an Order ID."
            )

        elif not new_customer_id.strip():

            st.error(
                "Please enter a Customer ID."
            )

        elif not new_product.strip():

            st.error(
                "Please enter a Product."
            )

        elif new_order_id in sales[
            "Order_ID"
        ].astype(str).values:

            st.error(
                "This Order ID already exists."
            )

        else:

            new_row = pd.DataFrame(
                [
                    {
                        "Order_ID":
                            new_order_id,

                        "Order_Date":
                            pd.to_datetime(
                                new_order_date
                            ),

                        "Customer_ID":
                            new_customer_id,

                        "Product":
                            new_product,

                        "Category":
                            new_category,

                        "Quantity":
                            new_quantity,

                        "Sales":
                            new_sales,

                        "Profit":
                            new_profit,

                        "Region":
                            new_region,

                        "Payment_Mode":
                            new_payment_mode
                    }
                ]
            )

            updated_sales = pd.concat(
                [
                    sales,
                    new_row
                ],
                ignore_index=True
            )

            try:

                # Save new data
                save_sales_data(
                    updated_sales
                )

                # Rebuild ML outputs
                rebuild_ml_outputs(
                    updated_sales
                )

                # Clear Streamlit cache
                st.cache_data.clear()

                st.success(
                    "Sale added successfully! "
                    "Sales analysis, customer segmentation "
                    "and forecast have been updated."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Could not add the sale: {e}"
                )

    st.divider()

    # =====================================================
    # UPDATE
    # =====================================================

    st.subheader(
        "?? Update Existing Sale"
    )

    order_ids = (
        sales["Order_ID"]
        .astype(str)
        .tolist()
    )

    selected_order_id = st.selectbox(
        "Select Order ID to update",
        order_ids,
        key="update_order"
    )

    selected_row = sales[
        sales["Order_ID"].astype(str)
        == selected_order_id
    ].iloc[0]

    with st.form(
        "update_sale_form"
    ):

        col1, col2 = st.columns(2)

        with col1:

            update_order_date = st.date_input(
                "Order Date",
                value=pd.to_datetime(
                    selected_row["Order_Date"]
                ).date()
            )

            update_customer_id = st.text_input(
                "Customer ID",
                value=str(
                    selected_row["Customer_ID"]
                )
            )

            update_product = st.text_input(
                "Product",
                value=str(
                    selected_row["Product"]
                )
            )

            update_category = st.text_input(
                "Category",
                value=str(
                    selected_row["Category"]
                )
            )

        with col2:

            update_quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=int(
                    selected_row["Quantity"]
                ),
                step=1
            )

            update_sales = st.number_input(
                "Sales",
                min_value=0.0,
                value=float(
                    selected_row["Sales"]
                ),
                step=100.0
            )

            update_profit = st.number_input(
                "Profit",
                value=float(
                    selected_row["Profit"]
                ),
                step=100.0
            )

            update_region = st.text_input(
                "Region",
                value=str(
                    selected_row["Region"]
                )
            )

            update_payment_mode = st.text_input(
                "Payment Mode",
                value=str(
                    selected_row["Payment_Mode"]
                )
            )

        update_button = st.form_submit_button(
            "?? Update Sale"
        )

    if update_button:

        updated_sales = sales.copy()

        mask = (
            updated_sales[
                "Order_ID"
            ].astype(str)
            == selected_order_id
        )

        updated_sales.loc[
            mask,
            "Order_Date"
        ] = pd.to_datetime(
            update_order_date
        )

        updated_sales.loc[
            mask,
            "Customer_ID"
        ] = update_customer_id

        updated_sales.loc[
            mask,
            "Product"
        ] = update_product

        updated_sales.loc[
            mask,
            "Category"
        ] = update_category

        updated_sales.loc[
            mask,
            "Quantity"
        ] = update_quantity

        updated_sales.loc[
            mask,
            "Sales"
        ] = update_sales

        updated_sales.loc[
            mask,
            "Profit"
        ] = update_profit

        updated_sales.loc[
            mask,
            "Region"
        ] = update_region

        updated_sales.loc[
            mask,
            "Payment_Mode"
        ] = update_payment_mode

        try:

            save_sales_data(
                updated_sales
            )

            rebuild_ml_outputs(
                updated_sales
            )

            st.cache_data.clear()

            st.success(
                "Sale updated successfully! "
                "All dependent analysis and ML outputs "
                "have been recalculated."
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Could not update the sale: {e}"
            )

    st.divider()

    # =====================================================
    # DELETE
    # =====================================================

    st.subheader(
        " Delete Existing Sale"
    )

    delete_order_id = st.selectbox(
        "Select Order ID to delete",
        order_ids,
        key="delete_order"
    )

    st.warning(
        f"You are about to delete Order ID: "
        f"{delete_order_id}"
    )

    if st.button(
        " Delete Sale",
        type="primary"
    ):

        updated_sales = sales[
            sales["Order_ID"].astype(str)
            != delete_order_id
        ].copy()

        if len(updated_sales) == len(
            sales
        ):

            st.error(
                "The selected order could not be found."
            )

        else:

            try:

                save_sales_data(
                    updated_sales
                )

                rebuild_ml_outputs(
                    updated_sales
                )

                st.cache_data.clear()

                st.success(
                    "Sale deleted successfully! "
                    "All dependent analysis and ML outputs "
                    "have been recalculated."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Could not delete the sale: {e}"
                )

    st.divider()

    st.subheader(
        "?? Current Sales Data"
    )

    st.dataframe(
        sales,
        width="stretch"
    )


# =========================================================
# SALES FORECAST
# =========================================================

elif page == "Sales Forecast":

    st.header(
        "?? Sales Forecast"
    )

    # -----------------------------------------------------
    # Recalculate button
    # -----------------------------------------------------

    if st.button(
        "?? Recalculate Forecast & Customer Intelligence"
    ):

        try:

            with st.spinner(
                "Rebuilding Machine Learning outputs..."
            ):

                results = rebuild_ml_outputs(
                    sales
                )

            st.success(
                "Forecast and customer intelligence "
                "have been recalculated successfully."
            )

            st.cache_data.clear()

            st.rerun()

        except Exception as e:

            st.error(
                f"Could not rebuild ML outputs: {e}"
            )

    # -----------------------------------------------------
    # Load forecast
    # -----------------------------------------------------

    if os.path.exists(
        FORECAST_FILE
    ):

        forecast = pd.read_csv(
            FORECAST_FILE
        )

        forecast["Month"] = pd.to_datetime(
            forecast["Month"],
            errors="coerce"
        )

        st.subheader(
            "Future Sales Forecast"
        )

        st.dataframe(
            forecast,
            width="stretch"
        )

        if (
            "Month" in forecast.columns
            and
            "Predicted_Sales"
            in forecast.columns
        ):

            fig = px.line(
                forecast,
                x="Month",
                y="Predicted_Sales",
                markers=True,
                title="Future Sales Forecast"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        # -------------------------------------------------
        # Model metrics
        # -------------------------------------------------

        if os.path.exists(
            METRICS_FILE
        ):

            metrics = pd.read_csv(
                METRICS_FILE
            )

            st.subheader(
                "?? Model Evaluation"
            )

            st.dataframe(
                metrics,
                width="stretch"
            )

    else:

        st.warning(
            "Forecast output is not available."
        )

        st.info(
            "Click 'Recalculate Forecast & "
            "Customer Intelligence'."
        )


# =========================================================
# CUSTOMER INTELLIGENCE
# =========================================================

elif page == "Customer Intelligence":

    st.header(
        "?? Customer Intelligence"
    )

    # -----------------------------------------------------
    # Recalculate
    # -----------------------------------------------------

    if st.button(
        "?? Recalculate Customer Segments"
    ):

        try:

            with st.spinner(
                "Recalculating RFM and K-Means segmentation..."
            ):

                rebuild_ml_outputs(
                    sales
                )

            st.success(
                "Customer segmentation updated."
            )

            st.cache_data.clear()

            st.rerun()

        except Exception as e:

            st.error(
                f"Could not calculate customer segments: {e}"
            )

    # -----------------------------------------------------
    # Load segments
    # -----------------------------------------------------

    if os.path.exists(
        SEGMENT_FILE
    ):

        segments = pd.read_csv(
            SEGMENT_FILE
        )

        st.subheader(
            "Customer Segmentation"
        )

        st.dataframe(
            segments,
            width="stretch"
        )

        if (
            "Customer_Segment"
            in segments.columns
        ):

            segment_counts = (
                segments[
                    "Customer_Segment"
                ]
                .value_counts()
                .reset_index()
            )

            segment_counts.columns = [
                "Segment",
                "Customers"
            ]

            st.subheader(
                "Customer Segment Distribution"
            )

            fig = px.bar(
                segment_counts,
                x="Segment",
                y="Customers",
                title="Customers by Segment"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            fig2 = px.pie(
                segment_counts,
                names="Segment",
                values="Customers",
                title="Customer Segment Share"
            )

            st.plotly_chart(
                fig2,
                width="stretch"
            )

        if os.path.exists(
            SEGMENT_SUMMARY_FILE
        ):

            summary = pd.read_csv(
                SEGMENT_SUMMARY_FILE
            )

            st.subheader(
                "Customer Segment Summary"
            )

            st.dataframe(
                summary,
                width="stretch"
            )

    else:

        st.warning(
            "Customer segmentation output is not available."
        )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.header(
        "?? About the Project"
    )

    st.write(
        """
        ### AI-Based Sales Forecasting and Customer Intelligence System

        This project uses Machine Learning to help businesses
        understand historical sales, forecast future sales,
        and identify different types of customers.

        ### Main Features

        - Insert sales records
        - Update sales records
        - Delete sales records
        - Automatic data recalculation
        - Sales KPIs
        - Sales analysis
        - Monthly sales trends
        - Random Forest sales forecasting
        - RFM customer intelligence
        - K-Means customer segmentation
        - Interactive visualizations

        ### Machine Learning Components

        **Sales Forecasting**

        Random Forest Regression uses historical monthly sales,
        lag features, rolling averages and calendar features
        to forecast future sales.

        **Customer Intelligence**

        RFM analysis calculates:

        - Recency
        - Frequency
        - Monetary Value

        K-Means clustering then groups customers according
        to their purchasing behaviour.

        ### Dynamic Data Management

        When a user inserts, updates or deletes a sales record,
        the system saves the new data and recalculates the
        dependent sales analysis, customer segmentation and
        sales forecast.

        This means the dashboard is based on the current
        sales dataset rather than permanently displaying
        the original results.
        """
    )


