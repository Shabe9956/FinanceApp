import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Financial ML App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
    }
    .stInfo {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = None

# Helper function to download data
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="financial_data.csv">Download CSV File</a>'
    return href

# Welcome Interface
st.title("📈 Financial Machine Learning Application")
st.markdown("""
Welcome to this interactive financial machine learning application. 
This tool allows you to analyze financial data using machine learning models.
""")

# Add finance-themed GIF
st.image("data:image/jpeg;base64,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", width=300)

# Sidebar for data loading
with st.sidebar:
    st.header("Data Loading Options")
    data_source = st.radio("Select data source:", 
                          ("Upload Kragle Dataset", "Fetch from Yahoo Finance"))
    
    if data_source == "Upload Kragle Dataset":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.processed = False
                st.session_state.ticker = None
                st.success("Kragle dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    elif data_source == "Fetch from Yahoo Finance":
        ticker = st.text_input("Enter stock ticker (e.g., AAPL):", "AAPL")
        start_date = st.date_input("Start date:", datetime(2020, 1, 1))
        end_date = st.date_input("End date:", datetime.today())
        
        if st.button("Fetch Data"):
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                if not data.empty:
                    # Flatten MultiIndex columns
                    data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns.values]
                    data.reset_index(inplace=True)
                    
                    # Calculate returns
                    close_col = f"Close_{ticker}" if f"Close_{ticker}" in data.columns else "Close"
                    data['Return'] = data[close_col].pct_change()
                    
                    st.session_state.df = data
                    st.session_state.processed = False
                    st.session_state.ticker = ticker
                    st.success(f"Successfully fetched {ticker} data from Yahoo Finance!")
                else:
                    st.error("No data found for this ticker and date range.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")

# Main content area
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Preview", "Preprocessing", "Feature Engineering", 
    "Train/Test Split", "Model Training", "Evaluation"
])

with tab1:
    st.header("Data Preview")
    if st.session_state.df is not None:
        st.write("First 10 rows of the dataset:")
        st.dataframe(st.session_state.df.head(10))
        
        st.write("Dataset summary statistics:")
        st.dataframe(st.session_state.df.describe())
        
        # Download button
        st.markdown(get_table_download_link(st.session_state.df), unsafe_allow_html=True)
    else:
        st.warning("Please load data first using the sidebar options.")

with tab2:
    st.header("Data Preprocessing")
    if st.session_state.df is not None:
        if st.button("Start Preprocessing"):
            df = st.session_state.df.copy()
            
            # Handle missing values
            missing_values = df.isnull().sum()
            st.write("Missing values before processing:")
            st.write(missing_values)
            
            # Fill missing values
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            
            st.write("Missing values after processing:")
            st.write(df.isnull().sum())
            
            # Remove potential outliers in returns
            if 'Return' in df.columns:
                df = df[(df['Return'] > -0.5) & (df['Return'] < 0.5)]
            
            st.session_state.df = df
            st.session_state.processed = True
            st.success("Data preprocessing completed!")
            
            # Visualize the cleaned data
            close_col = f"Close_{st.session_state.ticker}" if st.session_state.ticker and f"Close_{st.session_state.ticker}" in df.columns else "Close"
            if close_col in df.columns and 'Date' in df.columns:
                fig = px.line(df, x='Date', y=close_col, 
                              title=f"{st.session_state.ticker if st.session_state.ticker else 'Dataset'} Closing Prices")
                st.plotly_chart(fig)
    else:
        st.warning("Please load data first using the sidebar options.")

with tab3:
    st.header("Feature Engineering")
    if st.session_state.df is not None and st.session_state.processed:
        df = st.session_state.df.copy()
        
        st.write("Available columns:")
        st.write(list(df.columns))
        
        # Create common financial features
        close_col = f"Close_{st.session_state.ticker}" if st.session_state.ticker and f"Close_{st.session_state.ticker}" in df.columns else "Close"
        
        if close_col in df.columns:
            df['MA_7'] = df[close_col].rolling(window=7).mean().fillna(method='bfill')
            df['MA_30'] = df[close_col].rolling(window=30).mean().fillna(method='bfill')
            
            if 'Return' in df.columns:
                df['Volatility'] = df['Return'].rolling(window=30).std().fillna(method='bfill')
                df['Lag1_Return'] = df['Return'].shift(1).fillna(0)
            
            df.dropna(inplace=True)
            
            st.write("New features created:")
            st.write(df[['MA_7', 'MA_30', 'Volatility', 'Lag1_Return']].head())
            
            # Feature selection
            feature_options = ['MA_7', 'MA_30', 'Volatility', 'Lag1_Return']
            selected_features = st.multiselect(
                "Select features for modeling:", 
                feature_options,
                default=feature_options
            )
            
            target = st.selectbox(
                "Select target variable:",
                ['Return', close_col]
            )
            
            if st.button("Confirm Features"):
                st.session_state.features = selected_features
                st.session_state.target = target
                st.session_state.df = df
                st.success("Features selected successfully!")
                
                # Feature importance visualization
                if target == 'Return':
                    corr_matrix = df[selected_features + ['Return']].corr()
                    fig = px.imshow(corr_matrix, text_auto=True)
                    st.plotly_chart(fig)
    else:
        st.warning("Please complete data loading and preprocessing first.")

with tab4:
    st.header("Train/Test Split")
    if st.session_state.features is not None:
        df = st.session_state.df.copy()
        X = df[st.session_state.features]
        y = df[st.session_state.target]
        
        test_size = st.slider("Select test size ratio:", 0.1, 0.5, 0.3)
        
        if st.button("Split Data"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success("Data split successfully!")
            
            # Visualize the split
            sizes = [len(X_train), len(X_test)]
            labels = ['Train', 'Test']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
            ax1.set_title('Train/Test Split')
            
            ax2.bar(labels, sizes, color=['green', 'blue'])
            ax2.set_title('Number of Samples')
            ax2.set_ylabel('Count')
            
            st.pyplot(fig)
    else:
        st.warning("Please complete feature engineering first.")

with tab5:
    st.header("Model Training")
    if st.session_state.X_train is not None:
        st.write("Training Linear Regression model...")
        
        if st.button("Train Model"):
            model = LinearRegression()
            model.fit(st.session_state.X_train, st.session_state.y_train)
            
            st.session_state.model = model
            st.success("Model trained successfully!")
            
            # Show model coefficients
            coeff_df = pd.DataFrame({
                'Feature': st.session_state.features,
                'Coefficient': model.coef_
            })
            
            st.write("Model coefficients:")
            st.dataframe(coeff_df)
            
            # Visualize coefficients
            fig = px.bar(coeff_df, x='Feature', y='Coefficient', 
                         title='Feature Coefficients')
            st.plotly_chart(fig)
    else:
        st.warning("Please complete train/test split first.")

with tab6:
    st.header("Model Evaluation")
    if st.session_state.model is not None:
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        
        mse = mean_squared_error(st.session_state.y_test, y_pred)
        r2 = r2_score(st.session_state.y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.6f}")
        with col2:
            st.metric("R² Score", f"{r2:.4f}")
        
        # Actual vs Predicted plot
        fig1 = px.scatter(
            x=st.session_state.y_test,
            y=y_pred,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title="Actual vs Predicted Values"
        )
        fig1.add_shape(
            type="line", line=dict(dash='dash'),
            x0=min(st.session_state.y_test),
            y0=min(st.session_state.y_test),
            x1=max(st.session_state.y_test),
            y1=max(st.session_state.y_test)
        )
        st.plotly_chart(fig1)
        
        # Time series plot for time-based data
        if 'Date' in st.session_state.df.columns:
            test_indices = st.session_state.X_test.index
            plot_df = st.session_state.df.loc[test_indices]
            
            fig2 = px.line(
                x=plot_df['Date'],
                y=st.session_state.y_test,
                title="Actual vs Predicted Over Time",
                labels={'x': 'Date', 'y': st.session_state.target}
            )
            fig2.add_scatter(
                x=plot_df['Date'],
                y=y_pred,
                name='Predicted',
                line=dict(color='red')
            )
            st.plotly_chart(fig2)
    else:
        st.warning("Please train the model first.")

# Footer
st.markdown("---")
st.markdown("""
### How to Use This App:
1. Load data using the sidebar (either upload or fetch from Yahoo Finance)
2. Proceed through the tabs from left to right
3. At each step, click the appropriate buttons to execute the workflow
4. View results and visualizations at each stage
""")