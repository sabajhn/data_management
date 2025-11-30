"""
Module: app.py
Description: The main entry point for the Streamlit Web Application.
             It orchestrates Data Processing, Analysis, ML, and Visualization.
"""
import os
import streamlit as st
import ast # Required for list parsing
import pandas as pd
from MovieDataProcessor.data_cleaner import MovieDataProcessor, CSV_PATH
from MovieAnalyzer.analysis import MovieAnalyzer
from MovieVisualizer.visualize import MovieVisualizer 

from models.model_regression import RevenueRegressor
from models.model_classification import SuccessClassifier


st.set_page_config(page_title="CineMetrics: Blockbuster Analytics", page_icon="🎬", layout="wide")

st.title("🎬 CineMetrics: The Blockbuster Blueprint")
st.markdown("Analyze **45,000+ movies** with Data Mining and **Advanced Machine Learning**.")

# --- 1. Data Loading ---
@st.cache_data(show_spinner="Loading data...")
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    processor = MovieDataProcessor(path=CSV_PATH)
    df_raw = processor.load_data()
    df_clean = processor.preprocess_data(df_raw)
    
    # CRITICAL FIX for Streamlit caching (TypeError: unhashable type: 'list'/'dict'):
    # Convert unhashable types (lists, dicts) to strings to keep the data safe for caching.
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                pd.util.hash_pandas_object(df_clean[[col]])
            except TypeError:
                df_clean[col] = df_clean[col].astype(str)
                
    return df_clean

# --- 2. ML Training (Cached) ---
@st.cache_resource(show_spinner="Training Advanced ML Models (Gradient Boosting)...")
def train_models(df):
    # Train Regressor
    regressor = RevenueRegressor(df)
    reg_metrics = regressor.train()
    
    # Train Classifier
    classifier = SuccessClassifier(df)
    clf_metrics = classifier.train()
    
    return regressor, classifier, reg_metrics, clf_metrics

# --- Execution ---
df = load_data()
if df is None:
    st.error(f"❌ Dataset not found at `{CSV_PATH}`.")
    st.info("Please create a 'data' folder and put 'movies_metadata.csv' inside it.")
    st.stop()

# Helper to restore list structures for the advanced visualizations
def restore_list_cols(df_in):
    """
    Safely parses stringified lists back into Python lists of STRINGS.
    Ensures "[{'name': 'Action'}]" becomes ['Action'].
    """
    df_out = df_in.copy()
    
    def parse_genres(x):
        try:
            # 1. If it's a string representation, evaluate it to a Python object
            if isinstance(x, str):
                try:
                    x = ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []
            
            # 2. If it's a list, check contents
            if isinstance(x, list):
                if not x: return []
                
                # Case A: List of Dicts (from raw JSON) -> Extract 'name'
                if isinstance(x[0], dict):
                    return [d.get('name') for d in x if isinstance(d, dict) and 'name' in d]
                
                # Case B: List of Strings (already processed) -> Return as is
                if isinstance(x[0], str):
                    return x
            
            return []
        except:
            return []

    if 'genres_list' in df_out.columns:
        df_out['genres_list'] = df_out['genres_list'].apply(parse_genres)
        
    return df_out

# Create a visualization-ready dataframe (with lists restored)
df_viz = restore_list_cols(df)

# Train Models
regressor, classifier, reg_metrics, clf_metrics = train_models(df)
st.sidebar.success(f"ML Ready. Regressor R²: {reg_metrics['r2']:.2f} | Classifier Acc: {clf_metrics['accuracy']:.0%}")

# Initialize Classes
analyzer = MovieAnalyzer(df)
visualizer = MovieVisualizer()

# --- Navigation ---
st.sidebar.header("Navigation")
options = st.sidebar.radio("Select Module:", 
    ["Overview", "Financial Analysis", "Genre & Studio", "Data Mining", "💰 Revenue Predictor", "🏆 Success Classifier"])

# Shared lists for UI inputs
genres_list = sorted(df['primary_genre'].dropna().unique())
langs_list = sorted(df['original_language'].dropna().unique())
en_index = langs_list.index('en') if 'en' in langs_list else 0
months_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
studios_list = sorted(df['lead_studio'].astype(str).unique())

# --- MODULE 1: OVERVIEW ---
if options == "Overview":
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{len(df):,}")
    col2.metric("Avg Budget", f"${df['budget'].mean():,.0f}")
    col3.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}")
    col4.metric("Avg ROI", f"{df['roi'].median():.2f}x")
    
    st.subheader("Raw Data Sample")
    # Filter out technical/messy columns for a cleaner display
    cols_to_hide = ['genres', 'production_companies', 'genres_list', 'companies_list', 'belongs_to_collection']
    display_cols = [c for c in df.columns if c not in cols_to_hide]
    
    st.dataframe(df[display_cols].head(10))
    
    st.markdown("### Key Correlations")
    st.dataframe(analyzer.get_correlation_matrix().style.background_gradient(cmap="coolwarm"))

# --- MODULE 2: FINANCIAL ANALYSIS ---
elif options == "Financial Analysis":
    st.header("💰 Financial Analysis")
    
    tab1, tab2 = st.tabs(["Budget vs. Revenue", "Historical Context"])
    
    with tab1:
        st.subheader("Correlation Analysis")
        st.markdown("Does throwing money at a movie guarantee success?")
        st.pyplot(visualizer.plot_budget_vs_revenue(df))
        
    with tab2:
        st.subheader("Historical Trends")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### Budget Growth Over Time")
            st.pyplot(visualizer.plot_yearly_trends(analyzer.get_yearly_trends()))
        with c2:
            st.markdown("#### Movie Releases by Decade")
            st.pyplot(visualizer.plot_decade_pie(df_viz))

# --- MODULE 3: GENRE & STUDIO ---
elif options == "Genre & Studio":
    st.header("🎭 Genre & Studio Analytics")
    
    # Word Cloud removed to ensure stability
    
    tab1, tab2 = st.tabs(["Profitability & Leaders", "Deep Dive: Distributions"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1: 
            st.subheader("Most Profitable Genres")
            st.pyplot(visualizer.plot_genre_roi(analyzer.get_genre_metrics()))
        with c2: 
            st.subheader("Top Studios (Revenue)")
            st.pyplot(visualizer.plot_top_studios(analyzer.get_top_studios()))
            
    with tab2:
        st.markdown("### Understanding Data Distributions")
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Popularity by Genre")
            st.markdown("*Boxen plots show the spread of popularity scores, revealing outliers.*")
            st.pyplot(visualizer.plot_genre_popularity_boxen(df_viz))
        with c4:
            st.subheader("Runtime by Genre")
            st.markdown("*Violin plots show the density of runtimes (fat sections = common lengths).*")
            st.pyplot(visualizer.plot_genre_runtime_violin(df_viz))

# --- MODULE 4: DATA MINING ---
elif options == "Data Mining":
    st.header("⛏️ Data Mining Deep Dive")
    
    c1, c2 = st.columns(2)
    with c1: 
        st.subheader("Seasonality Analysis")
        st.markdown("Is there a specific month where movies make the most money?")
        st.pyplot(visualizer.plot_seasonal_revenue(analyzer.get_seasonal_stats()))
    with c2:
        st.subheader("Popularity vs. Quality")
        st.markdown("Do popular movies actually get better ratings?")
        st.pyplot(visualizer.plot_popularity_vs_rating(df))
        
    st.subheader("The 'Sweet Spot': Runtime Analysis")
    st.table(analyzer.get_runtime_metrics().head())

# --- MODULE 5: ML REGRESSION ---
elif options == "💰 Revenue Predictor":
    st.header("💰 AI Revenue Prediction")
    st.markdown("Predict the **exact dollar amount** a movie might earn.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        b_in = st.number_input("Budget ($)", 1000, 500000000, 10000000)
        r_in = st.slider("Runtime (mins)", 30, 240, 120)
        p_in = st.slider("Popularity Score", 1.0, 50.0, 10.0)
        v_in = st.number_input("Vote Count (Hype)", 0, 15000, 1000)
        
        is_fran = st.checkbox("Is part of a Franchise?", value=False)
        is_fran_int = 1 if is_fran else 0
        
        g_in = st.selectbox("Genre", genres_list)
        l_in = st.selectbox("Language", langs_list, index=en_index)
        m_in = st.selectbox("Release Month", months_list)
        s_in = st.selectbox("Lead Studio", studios_list)

        if st.button("Predict Revenue"):
            rev = regressor.predict(b_in, r_in, p_in, is_fran_int, v_in, g_in, l_in, m_in, s_in)
            st.success(f"Predicted Revenue: **${rev:,.2f}**")
            if is_fran: st.info("💡 Franchise movies have a significant multiplier effect.")

    with c2:
        st.subheader("Feature Importance")
        st.pyplot(visualizer.plot_feature_importance(regressor.get_feature_importance()))

# --- MODULE 6: ML CLASSIFICATION ---
elif options == "🏆 Success Classifier":
    st.header("🏆 AI Success Classification")
    st.markdown(f"Predict if a movie will be a **Critical Hit** (Rated > {classifier.rating_threshold:.1f}).")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info("Enter details:")
        b_cl = st.number_input("Budget ($)", 1000, 500000000, 5000000, key='b_clf')
        r_cl = st.slider("Runtime (mins)", 30, 240, 100, key='r_clf')
        p_cl = st.slider("Popularity", 1.0, 50.0, 5.0, key='p_clf')
        v_cl = st.number_input("Vote Count", 0, 15000, 500, key='v_clf')
        
        is_fran_cl = st.checkbox("Is Franchise?", value=False, key='fran_clf')
        is_fran_int_cl = 1 if is_fran_cl else 0
        
        g_cl = st.selectbox("Genre", genres_list, key='g_clf')
        l_cl = st.selectbox("Language", langs_list, index=en_index, key='l_clf')
        m_cl = st.selectbox("Release Month", months_list, key='m_clf')
        s_cl = st.selectbox("Lead Studio", studios_list, key='s_clf')
        
        if st.button("Classify Success"):
            prob = classifier.predict_proba(b_cl, r_cl, p_cl, is_fran_int_cl, v_cl, g_cl, l_cl, m_cl, s_cl)
            if prob > 0.5:
                st.balloons()
                st.success(f"🌟 **CRITICAL HIT!** ({prob:.1%} confidence)")
            else:
                st.warning(f"📉 **Average/Flop** ({1-prob:.1%} confidence)")
    
    with c2:
        st.markdown("### Model Performance")
        st.pyplot(visualizer.plot_confusion_matrix(clf_metrics['confusion_matrix'], classifier.rating_threshold))