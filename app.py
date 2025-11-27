"""
Module: app.py
Description: The main entry point for the Streamlit Web Application.
             It orchestrates Data Processing, Analysis, ML, and Visualization.
"""
import os
import streamlit as st
import pandas as pd
from MovieDataProcessor.data_cleaner import MovieDataProcessor, CSV_PATH
from MovieAnalyzer.analysis import MovieAnalyzer
from MovieVisualizer.visualize import MovieVisualizer 

from models.model_regression import RevenueRegressor
from models.model_classification import SuccessClassifier

st.set_page_config(page_title="CineMetrics: Blockbuster Analytics", page_icon="🎬", layout="wide")

st.title("🎬 CineMetrics: The Blockbuster Blueprint")
st.markdown("Analyze **45,000+ movies** with Data Mining and **High-Accuracy Machine Learning**.")

# --- 1. Data Loading ---
@st.cache_data(show_spinner="Loading data...")
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    processor = MovieDataProcessor(path=CSV_PATH)
    df_raw = processor.load_data()
    df_clean = processor.preprocess_data(df_raw)
    
    # Drop unhashable list columns for Streamlit caching
    cols_to_drop = [c for c in df_clean.columns if df_clean[c].apply(lambda x: isinstance(x, list)).any()]
    return df_clean.drop(columns=cols_to_drop, errors='ignore')

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

regressor, classifier, reg_metrics, clf_metrics = train_models(df)
st.sidebar.success(f"ML Ready. Regressor R²: {reg_metrics['r2']:.2f} | Classifier Acc: {clf_metrics['accuracy']:.0%}")

analyzer = MovieAnalyzer(df)
visualizer = MovieVisualizer()

# --- Navigation ---
st.sidebar.header("Navigation")
options = st.sidebar.radio("Select Module:", 
    ["Overview", "Financial Analysis", "Genre & Studio", "Data Mining", "Revenue Predictor (Reg)", "Success Classifier (Clf)"])

# Shared lists for UI
genres_list = sorted(df['primary_genre'].dropna().unique())
langs_list = sorted(df['original_language'].dropna().unique())
# Handle case where 'en' might not be in list (rare)
en_index = langs_list.index('en') if 'en' in langs_list else 0
months_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
studios_list = sorted(df['lead_studio'].astype(str).unique())

if options == "Overview":
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{len(df):,}")
    col2.metric("Avg Budget", f"${df['budget'].mean():,.0f}")
    col3.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}")
    col4.metric("Avg ROI", f"{df['roi'].median():.2f}x")
    st.dataframe(df.head())
    st.markdown("### Key Correlations")
    st.dataframe(analyzer.get_correlation_matrix().style.background_gradient(cmap="coolwarm"))

elif options == "Financial Analysis":
    st.header("💰 Financial Analysis")
    st.pyplot(visualizer.plot_budget_vs_revenue(df))
    st.pyplot(visualizer.plot_yearly_trends(analyzer.get_yearly_trends()))

elif options == "Genre & Studio":
    st.header("🎭 Genre & Studio Analytics")
    c1, c2 = st.columns(2)
    with c1: st.pyplot(visualizer.plot_genre_roi(analyzer.get_genre_metrics()))
    with c2: st.pyplot(visualizer.plot_top_studios(analyzer.get_top_studios()))

elif options == "Data Mining":
    st.header("⛏️ Data Mining Deep Dive")
    c1, c2 = st.columns(2)
    with c1: 
        st.subheader("Seasonality")
        st.pyplot(visualizer.plot_seasonal_revenue(analyzer.get_seasonal_stats()))
    with c2:
        st.subheader("Popularity vs Quality")
        st.pyplot(visualizer.plot_popularity_vs_rating(df))
    st.subheader("Runtime Sweet Spot")
    st.table(analyzer.get_runtime_metrics().head())

# --- ML MODULE 1: REGRESSION ---
elif options == "Revenue Predictor (Reg)":
    st.header("💰 AI Revenue Prediction")
    st.markdown("Predict the **exact dollar amount** a movie might earn.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        b_in = st.number_input("Budget ($)", 1000, 500000000, 10000000)
        r_in = st.slider("Runtime (mins)", 30, 240, 120)
        p_in = st.slider("Popularity Score", 1.0, 50.0, 10.0)
        v_in = st.number_input("Vote Count (Hype)", 0, 15000, 1000, help="How many people rated this movie? More votes usually means more viewers.")
        
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
        st.markdown("Which factors contributed most to this model?")
        st.pyplot(visualizer.plot_feature_importance(regressor.get_feature_importance()))

# --- ML MODULE 2: CLASSIFICATION ---
elif options == "Success Classifier (Clf)":
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