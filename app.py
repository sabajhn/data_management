"""
Module: app.py
Description: The main entry point for the Streamlit Web Application.
             It orchestrates the Data Processing, Analysis, and Visualization classes
             to create an interactive dashboard.
"""

import streamlit as st
import pandas as pd
from MovieDataProcessor.data_cleaner import MovieDataProcessor, CSV_PATH
from MovieAnalyzer.analysis import MovieAnalyzer
from MovieVisualizer.visualize import MovieVisualizer 

# --- Page Configuration ---
st.set_page_config(
    page_title="CineMetrics: Blockbuster Analytics",
    page_icon="🎬",
    layout="wide"
)

# --- Title & Intro ---
st.title("🎬 CineMetrics: The Blockbuster Blueprint")
st.markdown("""
This interactive dashboard explores the factors that determine cinematic success. 
We analyze **45,000+ movies** to uncover trends in **Revenue**, **Budget**, **Genre**, and **Seasonality**.
""")

# --- 1. Data Loading (Cached) ---
@st.cache_data
def load_and_process_data():
    processor = MovieDataProcessor(path=CSV_PATH)
    df_raw = processor.load_data()
    df_clean = processor.preprocess_data(df_raw)
    return df_clean

# Check if data exists before trying to load
import os
if not os.path.exists(CSV_PATH):
    st.error(f"❌ Critical Error: Dataset not found at `{CSV_PATH}`.")
    st.info("Please create a 'data' folder and put 'movies_metadata.csv' inside it.")
    st.stop()

with st.spinner('Loading and cleaning data...'):
    try:
        df = load_and_process_data()
        st.success(f"Data loaded successfully! Analyzed {len(df):,} valid movie records.")
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

# --- Initialize Classes ---
analyzer = MovieAnalyzer(df)
visualizer = MovieVisualizer()

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
options = st.sidebar.radio(
    "Choose Analysis Module:", 
    ["Overview & Stats", "Financial Analysis", "Genre & Studio Insights", "Data Mining Deep Dive"]
)

# --- MODULE 1: Overview & Stats ---
if options == "Overview & Stats":
    st.header("📊 Dataset Overview")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{len(df):,}")
    col2.metric("Avg Budget", f"${df['budget'].mean():,.0f}")
    col3.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}")
    col4.metric("Avg ROI", f"{df['roi'].median():.2f}x")

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))
    
    st.subheader("Key Correlations")
    st.markdown("How strongly are these variables related? (1.0 = Perfect Correlation)")
    corr_matrix = analyzer.get_correlation_matrix()
    st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"))

# --- MODULE 2: Financial Analysis ---
elif options == "Financial Analysis":
    st.header("💰 Financial Analysis")
    
    st.subheader("Budget vs. Revenue")
    st.markdown("Does throwing money at a movie guarantee success?")
    fig_budget = visualizer.plot_budget_vs_revenue(df)
    st.pyplot(fig_budget)
    
    st.subheader("Historical Trends")
    st.markdown("How have movie budgets changed since 1980?")
    yearly_data = analyzer.get_yearly_trends()
    fig_trend = visualizer.plot_yearly_trends(yearly_data)
    st.pyplot(fig_trend)

# --- MODULE 3: Genre & Studio Insights ---
elif options == "Genre & Studio Insights":
    st.header("🎭 Genre & Studio Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Profitable Genres")
        genre_data = analyzer.get_genre_metrics()
        fig_genre = visualizer.plot_genre_roi(genre_data)
        st.pyplot(fig_genre)
        with st.expander("View Genre Data"):
            st.dataframe(genre_data)

    with col2:
        st.subheader("Top Studios (Revenue)")
        studio_data = analyzer.get_top_studios()
        fig_studio = visualizer.plot_top_studios(studio_data)
        st.pyplot(fig_studio)
        with st.expander("View Studio Data"):
            st.dataframe(studio_data)

# --- MODULE 4: Data Mining Deep Dive ---
elif options == "Data Mining Deep Dive":
    st.header("⛏️ Data Mining: Hidden Patterns")
    
    st.subheader("1. The 'Blockbuster Season'")
    st.markdown("Is there a specific month where movies make the most money?")
    seasonal_data = analyzer.get_seasonal_stats()
    fig_season = visualizer.plot_seasonal_revenue(seasonal_data)
    st.pyplot(fig_season)
    
    st.subheader("2. Popularity vs. Quality")
    st.markdown("Are popular movies actually rated higher by critics?")
    fig_pop = visualizer.plot_popularity_vs_rating(df)
    st.pyplot(fig_pop)
    
    st.subheader("3. The Runtime Sweet Spot")
    st.markdown("Do audiences prefer specific movie lengths?")
    runtime_data = analyzer.get_runtime_metrics()
    st.table(runtime_data)