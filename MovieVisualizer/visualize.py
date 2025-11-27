"""
Module: visualize.py
Description: Generates matplotlib/seaborn figures for the Streamlit app.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class MovieVisualizer:
    """
    Creates visualizations for the analysis.
    """
    def __init__(self):
        sns.set_theme(style="whitegrid")

    def plot_genre_roi(self, genre_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=genre_df.head(10), x='primary_genre', y='median_roi', palette='viridis', hue='primary_genre', legend=False, ax=ax)
        ax.set_title('Top 10 Genres by ROI', fontsize=14)
        ax.set_ylabel('Median ROI', fontsize=12)
        plt.xticks(rotation=45)
        return fig

    def plot_budget_vs_revenue(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(data=df, x='budget', y='revenue', scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'}, ax=ax)
        ax.set_title('Budget vs. Revenue (Log Scale)', fontsize=14)
        ax.set_xscale('log'); ax.set_yscale('log')
        return fig

    def plot_yearly_trends(self, yearly_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=yearly_df, x='year', y='avg_budget', marker='o', color='purple', ax=ax)
        ax.set_title('Average Movie Budget Over Time', fontsize=14)
        return fig

    def plot_seasonal_revenue(self, seasonal_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=seasonal_df, x='month', y='median_revenue', palette='coolwarm', hue='month', legend=False, ax=ax)
        ax.set_title('Revenue by Release Month', fontsize=14)
        plt.xticks(rotation=45)
        return fig

    def plot_top_studios(self, studio_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=studio_df, y='lead_studio', x='median_revenue', palette='magma', hue='lead_studio', legend=False, ax=ax)
        ax.set_title('Top Studios by Revenue', fontsize=14)
        return fig

    def plot_popularity_vs_rating(self, df: pd.DataFrame):
        df_plot = df.copy()
        df_plot['log_popularity'] = np.log1p(df_plot['popularity'])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x='log_popularity', y='vote_average', hue='primary_genre', alpha=0.5, s=15, legend=False, ax=ax)
        ax.set_title('Popularity vs. Critic Rating', fontsize=14)
        return fig

    def plot_feature_importance(self, importance_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='mako', hue='feature', legend=False, ax=ax)
        ax.set_title('Feature Importance (What drives Revenue?)', fontsize=14)
        return fig

    def plot_confusion_matrix(self, cm, threshold_val):
        """Visualizes the confusion matrix for the classifier."""
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Average', 'Hit'], yticklabels=['Average', 'Hit'])
        ax.set_title(f'Classification Accuracy\n(Hit Threshold: Rating >= {threshold_val:.1f})', fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        return fig