"""
Module: visualize.py
Description: Generates matplotlib/seaborn figures for the Streamlit app.
             Includes advanced visualizations like WordClouds, Violin plots, and Boxen plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud

class MovieVisualizer:
    """
    Creates visualizations for the analysis.
    """
    def __init__(self):
        # Set a global theme for all plots
        sns.set_theme(style="whitegrid")

    # --- Existing Plots ---

    def plot_genre_roi(self, genre_df: pd.DataFrame):
        """Returns a bar chart figure for Median ROI by Genre."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=genre_df.head(10), 
            x='primary_genre', 
            y='median_roi', 
            palette='viridis', 
            hue='primary_genre', 
            legend=False,
            ax=ax
        )
        ax.set_title('Top 10 Genres by Return on Investment (ROI)', fontsize=14)
        ax.set_xlabel('Genre', fontsize=12)
        ax.set_ylabel('Median ROI (Profit/Budget)', fontsize=12)
        plt.xticks(rotation=45)
        return fig

    def plot_budget_vs_revenue(self, df: pd.DataFrame):
        """Returns a scatter plot figure with regression line."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.regplot(
            data=df, 
            x='budget', 
            y='revenue', 
            scatter_kws={'alpha': 0.3, 's': 10}, 
            line_kws={'color': 'red'},
            ax=ax
        )
        ax.set_title('Correlation: Budget vs. Revenue (Log Scale)', fontsize=14)
        ax.set_xlabel('Budget (USD)', fontsize=12)
        ax.set_ylabel('Revenue (USD)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        return fig

    def plot_yearly_trends(self, yearly_df: pd.DataFrame):
        """Returns a line chart figure for budget trends."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(
            data=yearly_df, 
            x='year', 
            y='avg_budget', 
            marker='o', 
            color='purple',
            ax=ax
        )
        ax.set_title('Trend of Average Movie Budgets (1980-Present)', fontsize=14)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Average Budget (USD)', fontsize=12)
        return fig

    def plot_seasonal_revenue(self, seasonal_df: pd.DataFrame):
        """Returns a bar chart figure for seasonality."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=seasonal_df, 
            x='month', 
            y='median_revenue', 
            palette='coolwarm', 
            hue='month', 
            legend=False,
            ax=ax
        )
        ax.set_title('Blockbuster Season: Revenue by Release Month', fontsize=14)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Median Revenue (USD)', fontsize=12)
        plt.xticks(rotation=45)
        return fig

    def plot_top_studios(self, studio_df: pd.DataFrame):
        """Returns a horizontal bar chart figure for top studios."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.barplot(
            data=studio_df, 
            y='lead_studio', 
            x='median_revenue', 
            palette='magma', 
            hue='lead_studio', 
            legend=False,
            ax=ax
        )
        ax.set_title('Top 10 Major Studios by Median Box Office', fontsize=14)
        ax.set_xlabel('Median Revenue (USD)', fontsize=12)
        ax.set_ylabel('Production Company', fontsize=12)
        return fig

    def plot_popularity_vs_rating(self, df: pd.DataFrame):
        """Returns a scatter plot of Popularity vs Rating."""
        df_plot = df.copy()
        df_plot['log_popularity'] = np.log1p(df_plot['popularity'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_plot, 
            x='log_popularity', 
            y='vote_average', 
            hue='primary_genre',
            alpha=0.5, 
            s=15,
            legend=False,
            ax=ax
        )
        ax.set_title('Popularity vs. Critic Rating', fontsize=14)
        ax.set_xlabel('Log(Popularity Score)', fontsize=12)
        ax.set_ylabel('Vote Average (0-10)', fontsize=12)
        return fig

    def plot_feature_importance(self, importance_df: pd.DataFrame):
        """Returns a bar chart of the Machine Learning Feature Importance."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=importance_df,
            x='importance',
            y='feature',
            palette='mako',
            hue='feature',
            legend=False,
            ax=ax
        )
        ax.set_title('Feature Importance (What drives the Model?)', fontsize=14)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
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

    # --- NEW: Advanced Visualizations (CLEANED) ---

    def plot_decade_pie(self, df: pd.DataFrame):
        """
        Creates a Pie Chart showing the distribution of movies by decade.
        Cleaned to prevent overlapping labels by using a threshold.
        """
        # Create a copy to avoid SettingWithCopyWarning
        df_plot = df.copy()
        df_plot = df_plot.dropna(subset=['year'])
        df_plot['decade'] = (df_plot['year'] // 10 * 10).astype(int).astype(str) + 's'
        
        counts = df_plot['decade'].value_counts()
        total = counts.sum()
        
        # Define threshold for showing labels on the pie (e.g., 2%)
        threshold = 0.02
        
        # Create labels list: show label only if slice > threshold
        labels = [label if (count/total) > threshold else '' for label, count in counts.items()]
        
        # Custom autopct to hide small percentages
        def autopct_format(pct):
            return ('%1.1f%%' % pct) if pct > (threshold * 100) else ''

        fig, ax = plt.subplots(figsize=(10, 6)) # Wider figure for legend
        
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=labels, 
            autopct=autopct_format,
            startangle=90,
            colors=sns.color_palette('pastel'),
            explode=[0.05 if i == 0 else 0 for i in range(len(counts))], # Explode biggest slice
            textprops={'fontsize': 10}
        )
        
        # Add a side legend for clarity
        ax.legend(wedges, counts.index,
          title="Decades",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
        
        ax.set_title('Movie Releases by Decade', fontsize=16)
        plt.tight_layout()
        return fig

    def plot_genre_wordcloud(self, df: pd.DataFrame):
        """
        Generates a Word Cloud of movie genres.
        """
        # Flatten the list of lists into a single list of genres
        # Filter out invalid entries first
        valid_genres = df['genres_list'].dropna()
        all_genres = [genre for sublist in valid_genres for genre in sublist]
        
        # Create a frequency dictionary
        genre_text = " ".join(all_genres)
        
        # Generate WordCloud
        wc = WordCloud(
            background_color="white", 
            width=800, 
            height=400, 
            colormap='magma',
            max_words=100
        ).generate(genre_text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.set_title('Genre Word Cloud', fontsize=16)
        return fig

    def plot_genre_popularity_boxen(self, df: pd.DataFrame):
        """
        Creates a Boxen Plot (Letter-Value Plot) for Popularity by Genre.
        Good for showing "heavy tailed" distributions.
        """
        # Explode the genres so a movie with ['Action', 'Thriller'] counts for both
        df_exploded = df.explode('genres_list')
        
        # Keep top 10 genres to keep the plot readable
        top_genres = df_exploded['genres_list'].value_counts().head(10).index
        df_plot = df_exploded[df_exploded['genres_list'].isin(top_genres)]
        
        # Remove 0 popularity for log scale
        df_plot = df_plot[df_plot['popularity'] > 0]

        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.boxenplot(
            data=df_plot,
            x='genres_list',
            y='popularity',
            palette='Spectral',
            ax=ax
        )
        
        ax.set_yscale('log') # Log scale handles the skew in popularity
        ax.set_title('Popularity Distribution by Genre (Boxen Plot)', fontsize=16)
        ax.set_xlabel('Genre', fontsize=14)
        ax.set_ylabel('Popularity (Log Scale)', fontsize=14)
        plt.xticks(rotation=45)
        return fig

    def plot_genre_runtime_violin(self, df: pd.DataFrame):
        """
        Creates a Violin Plot for Runtime by Genre.
        Violin plots show the probability density of the data at different values.
        """
        df_exploded = df.explode('genres_list')
        top_genres = df_exploded['genres_list'].value_counts().head(10).index
        df_plot = df_exploded[df_exploded['genres_list'].isin(top_genres)]
        
        # Filter reasonable runtimes
        df_plot = df_plot[(df_plot['runtime'] > 60) & (df_plot['runtime'] < 200)]

        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.violinplot(
            data=df_plot,
            x='genres_list',
            y='runtime',
            palette='Set3',
            inner='quartile', # Show quartiles inside the violin
            ax=ax
        )
        
        ax.set_title('Runtime Distribution by Genre (Violin Plot)', fontsize=16)
        ax.set_xlabel('Genre', fontsize=14)
        ax.set_ylabel('Runtime (Minutes)', fontsize=14)
        plt.xticks(rotation=45)
        return fig