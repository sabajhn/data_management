import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

def plot_genre_roi(genre_df: pd.DataFrame):
    """
    Bar plot of Median ROI by Genre.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(data=genre_df.head(10), x='primary_genre', y='median_roi', palette='viridis')
    plt.title('Top 10 Genres by Median ROI (Return on Investment)')
    plt.xlabel('Genre')
    plt.ylabel('Median ROI (Profit / Budget)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/genre_roi.png')
    print("Saved plot: results/genre_roi.png")
    plt.close()

def plot_budget_vs_revenue(df: pd.DataFrame):
    """
    Scatter plot of Budget vs Revenue with a regression line.
    """
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='budget', y='revenue', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title('Budget vs. Revenue')
    plt.xlabel('Budget (USD)')
    plt.ylabel('Revenue (USD)')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('results/budget_vs_revenue.png')
    print("Saved plot: results/budget_vs_revenue.png")
    plt.close()

def plot_yearly_trends(yearly_df: pd.DataFrame):
    """
    Line plot of Average Budget over the years.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=yearly_df, x='year', y='avg_budget', marker='o', color='purple')
    plt.title('Trend of Average Movie Budgets (1980+)')
    plt.xlabel('Year')
    plt.ylabel('Average Budget (USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/yearly_budget_trend.png')
    print("Saved plot: results/yearly_budget_trend.png")
    plt.close()