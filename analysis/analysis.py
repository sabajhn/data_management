import pandas as pd

def get_genre_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups data by 'primary_genre' and calculates median financial metrics.
    Only considers genres with at least 50 movies to ensure statistical significance.
    """
    print("Calculating genre success metrics...")
    
    # Aggregate data
    genre_stats = df.groupby('primary_genre').agg(
        count=('id', 'count'),
        median_budget=('budget', 'median'),
        median_revenue=('revenue', 'median'),
        median_roi=('roi', 'median'),
        avg_vote=('vote_average', 'mean')
    ).reset_index()
    
    # Filter for significant genres (e.g., > 50 movies)
    significant_genres = genre_stats[genre_stats['count'] >= 50].sort_values(by='median_roi', ascending=False)
    
    return significant_genres

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the correlation between numerical columns.
    """
    cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity', 'roi']
    return df[cols].corr()

def get_yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups data by year to see trends over time.
    """
    yearly_stats = df.groupby('year').agg(
        total_revenue=('revenue', 'sum'),
        avg_budget=('budget', 'mean'),
        movie_count=('id', 'count')
    ).reset_index()
    
    # Filter out very recent years if data is incomplete, or years way in the past with few movies
    return yearly_stats[yearly_stats['year'] > 1980]