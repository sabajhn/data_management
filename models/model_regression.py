"""
Module: model_regression.py
Description: Contains the RevenueRegressor class using Gradient Boosting 
             for high-accuracy prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

class RevenueRegressor:
    """
    Gradient Boosting Regressor to predict movie revenue.
    Optimized with 'vote_count' (hype) and 'lead_studio' for maximum accuracy.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.pipeline = None
        self.feature_names = []
        self.top_studios = [] # To store the list of major studios
        
        # Expanded Feature Set
        self.numeric_features = ['budget', 'runtime', 'popularity', 'is_franchise', 'vote_count']
        self.categorical_features = ['primary_genre', 'original_language', 'month', 'lead_studio']

        # Preprocessing Pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), self.numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), self.categorical_features)
            ])

    def preprocess_studios(self, data):
        """
        Groups small studios into 'Other' to improve model stability.
        Keeps only the top 50 studios by frequency.
        """
        # Calculate top 50 studios
        top_50 = data['lead_studio'].value_counts().nlargest(50).index.tolist()
        self.top_studios = top_50
        
        # Apply grouping
        data['lead_studio'] = data['lead_studio'].apply(lambda x: x if x in top_50 else 'Other')
        return data

    def train(self):
        """Trains the Gradient Boosting Regressor."""
        # 1. Prepare Data
        # Filter valid revenues
        data = self.df.dropna(subset=self.numeric_features + ['revenue'])
        data = data[data['revenue'] > 0]
        
        # Optimize Studio feature
        data = self.preprocess_studios(data)

        X = data[self.numeric_features + self.categorical_features]
        y = data['revenue']

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Define Pipeline (Gradient Boosting is generally more accurate than Random Forest)
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
        ])

        # 4. Fit
        self.pipeline.fit(X_train, y_train)

        # 5. Evaluate
        preds = self.pipeline.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, preds),
            'r2': r2_score(y_test, preds)
        }
        
        # Extract feature names
        try:
            onehot_cols = self.pipeline.named_steps['preprocessor'].transformers_[1][1]\
                .named_steps['onehot'].get_feature_names_out(self.categorical_features)
            self.feature_names = self.numeric_features + list(onehot_cols)
        except:
            self.feature_names = self.numeric_features

        return metrics

    def predict(self, budget, runtime, popularity, is_franchise, vote_count, genre, language, month, studio):
        """Predicts revenue for a single instance."""
        # Handle 'Other' studio logic for new inputs
        studio_processed = studio if studio in self.top_studios else 'Other'
        
        input_df = pd.DataFrame([[budget, runtime, popularity, is_franchise, vote_count, genre, language, month, studio_processed]], 
                                columns=self.numeric_features + self.categorical_features)
        
        return self.pipeline.predict(input_df)[0]

    def get_feature_importance(self):
        """Returns feature importance DataFrame."""
        if not self.pipeline: return None
        importances = self.pipeline.named_steps['regressor'].feature_importances_
        if len(self.feature_names) == len(importances):
            return pd.DataFrame({'feature': self.feature_names, 'importance': importances})\
                     .sort_values(by='importance', ascending=False).head(10)
        return None