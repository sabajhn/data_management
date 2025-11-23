# Data Management Project: CineMetrics

This repository is for managing and analyzing data related to **cinematic success factors**. It processes a dataset of 45,000 movies to explore relationships between budget, revenue, genres, and seasonality, aiming to determine the statistical "formula" for a hit movie.

## Project Overview

The **CineMetrics** project performs comprehensive data mining to identify key performance indicators in the film industry. It handles raw, semi-structured data (containing stringified JSON) and transforms it into actionable insights.

### Key Features
- **Data Cleaning:** Robust handling of raw CSV data, including parsing complex JSON columns for genres and production companies.
- **Financial Analysis:** Calculation of key metrics like ROI (Return on Investment) and profitability.
- **Data Mining:** Identification of "Blockbuster Seasons" and high-performing production studios.
- **Visualization:** Generation of professional-grade charts to visualize trends and correlations.

## Project Structure

This repository is organized as follows:

- `main.py`: The primary entry point that orchestrates the entire analysis pipeline.
- `data_cleaner.py`: Handles data loading, cleaning, JSON parsing, and feature engineering.
- `analysis.py`: Performs statistical computations (correlations, aggregations, groupings).
- `visualize.py`: Generates and saves plots for genres, budgets, and seasonality.
- `data/`: Folder containing the input dataset (`movies_metadata.csv`).
- `results/`: Folder where the output graphs and charts are saved.

## Getting Started

Follow these instructions to set up the project locally.

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries using:

```bash
pip install -r requirements.txt


