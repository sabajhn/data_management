# **CineMetrics: The Blockbuster Blueprint**

> **Note:** This project is an advanced data analytics and machine learning dashboard built with **Python** and **Streamlit**.

## **Project Overview**

**CineMetrics** processes a dataset of over 45,000 movies to uncover _hidden trends_, predict <ins>box office revenue</ins>, classify potential hits, and recommend similar movies based on content.

The primary goal of this project is to:
- Provide an **Interactive Dashboard** for high-level metrics (ROI, Budget vs. Revenue).
- Utilize **Machine Learning** to predict financial success.
- Rank movies using a **Weighted Rating Formula**.

---

## **Data Sources**

This project utilizes **The Movies Dataset** from Kaggle.

> "Data is the new oil." — _Clive Humby_

The data was sourced from:
- **Primary Dataset:** [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data)  
  - <sub>Source: Kaggle</sub>  
  - <ins>Description:</ins> Metadata on over 45,000 movies.

---

## **Features**

### **1. 📊 Interactive Dashboard**
- **Overview:** High-level metrics including **Total Movies**, **Avg Budget/Revenue**, and <ins>ROI</ins>.
- **Financial Analysis:** Budget vs. Revenue correlations and historical market trends (decades, yearly growth).
- **Genre & Studio Analytics:**
  - Genre Word Clouds.
  - _ROI Analysis_ by Genre.
- **Distributions:** Boxen plots (Popularity) and Violin plots (Runtime).
- **Data Mining:** Seasonality analysis (best month to release?) and Popularity vs. Quality checks.

---

### **2. 🤖 Machine Learning**
- **Success Classifier (Classification):**
  - Uses Gradient Boosting to classify a movie as a **"Hit"** (Critical Success) or **"Average/Flop"**.
  - Includes a <ins>Confusion Matrix</ins> for model performance evaluation.

---

### **3. 🌟 Top Charts (Ranking Algorithm)**
- Implements the **IMDB Weighted Rating Formula** to rank movies fairly.
- Balances a movie’s average rating with its number of votes.
- Allows filtering by **Genre** (e.g., *Top 10 Horror Movies*).

---

# **Setup & Installation**

Follow these instructions to set up the project on your local machine.

---

## **1. Prerequisites**
Ensure you have **Python 3.8+** installed.

---

## **2. Install Dependencies**
Run the following command to install all required libraries:

```bash
pip install -r requirements.txt

