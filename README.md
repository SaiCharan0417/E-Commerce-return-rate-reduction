# E-Commerce-return-rate-reduction

## Project Overview
This project analyzes real-world e-commerce behavior using the Olist Brazilian E-Commerce Dataset to understand why customers cancel/return orders, identify high-risk product categories, and build a machine learning model to predict return probability.
The goal is to help e-commerce businesses reduce operational losses, improve customer satisfaction, and optimize seller/product performance.

## Dataset
Source: Olist Multi-Table Dataset
Contains 9 CSV files:
* Orders
* Order Items
* Payments
* Reviews
* Customers
* Sellers
* Products
* Geolocation
* Category Translation

## Tools used
* SQL (MySQL): Data import, joining tables, cleaning, creating master dataset
* Python (Pandas, Scikit-Learn):
Exploratory Data Analysis
Feature Engineering
Return Prediction Model (Random Forest / Logistic Regression)
* Power BI:
Category-wise return dashboard
Seller performance
Customer geography insights
ML prediction integration

## Key Steps in the Project
* Importing CSVs into MySQL
Enabled local_infile
Loaded all 9 CSVs into tables
Fixed datetime parsing & duplicates
Created a clean orders_master table with joins

* Data Cleaning & Pre-processing
Converted date columns safely
Handled invalid timestamps
Removed duplicates
Created new features:
Delivery delay days
Price-per-kg
Weekend purchase flag
Seller top-50 flag
Month, weekday, etc.

* Exploratory Data Analysis
Category-wise return %
Seller-wise return %
Relationship between delivery delay & returns
Review score vs cancellations
Price/freight distribution

* Machine Learning Model
Built a classification model to predict return probability.
Algorithms used:
Random Forest (primary)
Logistic Regression (baseline)
Evaluation metrics:
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix

* Power BI Dashboard
Delivered insights through dashboard

## Key Insights
* Certain categories have significantly higher return rates due to quality issues, logistics, size mismatch, etc.
* Delivery delays strongly correlate with cancellations.
* A small set of sellers contribute disproportionately to returns.
* High freight cost relative to product price increases return probability.

## Recommendations
* Improve seller onboarding & QC for high-return sellers.
* Use ML return probability to flag risky orders before shipping.
* Improve product images, size charts & descriptions.
* Prioritize accurate delivery time estimates.
* Review top return categories for packaging or design improvements.
