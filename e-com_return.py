# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import seaborn as sns

# 1. load the joined CSV (orders_master.csv)
df = pd.read_csv('orders_master_export.csv', parse_dates=[
    'order_purchase_timestamp','order_approved_at','order_delivered_customer_date','order_estimated_delivery_date'
])

# 2. basic cleaning
df['return_flag'] = df['return_flag'].astype(int)
df['delivery_delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days

df['delivery_delay_days'] = df['delivery_delay_days'].fillna(0)

df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
df['freight_value'] = pd.to_numeric(df['freight_value'], errors='coerce').fillna(0)

df['price_per_kg'] = df.apply(lambda r: (r['price'] / (r['product_weight_g']/1000)) if r['product_weight_g']>0 else r['price'], axis=1)

df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
df['is_weekend'] = df['purchase_dayofweek'].isin([5,6]).astype(int)


top_sellers = df['seller_id'].value_counts().nlargest(50).index
df['seller_top50'] = df['seller_id'].where(df['seller_id'].isin(top_sellers), 'other')


features = [
    'price','freight_value','delivery_delay_days','review_score',
    'product_length_cm','product_height_cm','product_width_cm','product_weight_g',
    'product_category_name','payment_type','purchase_month','is_weekend','seller_top50'
]
# drop rows with missing target
df = df.dropna(subset=['return_flag'])

X = df[features]
y = df['return_flag']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

num_features = ['price','freight_value','delivery_delay_days','review_score',
                'product_length_cm','product_height_cm','product_width_cm','product_weight_g']
cat_features = ['product_category_name','payment_type','seller_top50','purchase_month','is_weekend']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ],
    remainder='drop'
)


clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# train
clf.fit(X_train, y_train)

# predict + evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1:', f1_score(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_proba))
print('\nClassification report:\n', classification_report(y_test, y_pred))

# confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.show()

ohe = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
cat_cols = ohe.get_feature_names_out(cat_features)
all_features = num_features + list(cat_cols)
importances = clf.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(30)
plt.figure(figsize=(8,6))
feat_imp.plot(kind='barh')
plt.title('Top 30 Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# save predictions for Power BI dashboard
X_test2 = X_test.copy()
X_test2['return_prob'] = y_proba
X_test2['predicted_return'] = y_pred
X_test2['actual_return'] = y_test.values
X_test2.to_csv('return_predictions_for_powerbi.csv', index=False)
