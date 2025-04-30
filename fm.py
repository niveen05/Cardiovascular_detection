import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\nivee\Downloads\MLOPs_miniproj\cardio_train.csv", sep=';')  # Replace with your actual filename

# Clean column names: lowercase and strip spaces
df.columns = df.columns.str.strip().str.lower()

# Step 2: Drop 'id' if exists
df = df.drop(columns=['id'], errors='ignore')

# Step 3: Convert age from days to years
df['age'] = (df['age'] / 365).astype(int)

# Step 4: Filter outliers
df = df[
    (df['ap_hi'] >= 80) & (df['ap_hi'] <= 250) &
    (df['ap_lo'] >= 40) & (df['ap_lo'] <= 200) &
    (df['height'] >= 120) & (df['height'] <= 220) &
    (df['weight'] >= 30) & (df['weight'] <= 200)
]

# Step 5: Feature Engineering
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4]).astype('int')

# Step 6: Prepare data
X = df.drop(columns=['cardio'])  # Target variable
y = df['cardio']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 8: Train XGBoost Classifier
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = xgb_model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
import joblib

# Assuming `xgb_model` is your trained model
model = joblib.load("xgb_heart_model.pkl")


