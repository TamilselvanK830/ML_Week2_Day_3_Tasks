import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib

df = pd.read_csv("customers.csv", sep="\t")

print(df.head(10))

print(df.shape)

df.info()

print(df.isnull().sum())

X = df.drop("churned", axis=1)

y = df["churned"]

print(X.head())

print(y.head())

print(y.value_counts())

def create_features(data):

    data = data.copy()

    reference_date = pd.Timestamp("2025-01-01")

    data["signup_date"] = pd.to_datetime(data["signup_date"])

    data["last_purchase_date"] = pd.to_datetime(
        data["last_purchase_date"]
    )

    data["days_since_last_purchase"] = (
        reference_date - data["last_purchase_date"]
    ).dt.days

    data["signup_month"] = (
        data["signup_date"].dt.month.astype(str)
    )

    data["signup_weekday"] = (
        data["signup_date"].dt.day_name()
    )

    return data[[
        "total_spend",
        "visits",
        "days_since_last_purchase",
        "signup_month",
        "signup_weekday"
    ]]


print("Feature engineering function created successfully!")

numeric_features = [
    "total_spend",
    "visits",
    "days_since_last_purchase"
]

categorical_features = [
    "signup_month",
    "signup_weekday"
]

print(numeric_features)

print(categorical_features)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print(preprocessor)

model = Pipeline(steps=[
    ("feature_engineering",
     FunctionTransformer(create_features, validate=False)),

    ("preprocessor", preprocessor),

    ("classifier", LogisticRegression(max_iter=1000))
])

print(model)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Individual Fold Accuracy:")
print(np.round(cv_scores, 4))

print("\nPipeline CV accuracy: {:.2f} +/- {:.2f}".format(
    cv_scores.mean(),
    cv_scores.std()
))

print("===== DATA LEAKAGE PREVENTION =====")

print("1. The dataset is divided into 5 folds during cross-validation.")

print("2. Imputation, scaling and encoding are fitted only on each fold's training data.")

print("3. The fitted preprocessing steps are applied to that fold's validation data.")

print("4. Validation data is not used to learn preprocessing parameters.")

print("5. Therefore, preprocessing inside the Pipeline prevents leakage between training and validation folds.")

model.fit(X, y)

print("Training samples:", len(X))

new_customers = pd.DataFrame({
    "signup_date": [
        "2024-03-15",
        "2023-06-20",
        "2024-09-10"
    ],

    "last_purchase_date": [
        "2024-12-20",
        "2024-05-10",
        "2024-11-15"
    ],

    "total_spend": [
        30000,
        15000,
        45000
    ],

    "visits": [
        20,
        5,
        15
    ]
})

print("===== THREE NEW RAW CUSTOMER ROWS =====")

print(new_customers.to_string(index=False))

predictions = model.predict(new_customers)

print("Predictions:", predictions)

for i, prediction in enumerate(predictions, start=1):

    if prediction == 1:
        result = "Customer may churn"
    else:
        result = "Customer may not churn"

    print(f"Customer {i}: {result}")

joblib.dump(model, "customer_churn_pipeline.pkl")

loaded_model = joblib.load("customer_churn_pipeline.pkl")

loaded_predictions = loaded_model.predict(new_customers)

print("===== LOADED PIPELINE PREDICTIONS =====")

print("Original Predictions:", predictions)

print("Loaded Predictions:", loaded_predictions)

print("\nPredictions match:",
      np.array_equal(predictions, loaded_predictions))

print("Dataset Shape:", df.shape)

print("Number of Features:", X.shape[1])

print("\n5-Fold Cross Validation Accuracy:")
print(f"{cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")
print("\nNew Customer Predictions:", predictions)
print("Pipeline Saved: customer_churn_pipeline.pkl")


