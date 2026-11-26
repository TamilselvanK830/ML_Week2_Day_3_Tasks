Task 1: Imputing and Encoding

Handled missing values using SimpleImputer and KNNImputer.

Applied OneHotEncoder and OrdinalEncoder.

Used LabelEncoder for target labels.

Created the final encoded DataFrame.

Task 2: Scaling

Applied StandardScaler and MinMaxScaler.

Trained KNN with and without scaling.

Trained Decision Tree with and without scaling.

Compared model accuracy.

Task 3: Creating New Features

Created a dataset of 200 customers.

Extracted year, month, and weekday from dates.

Created days_since_last_purchase and spend_per_visit.

Applied log1p transformation.

Visualized histograms and checked feature correlations with churn.

Task 4: Pipeline and ColumnTransformer

Built numerical and categorical preprocessing pipelines.

Used SimpleImputer, StandardScaler, and OneHotEncoder.

Combined preprocessing with LogisticRegression.

Evaluated using 5-fold cross-validation.

Predicted churn for 3 new customers.

Saved and loaded the Pipeline using joblib.

Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-learn

Joblib
