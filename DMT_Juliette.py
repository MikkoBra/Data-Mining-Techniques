import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# dataset
df = pd.read_csv("num.csv", sep=";")
df = df.drop(columns=["timestamp"])  # Drop non-numeric column

# features and target
X = df.drop(columns=["stress"])
y = df["stress"]

# 66/33 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# linear regression pipeline
lr_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# cross validation on training set
cv = KFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_scores = -cross_val_score(lr_pipeline, X_train, y_train, scoring="neg_mean_absolute_error", cv=cv)
lr_cv_mean = np.mean(lr_cv_scores)
lr_ci = 1.96 * np.std(lr_cv_scores) / np.sqrt(len(lr_cv_scores))

# fit on full training set and evaluate on test set
lr_pipeline.fit(X_train, y_train)
lr_test_preds = lr_pipeline.predict(X_test)
lr_test_mae = mean_absolute_error(y_test, lr_test_preds)
lr_test_mse = mean_squared_error(y_test, lr_test_preds)

# random forest with hyperparameter tuning
rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("regressor", RandomForestRegressor(random_state=42))
])

param_grid = {
    "regressor__n_estimators": [50, 100],
    "regressor__max_depth": [None, 10, 20]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, scoring="neg_mean_absolute_error", cv=cv)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# MAE on training set (cross validated)
rf_cv_scores = -cross_val_score(best_rf, X_train, y_train, scoring="neg_mean_absolute_error", cv=cv)
rf_cv_mean = np.mean(rf_cv_scores)
rf_ci = 1.96 * np.std(rf_cv_scores) / np.sqrt(len(rf_cv_scores))

# evaluate on test set
rf_test_preds = best_rf.predict(X_test)
rf_test_mae = mean_absolute_error(y_test, rf_test_preds)
rf_test_mse = mean_squared_error(y_test, rf_test_preds)

# results
print("=== Linear Regression ===")
print(f"Cross-validated MAE: {lr_cv_mean:.3f} ± {lr_ci:.3f} (95% CI)")
print(f"Test MAE: {lr_test_mae:.3f}")
print(f"Test MSE: {lr_test_mse:.3f}")

print("\n=== Random Forest ===")
print("Best Params:", grid_search.best_params_)
print(f"Cross-validated MAE: {rf_cv_mean:.3f} ± {rf_ci:.3f} (95% CI)")
print(f"Test MAE: {rf_test_mae:.3f}")
print(f"Test MSE: {rf_test_mse:.3f}")
