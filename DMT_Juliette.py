import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load dataset
df = pd.read_csv("num.csv", sep=";")
df = df.drop(columns=["timestamp"])  # Drop non-numeric column

# features + target
X = df.drop(columns=["stress"])
y = df["stress"]

# k-fold cross validation 
k = 5
cv = KFold(n_splits=k, shuffle=True, random_state=42)

# linear regression pipeline
lr_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# evaluate linear regression
lr_mae_scores = -cross_val_score(lr_pipeline, X, y, scoring="neg_mean_absolute_error", cv=cv)
lr_mse_scores = -cross_val_score(lr_pipeline, X, y, scoring="neg_mean_squared_error", cv=cv)

# hyperparameter optimization
rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("regressor", RandomForestRegressor(random_state=42))
])

param_grid = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__max_depth": [None, 10, 20, 30]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, scoring="neg_mean_absolute_error", cv=cv)
grid_search.fit(X, y)
best_rf = grid_search.best_estimator_

# evaluate random forest
rf_mae_scores = -cross_val_score(best_rf, X, y, scoring="neg_mean_absolute_error", cv=cv)
rf_mse_scores = -cross_val_score(best_rf, X, y, scoring="neg_mean_squared_error", cv=cv)

# results
print("\nLinear Regression:")
print(f"MAE: {lr_mae_scores.mean():.3f} ± {1.96 * lr_mae_scores.std() / np.sqrt(k):.3f}")
print(f"MSE: {lr_mse_scores.mean():.3f}")

print("\nRandom Forest (Best Params):", grid_search.best_params_)
print(f"MAE: {rf_mae_scores.mean():.3f} ± {1.96 * rf_mae_scores.std() / np.sqrt(k):.3f}")
print(f"MSE: {rf_mse_scores.mean():.3f}")

# plot MAE per fold
plt.plot(range(1, k+1), lr_mae_scores, label="Linear Regression")
plt.plot(range(1, k+1), rf_mae_scores, label="Random Forest")
plt.xlabel("Fold")
plt.ylabel("MAEr")
plt.title("MAE per Fold")
plt.legend()
plt.grid(True)
plt.show()
