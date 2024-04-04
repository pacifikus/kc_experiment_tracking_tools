import mlflow
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV

mlflow.set_tracking_uri("http://127.0.0.1:8085")
experiment = mlflow.set_experiment("Sklearn diabetes grid search 0")

# enable autologging
mlflow.sklearn.autolog(max_tuning_runs=10)

# prepare training data
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2024)

parameters = {
    'alpha': np.linspace(0.1, 1, 10),
    'l1_ratio': np.linspace(0.2, 0.6, 3)
}

model = GridSearchCV(
    ElasticNet(),
    parameters,
    scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
    refit='neg_root_mean_squared_error',
)

with mlflow.start_run() as run:
    model.fit(X, y)
