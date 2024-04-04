import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:8085")
experiment = mlflow.set_experiment("Sklearn diabetes 1")


def eval_metrics(actual, pred):
    rmse = mean_squared_error(actual, pred, squared=False)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(param_config=None):
    if param_config is None:
        param_config = {'alpha': 0.5, 'l1_ratio': 0.5}

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2024)

    with mlflow.start_run():
        lr = ElasticNet(
            alpha=param_config['alpha'],
            l1_ratio=param_config['l1_ratio'],
            random_state=2024,
        )
        lr.fit(X_train, y_train)
        predicted_qualities = lr.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, predicted_qualities)

        signature = infer_signature(X_test, predicted_qualities)

        for param_name, param_value in param_config.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="elastic_model",
            signature=signature,
            registered_model_name="sk-learn-elastic"
        )


train()
train({'alpha': 1, 'l1_ratio': 0.5})
