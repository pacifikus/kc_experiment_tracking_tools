import joblib
import wandb
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def eval_metrics(actual, pred):
    rmse = mean_squared_error(actual, pred, squared=False)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(param_config=None, run=0):
    if param_config is None:
        param_config = {'alpha': 0.5, 'l1_ratio': 0.5}

    run = wandb.init(
        project="sklearn_elastic_0",
        name=f"experiment_{run}",
        config=param_config,
    )

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2024)

    lr = ElasticNet(
        alpha=param_config['alpha'],
        l1_ratio=param_config['l1_ratio'],
        random_state=2024,
    )
    lr.fit(X_train, y_train)
    predicted_qualities = lr.predict(X_test)
    rmse, mae, r2 = eval_metrics(y_test, predicted_qualities)

    wandb.summary['rmse'] = rmse
    wandb.summary['r2'] = r2
    wandb.summary['mae'] = mae

    joblib.dump(lr, 'elastic_model.pkl', compress=True)

    artifact = wandb.Artifact("elastic_model", type="model")
    artifact.add_file('elastic_model.pkl')
    wandb.log_artifact(artifact)
    wandb.finish()


train()
train({'alpha': 1, 'l1_ratio': 0.5}, run=1)
