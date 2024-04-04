import joblib
import numpy as np
import wandb
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Prepare sweeper config and setup hyperparams
sweep_config = {
    'method': 'grid',
    'parameters': {
        'alpha': {
            'values': list(np.linspace(0.1, 1, 10))
        },
        'l1_ratio': {
            'values': list(np.linspace(0.2, 0.6, 3))
        }
    },
    'metric': {
        'name': 'rmse',
        'goal': 'minimize'
    }
}
sweep_id = wandb.sweep(sweep_config, project="sklearn_elastic_hyperparams_searh")

# Prepare data
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2024)


def eval_metrics(actual, pred):
    rmse = mean_squared_error(actual, pred, squared=False)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        lr = ElasticNet(
            alpha=config.alpha,
            l1_ratio=config.alpha,
            random_state=2024,
        )
        lr.fit(X_train, y_train)
        predicted_qualities = lr.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, predicted_qualities)
        wandb.log({"rmse": rmse, "r2": r2, "mae": mae})

        joblib.dump(lr, 'elastic_model.pkl', compress=True)

        artifact = wandb.Artifact("elastic_model", type="model")
        artifact.add_file('elastic_model.pkl')
        wandb.log_artifact(artifact)
        wandb.finish()


wandb.agent(sweep_id, train)
