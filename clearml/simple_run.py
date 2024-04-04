import joblib

from clearml import Task

from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

task = Task.init(
    project_name='ClearML_Test',
    task_name='sklearn_diabetes',
    tags=['ElasticNet']
)


def eval_metrics(actual, pred):
    rmse = mean_squared_error(actual, pred, squared=False)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(param_config=None):
    if param_config is None:
        param_config = {'alpha': 0.5, 'l1_ratio': 0.5}

    task.connect(param_config)

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

    logger = task.get_logger()
    logger.report_single_value(name='rmse', value=rmse)
    logger.report_single_value(name='r2', value=r2)
    logger.report_single_value(name='mae', value=mae)

    joblib.dump(lr, 'elastic_model.pkl', compress=True)

    task.close()


train()
train({'alpha': 1, 'l1_ratio': 0.5})
