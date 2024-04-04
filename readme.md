## karpov.courses ML Engineering 

An example of working with experiment tracking tools

### How to run

Install all dependencies first

```
pip install -r requirements.txt
```

#### MLFlow

Start MLFlow
```
mlflow ui
mlflow server --host 127.0.0.1 --port 8085
```

Run python script

```
python mlflow/simple_run.py
```

To run example with hyperparams optimization use

```
python mlflow/hyperparams_search.py
```

#### ClearML

Run ClearML initialization

```commandline
clearml-init
```

Register with Clearml and create your application credentials [here](https://app.clear.ml/settings/workspace-configuration)

Copy the app credentials for input to 'clearml-init' configuration request, or modify your existing clearml.conf

Run python script

```
python clearml/simple_run.py
```

#### Weights & Biases

Login to W&B from CLI

```commandline
wandb login
```

Run python script

```
python w_and_b/simple_run.py
```

To run example with hyperparams optimization use

```
python mlflow/hyperparams_sweeper.py
```