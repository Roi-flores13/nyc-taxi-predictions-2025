import os
import optuna
import pathlib
import xgboost
import pandas as pd
import mlflow
import pickle
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from mlflow import MlflowClient
from prefect import flow, task
import importlib
from sklearn.ensemble import RandomForestRegressor

task_files = importlib.import_module("02-train_pipeline_prefect")

def objective_rtr(trial: optuna.trial.Trial, X_train, X_val, y_train, y_val) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 50, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0)
    }
    
    with mlflow.start_run(nested=True):
        mlflow.set_tag("model_family", "rfr")
        mlflow.log_params(params)
        
        model_rfr = RandomForestRegressor(**params)
        model_rfr.fit(X_train, y_train)
        
        y_pred = model_rfr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        
        mlflow.log_metric("validation_rmse", rmse)
        
        signature = infer_signature(X_val, y_pred)
        
        mlflow.sklearn.log_model(
            model_rfr,
            name="model",
            input_example=X_val[:5],
            signature=signature
        )
        
        return rmse

@task(name="find_best_params-rfr")
def hyper_parameter_tuning_rfr(X_train, X_val, y_train, y_val, dv):
    mlflow.sklearn.autolog(log_models=False)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    with mlflow.start_run(run_name="Random Forest Regressor/Optima", nested=True):
        study.optimize(
            lambda trial: objective_rtr(trial, X_train, X_val, y_train, y_val),
            n_trials=3)
        
    best_params = study.best_params
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["random_state"] = 42
    
    return best_params
        
        
@task(name="train_best_model_rfr")
def train_best_model_rfr(X_train, X_val, y_train, y_val, dv, best_params_rfr) -> None:
    with mlflow.start_run(run_name="Best_rfr_model"):
        mlflow.log_params(best_params_rfr)
        
        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "random forest",
            "feature_set_version": 1,
        })
        
        rfr = RandomForestRegressor(**best_params_rfr)
        rfr.fit(X_train, y_train)
        
        y_pred = rfr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")
        
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)

        signature = infer_signature(input_example, y_val[:5])
        
        mlflow.sklearn.log_model(
            rfr,
            name="model",
            input_example=input_example,
            signature=signature
        )
    
    return None


@flow(name="HW Flow")
def main_hw_flow(year: str, month_train:str, month_val:str) -> None:
    
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    load_dotenv(override=True)
    EXPERIMENT_NAME = "/Users/roiflores.2213@gmail.com/nyc-taxi-experiments"
    
    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    
    df_train = task_files.read_data(train_path)
    df_val = task_files.read_data(val_path)
    
    X_train, X_val, y_train, y_val, dv = task_files.add_features(df_train, df_val)
    
    best_params_xgb  = task_files.hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv)
    task_files.train_best_model(X_train, X_val, y_train, y_val, dv, best_params_xgb)
    
    best_params_rfr = hyper_parameter_tuning_rfr(X_train, X_val, y_train, y_val, dv)
    params = [X_train, X_val, y_train, y_val, dv, best_params_rfr]
    train_best_model_rfr(*params)
    
    task_files.get_best_model(EXPERIMENT_NAME, model_name="workspace.default.nyc-taxi-experiment")
    
if __name__ == "__main__":
    main_hw_flow("2025", "01", "02")
    
    
    
    