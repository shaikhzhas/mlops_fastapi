import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig

_steps = [
    "data_register",
    "data_cleaning",
    "data_test",
    "data_split",
    "train_ml_model",
    "ml_model_test",
    "evaluate_ml_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "data_register" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_register"),
                "main",
                parameters={
                    "file_name": config["etl"]["file_name"],
                    "artifact_name": "census.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw census file"
                },
            )

        if "data_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_cleaning"),
                "main",
                parameters={
                    "input_artifact": "census.csv:latest",
                    "output_artifact": "clean_census.csv",
                    "output_type": "clean_census",
                    "output_description": "Data with outliers and null values removed"
                },
            )

        if "data_test" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_test"),
                "main",
                parameters={
                    "csv": "clean_census.csv:latest",
                    "ref": "clean_census.csv:latest",
                    "kl_threshold": config["data_test"]["kl_threshold"],
                    "min_age": config['etl']['min_age'],
                    "max_age": config['etl']['max_age']
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_split"),
                "main",
                parameters={
                    "input": "clean_census.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                },
            )

        if "train_ml_model" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            _ = mlflow.run(
                os.path.join(root_path, "src", "train_ml_model"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size" : config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "output_artifact": "random_forest_export"
                },
            )
        
        if "ml_model_test" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "ml_model_test"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "evaluate_ml_model" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "evaluate_ml_model"),
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:latest",
                    "test_dataset": "test_data.csv:latest"
                },
            )


if __name__ == "__main__":
    go()
