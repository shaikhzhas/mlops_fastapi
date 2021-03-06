"""
This script trains a Random Forest model
"""
import argparse
import logging
import json
import shutil
import importlib.util
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
LOGGER = logging.getLogger()

SPEC = importlib.util.spec_from_file_location(
    "common", os.path.abspath(
        __file__ + "/../../") + '/common/ml_pipeline.py')
ML_PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ML_PIPELINE)


def run_step(args):

    run = wandb.init(job_type="train_ml_model")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as file_name:
        rf_config = json.load(file_name)
    run.config.update(rf_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config['random_state'] = args.random_seed

    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    X_data = pd.read_csv(trainval_local_path)
    # this removes the column "salary" from X and puts it into y
    y_label = X_data.pop("salary")
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_label, test_size=args.val_size, stratify=X_data[args.stratify_by], random_state=args.random_seed
    )

    LOGGER.info("Preparing sklearn pipeline")
    sk_pipe, used_columns = ML_PIPELINE.get_inference_pipeline(rf_config)
    # Then fit it to the X_train, y_train data
    LOGGER.info("Fitting")
    sk_pipe.fit(X_train[used_columns], y_train)

    # Compute ROC AUC score
    LOGGER.info("Scoring")
    y_pred = sk_pipe.predict(X_val[used_columns])
    auc_score = roc_auc_score(y_val, y_pred, average="weighted")
    LOGGER.info(f"Score: {auc_score}")

    LOGGER.info("Exporting model")
    model_path = '../model'
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    signature = infer_signature(X_val[used_columns], y_pred)
    mlflow.sklearn.save_model(
        sk_pipe,
        model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=signature,
        input_example=X_val.iloc[:2],
    )
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type="model_export",
        description="Random Forest pipeline export",
        metadata=rf_config
    )
    artifact.add_dir('../model')
    run.log_artifact(artifact)

    # log auc score in W&B
    run.summary['auc'] = auc_score
    # Plot feature importance
    fig_feat_imp = ML_PIPELINE.plot_feature_importance(sk_pipe, used_columns)
    # Plot confusion matrix
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        sk_pipe,
        X_val[used_columns],
        y_val,
        ax=sub_cm,
        xticks_rotation=90,
    )
    fig_cm.tight_layout()
    # Upload to W&B the feture importance and confusion matrix visualization
    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
            "confusion_matrix": wandb.Image(fig_cm),
        }
    )
    shutil.copy2('../model/model.pkl', '../../model/model.pkl')


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description="Basic cleaning of dataset")
    PARSER.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )
    PARSER.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )
    PARSER.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )
    PARSER.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )
    PARSER.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )
    PARSER.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )
    ARGUMENTS = PARSER.parse_args()
    run_step(ARGUMENTS)
