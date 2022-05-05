#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
import itertools
from sklearn.metrics import roc_auc_score
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="evaluate_ml_model")
    run.config.update(args)

    logger.info("Downloading artifacts")
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    model_local_path = run.use_artifact(args.ml_model).download()

    # Download test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("salary")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    used_columns = list(itertools.chain.from_iterable([x[2] for x in sk_pipe['preprocessor'].transformers]))

    logger.info("Scoring test dataset")
    y_pred = sk_pipe.predict(X_test[used_columns])
    auc_score = roc_auc_score(y_test, y_pred, average="weighted")
    logger.info(f"Score: {auc_score}")
    run.summary['auc'] = auc_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--ml_model",
        type=str, 
        help="Input ML model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str, 
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
