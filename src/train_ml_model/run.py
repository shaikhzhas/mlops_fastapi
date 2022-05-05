"""
This script trains a Random Forest
"""
import argparse
import logging
import mlflow
from mlflow.models import infer_signature
import json
import wandb
import pandas as pd
import numpy as np
import itertools
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="train_ml_model")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config['random_state'] = args.random_seed

    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    X = pd.read_csv(trainval_local_path)
    y = X.pop("salary")  # this removes the column "salary" from X and puts it into y
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")
    sk_pipe, used_columns = get_inference_pipeline(rf_config)
    # Then fit it to the X_train, y_train data
    logger.info("Fitting")
    sk_pipe.fit(X_train[used_columns], y_train)

    # Compute ROC AUC score
    logger.info("Scoring")
    y_pred = sk_pipe.predict(X_val[used_columns])
    auc_score = roc_auc_score(y_val, y_pred, average = "weighted")
    logger.info(f"Score: {auc_score}")

    logger.info("Exporting model")
    model_path = '../model'
    signature = infer_signature(X_val[used_columns], y_pred)
    mlflow.sklearn.save_model(
        sk_pipe,
        model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=signature,
        input_example=X_val.iloc[:2],
    )
    artifact = wandb.Artifact(
        name = args.output_artifact,
        type = "model_export",
        description = "Random Forest pipeline export",
        metadata = rf_config
    )
    artifact.add_dir('../model')
    run.log_artifact(artifact)

    # log auc score in W&B
    run.summary['auc'] = auc_score
    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, used_columns)
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

def plot_feature_importance(pipe, feat_names):
    feat_names = np.array(
        pipe["preprocessor"].transformers[0][-1]
        + pipe["preprocessor"].transformers[1][-1]
    )
    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names)]
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp[idx], color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp

def get_inference_pipeline(rf_config):
    categorical = [
        "workclass",
        "education",
        "native-country",
        "education-num",
        "marital-status",
        'occupation', 
        'relationship',
        'race',
        'sex'
    ]
    categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown='ignore')
    )
    numeric_features = [
        'age',
        'fnlgt',
        'capital-gain',
        'capital-loss',
        'hours-per-week'
    ]
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )
    preprocessor = ColumnTransformer(
        transformers = [
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_preproc, categorical)
            ],
        remainder="drop",  # This drops the columns that we do not transform
    ) 
    used_columns = list(itertools.chain.from_iterable([x[2] for x in preprocessor.transformers]))
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**rf_config)),
        ]
    )
    return sk_pipe, used_columns


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")
    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )
    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )
    args = parser.parse_args()
    go(args)
