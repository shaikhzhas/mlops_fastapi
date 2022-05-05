#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="data_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    # cleaning column names from spaces
    df.columns = [col.strip() for col in df.columns]
    # dropping duplicated rows
    df = df.drop_duplicates()
    # preprocessing categorical columns
    categorical_columns = [
        'education','relationship','workclass','marital-status',
        'occupation','race','sex','native-country','salary'
    ]
    for col in categorical_columns:
        df[col] = df[col].str.strip().str.capitalize()
    # encoding salary category as boolean variables for modeling
    df['salary'] = df['salary'].apply(lambda x: x=='>50k')

    filename = "../../data/clean_census.csv"
    df.to_csv(filename,index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This steps cleans the data",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )
    parser.add_argument(
        "--output_artifact", type=str, help="Name for the output artifact", required=True
    )
    parser.add_argument(
        "--output_type", type=str, help="Type for the output artifact", required=True
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True,
    )
    args = parser.parse_args()
    go(args)
