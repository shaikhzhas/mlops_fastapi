#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
LOGGER = logging.getLogger()


def run_step(args):

    run = wandb.init(job_type="data_cleaning")
    run.config.update(args)

    LOGGER.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df_raw = pd.read_csv(artifact_local_path)
    # cleaning column names from spaces
    df_raw.columns = [col.strip() for col in df_raw.columns]
    # dropping duplicated rows
    df_clean = df_raw.drop_duplicates()
    # preprocessing categorical columns
    categorical_columns = [
        'education', 'relationship', 'workclass', 'marital-status',
        'occupation', 'race', 'sex', 'native-country', 'salary'
    ]
    for col in categorical_columns:
        df_clean[col] = df_clean[col].str.strip().str.capitalize()
    # encoding salary category as boolean variables for modeling
    df_clean['salary'] = df_clean['salary'].apply(lambda x: x == '>50k')

    filename = "../../data/clean_census.csv"
    df_clean.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    LOGGER.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This steps cleans the data",
        fromfile_prefix_chars="@",
    )
    PARSER.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )
    PARSER.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact",
        required=True)
    PARSER.add_argument(
        "--output_type",
        type=str,
        help="Type for the output artifact",
        required=True)
    PARSER.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True,
    )
    ARGUMENTS = PARSER.parse_args()
    run_step(ARGUMENTS)
