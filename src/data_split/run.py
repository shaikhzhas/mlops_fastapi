"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
LOGGER = logging.getLogger()


def run_step(args):
    """
    run data split step
    """
    run = wandb.init(job_type="data_split")
    run.config.update(args)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    LOGGER.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    df_clean = pd.read_csv(artifact_local_path)

    LOGGER.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df_clean,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df_clean[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save to output files
    for df_part, k in zip([trainval, test], ['trainval', 'test']):
        LOGGER.info(f"Uploading {k}_data.csv dataset")
        with tempfile.NamedTemporaryFile("w") as file_name:
            df_part.to_csv(file_name.name, index=False)
            artifact = wandb.Artifact(
                f"{k}_data.csv",
                type=f"{k}_data",
                description=f"{k} split of dataset",
            )
            artifact.add_file(file_name.name)
            run.log_artifact(artifact)
            artifact.wait()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Split test and remainder")
    PARSER.add_argument("input", type=str, help="Input artifact to split")
    PARSER.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items")
    PARSER.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False)
    PARSER.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default='none',
        required=False)
    ARGUMENTS = PARSER.parse_args()
    run_step(ARGUMENTS)
