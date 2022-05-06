"""
This script download a URL to a local destination
"""
import argparse
import logging
import os
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
LOGGER = logging.getLogger()


def run_step(args):
    """
    run data register step
    """
    run = wandb.init(job_type="data_register")
    run.config.update(args)
    LOGGER.info(f"Uploading {args.artifact_name} to Weights & Biases")
    # Log to W&B
    artifact = wandb.Artifact(
        args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(os.path.join("../../data", args.file_name))
    run.log_artifact(artifact)
    artifact.wait()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Download URL to a local destination")
    PARSER.add_argument("file_name", type=str, help="File name to upload")
    PARSER.add_argument(
        "artifact_name",
        type=str,
        help="Name for the output artifact")
    PARSER.add_argument(
        "artifact_type",
        type=str,
        help="Output artifact type.")
    PARSER.add_argument(
        "artifact_description",
        type=str,
        help="A brief description of this artifact")
    ARGUMENTS = PARSER.parse_args()
    run_step(ARGUMENTS)
