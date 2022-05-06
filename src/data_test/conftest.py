"""
conftest for data test step
"""
import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    """
    pytest addoption
    """
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_age", action="store")
    parser.addoption("--max_age", action="store")


@pytest.fixture(scope='session')
def data(request):
    """
    fixture for data
    """
    run = wandb.init(job_type="data_test", resume=True)
    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()
    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")
    df_test = pd.read_csv(data_path)
    return df_test


@pytest.fixture(scope='session')
def ref_data(request):
    """
    fixture for ref data
    """
    run = wandb.init(job_type="data_test", resume=True)
    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref).file()
    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")
    df_ref = pd.read_csv(data_path)
    return df_ref


@pytest.fixture(scope='session')
def kl_threshold(request):
    """
    fixture for kl_threshold
    """
    kl_threshold_ = request.config.option.kl_threshold
    if kl_threshold_ is None:
        pytest.fail("You must provide a threshold for the KL test")
    return float(kl_threshold_)


@pytest.fixture(scope='session')
def min_age(request):
    """
    fixture for min age
    """
    min_age_ = request.config.option.min_age
    if min_age_ is None:
        pytest.fail("You must provide min_age")
    return int(min_age_)


@pytest.fixture(scope='session')
def max_age(request):
    """
    fixture for max age
    """
    max_age_ = request.config.option.max_age
    if max_age_ is None:
        pytest.fail("You must provide max_age")
    return int(max_age_)
