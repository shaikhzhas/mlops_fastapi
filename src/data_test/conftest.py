import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_age", action="store")
    parser.addoption("--max_age", action="store")

@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_test", resume=True)
    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()
    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")
    df = pd.read_csv(data_path)
    return df

@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_test", resume=True)
    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref).file()
    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")
    df = pd.read_csv(data_path)
    return df

@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold
    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")
    return float(kl_threshold)

@pytest.fixture(scope='session')
def min_age(request):
    min_age = request.config.option.min_age
    if min_age is None:
        pytest.fail("You must provide min_age")
    return int(min_age)

@pytest.fixture(scope='session')
def max_age(request):
    max_age = request.config.option.max_age
    if max_age is None:
        pytest.fail("You must provide max_age")
    return int(max_age)