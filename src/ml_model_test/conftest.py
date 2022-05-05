import pytest
import numpy as np


def pytest_addoption(parser):
    parser.addoption("--rf_config", action="store")

@pytest.fixture(scope='session')
def dummy_feats_and_labels():
    feats = np.array(
        [
            [
                26, 257910, 0, 0, 60, 'Private', 'Some-college', 'United-states',
                10, 'Never-married', 'Other-service', 'Not-in-family', 'White','Male'
            ],
            [
                58, 183893, 0, 0, 40, 'Private', 'Some-college', 'United-states',
                10, 'Divorced', 'Adm-clerical', 'Unmarried', 'Black', 'Female'
            ],
            [
                45, 271962, 0, 0, 40, 'State-gov', 'Bachelors', 'United-states',
                13, 'Divorced', 'Protective-serv', 'Not-in-family', 'White','Female'
            ],
            [
                54, 96062, 0, 0, 40, 'Private', 'Assoc-acdm', 'United-states',
                12, 'Married-civ-spouse', 'Tech-support', 'Husband', 'White', 'Male'
            ],
            [
                24, 124971, 0, 0, 38, 'Private', 'Hs-grad', 'United-states',
                9,'Never-married', 'Sales', 'Not-in-family', 'White', 'Male'
            ],
            [
                21, 176486, 0, 0, 60, 'Private', 'Hs-grad', 'United-states',
                9, 'Married-spouse-absent', 'Exec-managerial', 'Other-relative','White', 'Female'
            ],
            [
                34, 312197, 0, 0, 75, 'Self-emp-not-inc', 'Hs-grad', 'Mexico',
                9, 'Married-civ-spouse', 'Transport-moving', 'Husband', 'White', 'Male'
            ],
            [
                44, 171722, 0, 0, 39, 'Private', 'Hs-grad', 'United-states', 
                9, 'Separated', 'Other-service', 'Unmarried', 'White', 'Female'
            ],
            [
                41, 47902, 0, 0, 45, 'Local-gov', 'Bachelors', 'United-states',
                13, 'Married-civ-spouse', 'Prof-specialty', 'Husband', 'White', 'Male'
            ],
            [
                54, 377701, 0, 0, 32, 'Private', 'Hs-grad', 'Mexico',
                9, 'Married-civ-spouse', 'Other-service', 'Husband', 'White', 'Male'
            ]
        ]
    )
    labels = np.array([False, False, False,  True, False, False,  True, False,  True, False])
    return feats, labels

@pytest.fixture(scope='session')
def rf_config(request):
    rf_config = request.config.option.rf_config
    if rf_config is None:
        pytest.fail("You must provide a rf config")
    return dict(rf_config)