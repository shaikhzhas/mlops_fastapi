"""
Conftest for ml testing
"""
import json
import pytest
import numpy as np
import pandas as pd


def pytest_addoption(parser):
    """
    pytest addoption
    """
    parser.addoption("--rf_config", action="store")


@pytest.fixture(scope='session')
def dummy_feats_and_labels():
    """
    fixture for dummy data
    """
    feats = pd.DataFrame({'age': [26,
                                  58,
                                  45,
                                  54,
                                  24,
                                  21,
                                  34,
                                  44,
                                  41],
                          'fnlgt': [257910,
                                    183893,
                                    271962,
                                    96062,
                                    124971,
                                    176486,
                                    312197,
                                    171722,
                                    47902],
                          'capital-gain': [0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0],
                          'capital-loss': [0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0],
                          'hours-per-week': [60,
                                             40,
                                             40,
                                             40,
                                             38,
                                             60,
                                             75,
                                             39,
                                             45],
                          'workclass': ['Private',
                                        'Private',
                                        'State-gov',
                                        'Private',
                                        'Private',
                                        'Private',
                                        'Self-emp-not-inc',
                                        'Private',
                                        'Local-gov'],
                          'education': ['Some-college',
                                        'Some-college',
                                        'Bachelors',
                                        'Assoc-acdm',
                                        'Hs-grad',
                                        'Hs-grad',
                                        'Hs-grad',
                                        'Hs-grad',
                                        'Bachelors'],
                          'native-country': ['United-states',
                                             'United-states',
                                             'United-states',
                                             'United-states',
                                             'United-states',
                                             'United-states',
                                             'Mexico',
                                             'United-states',
                                             'United-states'],
                          'education-num': [10,
                                            10,
                                            13,
                                            12,
                                            9,
                                            9,
                                            9,
                                            9,
                                            13],
                          'marital-status': ['Never-married',
                                             'Divorced',
                                             'Divorced',
                                             'Married-civ-spouse',
                                             'Never-married',
                                             'Married-spouse-absent',
                                             'Married-civ-spouse',
                                             'Separated',
                                             'Married-civ-spouse'],
                          'occupation': ['Other-service',
                                         'Adm-clerical',
                                         'Protective-serv',
                                         'Tech-support',
                                         'Sales',
                                         'Exec-managerial',
                                         'Transport-moving',
                                         'Other-service',
                                         'Prof-specialty'],
                          'relationship': ['Not-in-family',
                                           'Unmarried',
                                           'Not-in-family',
                                           'Husband',
                                           'Not-in-family',
                                           'Other-relative',
                                           'Husband',
                                           'Unmarried',
                                           'Husband'],
                          'race': ['White',
                                   'Black',
                                   'White',
                                   'White',
                                   'White',
                                   'White',
                                   'White',
                                   'White',
                                   'White'],
                          'sex': ['Male',
                                  'Female',
                                  'Female',
                                  'Male',
                                  'Male',
                                  'Female',
                                  'Male',
                                  'Female',
                                  'Male']})
    labels = np.array([False, False, False, True,
                       False, False, True, False, True])
    return feats, labels


@pytest.fixture(scope='session')
def rf_config(request):
    """
    fixture for rf config
    """
    rf_conf = request.config.option.rf_config
    if rf_conf is None:
        pytest.fail("You must provide a rf config")
    with open(rf_conf) as file_name:
        rf_conf = json.load(file_name)
    return rf_conf
