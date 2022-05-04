import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):
    expected_colums = [
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'salary'
    ]
    these_columns = data.columns.values
    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)

def test_sex_names(data):
    known_names = ["Male","Female"]
    sex_names = set(data['sex'].unique())
    # Unordered check
    assert set(known_names) == set(sex_names)

def test_sex_names(data):
    known_names = ["Male","Female"]
    sex_names = set(data['sex'].unique())
    # Unordered check
    assert set(known_names) == set(sex_names)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


########################################################
# Implement here test_row_count and test_price_range   #
########################################################

def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000

def test_price_range(data,min_price,max_price):
    assert data['price'].dropna().between(min_price, max_price).all(), (
            f"Column Price failed the test. Should be between {min_price} and {max_price}, "
            f"instead min={data['price'].min()} and max={data['price'].max()}"
        )

