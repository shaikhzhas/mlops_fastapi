"""
GitHub Action pytests
"""
import pytest
import pandas as pd


@pytest.fixture(scope='session')
def data():
    """
    fixture for data
    """
    df_test = pd.read_csv('test/sample_data.csv')
    return df_test


def test_column_names(data):
    """
    test column names
    """
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


def test_column_types(data):
    """
    test column types
    """
    required_columns = {
        "age": pd.api.types.is_integer_dtype,
        "workclass": pd.api.types.is_string_dtype,
        "fnlgt": pd.api.types.is_integer_dtype,
        "education": pd.api.types.is_string_dtype,
        "education-num": pd.api.types.is_integer_dtype,
        "marital-status": pd.api.types.is_string_dtype,
        "occupation": pd.api.types.is_string_dtype,
        "relationship": pd.api.types.is_string_dtype,
        "race": pd.api.types.is_string_dtype,
        "sex": pd.api.types.is_string_dtype,
        "capital-gain": pd.api.types.is_integer_dtype,
        # This is integer, not float as one might expect
        "capital-loss": pd.api.types.is_integer_dtype,
        "hours-per-week": pd.api.types.is_integer_dtype,
        "native-country": pd.api.types.is_string_dtype,
        "salary": pd.api.types.is_bool_dtype
    }
    for col_name, format_verification_funct in required_columns.items():
        assert format_verification_funct(
            data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_sex_names(data):
    """
    test sex names
    """
    known_names = ["Male", "Female"]
    sex_names = set(data['sex'].unique())
    # Unordered check
    assert set(known_names) == set(sex_names)


def test_race_names(data):
    """
    test race names
    """
    known_names = [
        'White',
        'Black',
        'Asian-pac-islander',
        'Amer-indian-eskimo',
        'Other']
    race_names = set(data['race'].unique())
    # Unordered check
    assert set(known_names) == set(race_names)


def test_education_names(data):
    """
    test education names
    """
    known_names = [
        'Hs-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th',
        'Assoc-acdm', '10th', '7th-8th', 'Prof-school', '9th', '12th',
        'Doctorate', '5th-6th', '1st-4th', 'Preschool'
    ]
    education_names = set(data['education'].unique())
    # Unordered check
    assert set(known_names) == set(education_names)


def test_column_ranges(data):
    """
    test column ranges
    """
    ranges = {
        "age": (17, 100),
        "education-num": (1, 16)
    }
    for col_name, (minimum, maximum) in ranges.items():
        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={data[col_name].min()} and max={data[col_name].max()}"
        )


def test_row_count(data):
    """
    test row count
    """
    assert 15000 < data.shape[0] < 1000000
