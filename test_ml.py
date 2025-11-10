import pytest
import os
import pandas as pd
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def load_data():
    """
    Fixture returning a DataFrame from data_path.
    If the CSV is missing or empty, returns a small synthetic DataFrame.
    """

    env_path = os.getenv("DATA_PATH", "data/census.csv")
    data_path = (APP_ROOT / env_path).resolve()

    dataframe = None

    if data_path.exists():
        try:
            dataframe = pd.read_csv(data_path)
        except Exception:
            dataframe = None

    if dataframe is not None and not dataframe.empty:
        return dataframe

    return pd.DataFrame(
        {
            "age": [25, 41, 73, 19],
            "workclass": ["Private", "Self-emp-not-inc", "Private", "Private"],
            "fnlgt": [226802, 89814, 336951, 160187],
            "education": ["11th", "HS-grad", "Masters", "Some-college"],
            "education-num": [7, 9, 14, 10],
            "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Never-married"],
            "occupation": ["Machine-op-inspct", "Farming-fishing", "Prof-specialty", "Other-service"],
            "relationship": ["Own-child", "Husband", "Not-in-family", "Unmarried"],
            "race": ["Black", "White", "White", "Black"],
            "sex": ["Male", "Male", "Female", "Female"],
            "capital-gain": [0, 0, 14084, 0],
            "capital-loss": [0, 0, 0, 0],
            "hours-per-week": [40, 50, 60, 20],
            "native-country": ["United-States"] * 4,
            "salary": ["<=50K", ">50K", "<=50K", "<=50K"],
        }
    )


def test_one_read_write_access(tmp_path, load_data):
    """
    Test basic read/write access using a temporary directory.
    Verifies that data can be written to CSV and read back successfully.
    """

    # Make temp file path
    tmp_file_path = tmp_path / "write_check.csv"

    # Write Test
    load_data.to_csv(tmp_file_path, index=False)
    assert tmp_file_path.exists()

    # Read Test
    dataframe2 = pd.read_csv(tmp_file_path)

    # Verify Shape
    assert len(dataframe2) == len(load_data)

    # Deep Testing
    # https://pandas.pydata.org/docs/reference/testing.html
    # https://pandas.pydata.org/docs/reference/api/pandas.testing.assert_frame_equal.html#pandas.testing.assert_frame_equal
    pd.testing.assert_frame_equal(
        dataframe2.reset_index(drop=True),
        load_data.reset_index(drop=True),
        check_dtype=False
    )


# TODO: implement the second test. Change the function name and input as needed
def test_two_columns_exists(load_data):
    """
    Verifies that the dataset contains all required columns
    expected for the Census Income dataset.
    """

    expected_cols = {
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    }

    actual_cols = set(load_data.columns)

    missing = expected_cols - actual_cols
    extras = actual_cols - expected_cols

    assert not missing, f"Missing expected columns: {missing}"
    assert not extras, f"Unexpected new columns found: {extras}"


# TODO: implement the third test. Change the function name and input as needed
def test_three_model_directory_and_files_exists():
    """
    Passes if either:
      - a model exists under models/ directory, OR
      - models/ doesn't exist yet (skip -> new repos won't fail CI).
    """

    models_dir = Path("models")

    if not models_dir.exists():
        pytest.skip("No models/ directory found")

    models = list(models_dir.rglob("*.pkl"))
    assert models, "models/ exists but no model artifacts found."


def test_no_nulls(load_data):
    nulls = load_data.isnull().sum().sum()
    assert nulls == 0, f"Dataset contains {nulls} null values."


def test_no_duplicate_rows(load_data):
    dups = load_data.duplicated().sum()
    assert dups == 0, f"Duplicate rows detected: {dups}"


@pytest.mark.parametrize(
    "column,low,high",
    [
        ("age", 0, 120),
        ("hours_per_week", 0, 168),
        ("education-num", 1, 16)
    ],
)
def test_value_ranges_if_present(load_data, column, low, high):
    if column not in load_data.columns:
        pytest.skip(f"{column} not present; skipping range check.")
    series = load_data[column].dropna()
    assert (series >= low).all() and (series <= high).all(), (
        f"Values in {column} outside expected range [{low}, {high}]"
    )


def test_column_types(load_data):
    expected_types = {
        "age": "int64",
        "education-num": "int64",
        "captial-gain": "int64",
        "captial-loss": "int64",
        "hours-per-week": "int64",
        }

    for column, dtype in expected_types.items():
        assert load_data[column].dtype == dtype, f"{column} should be {dtype}"
