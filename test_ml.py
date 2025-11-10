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
            "id": [1, 2, 3, 4],
            "age": [25, 41, 73, 19],
            "salary": [55000.0, 92000.0, 37500.0, 78000.0],
            "name": ["a", "b", "c", "d"],
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
