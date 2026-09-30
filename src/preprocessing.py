import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the crop recommendation CSV file.

    Parameters
    ----------
    csv_path: str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    df = pd.read_csv(csv_path)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split dataframe into training and test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=["label"])
    y = df["label"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_preprocessing_pipeline():
    """Create a preprocessing pipeline that scales numeric features.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])
    return pipeline
