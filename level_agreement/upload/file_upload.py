import pandas as pd


def load_file(file_path):
    """
    This function reads a CSV file.
    Args:
        file_path (str): A file path for the CSV file.
    Returns:
        Returns a DataFrame.
    """
    df = pd.read_csv(file_path)
    if not df.empty:
        return df
    else:
        raise AssertionError("DataFrame is empty")


def check_file_path_string(file_path):
    """
    This function checks if the loaded file ends with csv
    """
    if file_path.lower().endswith(".csv"):
        return True
    else:
        raise ValueError("File path should end with '.csv'")
