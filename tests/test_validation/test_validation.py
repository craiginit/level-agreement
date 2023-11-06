import pandas as pd
import pytest
from level_agreement.validation.file_validation import validate_file_column, \
    validate_empty_rows
from level_agreement.upload.file_upload import load_file


@pytest.mark.parametrize('file_path', ['./latest_model.csv'])
def test_validation_columns(file_path):
    """
    This test function validates if the 'prediction' column is present
    """
    file = load_file(file_path)
    df = validate_file_column(file)
    columns_to_check = ["id", "comment", "prediction"]
    assert columns_to_check[0] in df
    assert columns_to_check[1] in df
    assert columns_to_check[2] in df

@pytest.mark.parametrize('file_path', ['./no_id_column.csv'])
def test_validation_columns_id_error_message(file_path):
    """
    This test function checks if the right error message is shown if
    'prediction' column is not present
    """
    file = load_file(file_path)
    error_message = 'File is missing the id column'
    with pytest.raises(ValueError) as exec_info:
        validate_file_column(file)
    assert error_message == str(exec_info.value)

@pytest.mark.parametrize('file_path', ['./no_comment_column.csv'])
def test_validation_columns_comment_error_message(file_path):
    """
    This test function checks if the right error message is shown if
    'comment' columns is not present
    """
    file = load_file(file_path)
    error_message = 'File is missing the comment column'
    with pytest.raises(ValueError) as exec_info:
        validate_file_column(file)
    assert error_message == str(exec_info.value)


@pytest.mark.parametrize('file_path', ['./no_predictions_column.csv'])
def test_validation_columns_prediction_error_message(file_path):
    """
    This test function checks if the right error message is shown if
    'prediction' column is not present
    """
    file = load_file(file_path)
    error_message = 'File is missing the prediction column'
    with pytest.raises(ValueError) as exec_info:
        validate_file_column(file)
    assert error_message == str(exec_info.value)


@pytest.mark.parametrize('file_path', ['./latest_model.csv'])
def test_empty_rows(file_path):
    """
    This test function checks for any empty cells or null in a df
    """
    # Load input DataFrame
    df = load_file(file_path)

    # Check if DataFrame is empty
    results_df = validate_empty_rows(df)

    # Use assert to check if the DataFrame is empty
    assert not results_df.empty


@pytest.mark.parametrize('file_path', ['./empty_row.csv'])
def test_empty_rows_error_message(file_path):
    """
    This function tests to see if the correct error message is raised when conditions
    of validate_empty_rows function are not met
    """
    error_message = "DataFrame has empty cells"
    with pytest.raises(Exception) as exec_info:
        df = validate_empty_rows(file_path)
        assert exec_info == error_message


@pytest.mark.parametrize('file, comparison_file', [
    ('./latest_model.csv', './latest_human.csv'), ])
def test_validate_ids(file, comparison_file):
    dataframe1 = load_file(file)
    dataframe2 = load_file(comparison_file)
    assert dataframe1['id'].isin(dataframe2['id']).all() and dataframe2['id'].isin(dataframe1['id']).all()
