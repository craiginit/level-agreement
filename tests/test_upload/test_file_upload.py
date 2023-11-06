import pandas as pd
import pytest

from level_agreement.upload.file_upload import load_file, check_file_path_string


@pytest.mark.parametrize('file_path', ['./latest_human.csv'])
def test_load_file_type(file_path):
    df = load_file(file_path)
    assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize('file_path', ['./latest_model.csv'])
def test_empty_file(file_path):
    """
    This test function checks if DF is empty
    """
    df = load_file(file_path)
    assert not df.empty


@pytest.mark.parametrize('file_path', ['./empty_df.csv'])
def test_empty_upload_file_error_message(file_path):
    """
    This function checks if the correct error is raised
    """
    error_message = 'DataFrame is empty'
    with pytest.raises(AssertionError) as exec_info:
        df = load_file(file_path)
    assert str(exec_info.value) == error_message


@pytest.mark.parametrize('file_path', ['./craig_2023cv.pdf'])
def test_upload_type_error_message(file_path):
    """
    This function checks if the correct error is raised
    """
    error_message = 'The loaded file is not a Dataframe'
    with pytest.raises(ValueError) as exec_info:
        df = load_file(file_path)
        assert error_message == exec_info


@pytest.mark.parametrize('file_path', ['./latest_model.csv'])
def test_file_path_string(file_path):
    """
    This test function checks to see if the file path string ends
    with .csv
    """
    file_string = check_file_path_string(file_path)
    assert file_string


@pytest.mark.parametrize('file_path', ['./craig_2023cv.pdf'])
def test_file_path_string_error_message(file_path):
    """
    This test function checks to see if the file path string ends
    with .csv
    """
    error_message = "File path should end with '.csv'"
    with pytest.raises(ValueError) as exec_info:
        check_file_path_string(file_path)
    assert str(exec_info.value) == error_message



