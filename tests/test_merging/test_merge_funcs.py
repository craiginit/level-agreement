import pandas as pd
import pytest
from level_agreement.merging.merging_df import merge_df, compare_prediction_columns, determine_change
from level_agreement.upload.file_upload import load_file


@pytest.mark.parametrize('file_path_1, file_path_2', [
    ('./model_predictions_latest.csv',
     './human_predictions_latest.csv')
])
def test_merging_func(file_path_1, file_path_2):
    """
    This test function checks for correct & non-null columns in the merged DataFrames

    """
    # load DataFrames
    df1 = load_file(file_path_1)
    df2 = load_file(file_path_2)

    # call merge_df function
    merged_df = merge_df(df1, df2)

    # Check if 'prediction_x' and 'prediction_y' columns do not contain square brackets []
    has_square_brackets_x = merged_df['prediction_x'].str.contains(r'\[.*\]')
    has_square_brackets_y = merged_df['prediction_y'].str.contains(r'\[.*\]')

    # check if columns are present in merged_df
    assert "id" in merged_df
    assert "comment" in merged_df
    assert "prediction_x" in merged_df
    assert "prediction_y" in merged_df

    # check if prediction columns are null
    assert merged_df['prediction_x'].notnull().all()
    assert merged_df['prediction_y'].notnull().all()

    # check if prediction columns have square brackets []
    assert not has_square_brackets_x.any()
    assert not has_square_brackets_y.any()


@pytest.mark.parametrize('file_path_1, file_path_2', [
    ('./model_predictions_latest.csv', './comment_columns_with_ids.csv')
])
def test_merging_error_raised(file_path_1, file_path_2):
    """
    This test function checks if the correct error message is raised when merge fails.
    """
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)
    error_message = "Merge Not Successful, please check your input files"

    with pytest.raises(ValueError) as exec_info:
        df_merged = merge_df(df1, df2)
    assert str(exec_info.value) != error_message


@pytest.mark.parametrize('file_path_1, file_path_2', [
    ('./human_predictions_latest.csv',
     './model_predictions_latest.csv')
])
def test_matching_predicted_columns_match(file_path_1, file_path_2):
    """
    This test function checks to see if 2 prediction columns match

    """
    df1 = load_file(file_path_1)
    df2 = load_file(file_path_2)

    merged_file = compare_prediction_columns(df1, df2)

    # Check if the condition is met in compare_prediction_columns function
    human_predictions = set(merged_file["prediction_x"])
    model_predictions = set(merged_file["prediction_y"])

    # Assert to check if there are any differences in prediction columns
    assert human_predictions.difference(model_predictions)
    assert "id" in merged_file.columns
    assert "comment" in merged_file.columns
    assert "differences" in merged_file.columns


@pytest.mark.parametrize('file_path_1, expected_result', [
    ('./prediction_differences.csv', 'Both Predictions are Empty'),
])
def test_determine_change_both_empty_predictions(file_path_1, expected_result):
    """
    Test the determine_change function for removed themes.
    """
    # grab all rows with "differences" = True
    differences = True

    # labels placed in a list for comparison
    prediction_x = ''
    prediction_y = ''

    # Call the determine_change function with the extracted data
    result = determine_change(differences, prediction_x, prediction_y)
    if not prediction_x and not prediction_y:
        assert result == expected_result


@pytest.mark.parametrize('file_path_1, expected_result', [
    ('./prediction_differences.csv', 'Missing Labels in Prediction_X'),
])
def test_determine_change_x_empty_predictions(file_path_1, expected_result):
    """
    Test the determine_change function for removed themes.
    """
    # grab all rows with "differences" = True
    differences = True

    # labels placed in a list for comparison
    prediction_x = ''
    prediction_y = 'label.1'

    # Call the determine_change function with the extracted data
    result = determine_change(differences, prediction_x, prediction_y)
    if not prediction_x and prediction_y:
        assert result == expected_result


@pytest.mark.parametrize('file_path_1, expected_result', [
    ('./prediction_differences.csv', 'Missing Labels in Prediction_Y'),
])
def test_determine_change_y_empty_predictions(file_path_1, expected_result):
    """
    Test the determine_change function for removed themes.
    """
    # grab all rows with "differences" = True
    differences = True

    # labels placed in a list for comparison
    prediction_y = ''
    prediction_x = 'label.1'

    # Call the determine_change function with the extracted data
    result = determine_change(differences, prediction_x, prediction_y)
    if prediction_x and not prediction_y:
        assert result == expected_result


@pytest.mark.parametrize('file_path_1, expected_result', [
    ('./prediction_differences.csv', 'Themes(s) Added'),
])
def test_determine_change_added(file_path_1, expected_result):
    """
    Test the determine_change function for removed themes.
    """

    # grab all rows with "differences" = True
    differences = True

    # labels placed in a list for comparison
    prediction_x = "label.1, label.2"
    prediction_y = "label.1, label.2, label.3"

    # Call the determine_change function with the extracted data
    result = determine_change(differences, prediction_x, prediction_y)
    assert result == expected_result


@pytest.mark.parametrize('file_path_1, expected_result', [
    ('./prediction_differences.csv', 'Theme(s) Removed'),
])
def test_determine_change_removed(file_path_1, expected_result):
    """
     Test the determine_change function for equal number of themes.
    """
    # grab all rows with "differences" = True
    differences = True

    # labels placed in a list for comparison
    prediction_x = "label.1, label.2"
    prediction_y = "label.1"

    # Call the determine_change function with the extracted data
    result = determine_change(differences, prediction_x, prediction_y)
    assert result == expected_result


@pytest.mark.parametrize('file_path_1, expected_result', [
    ('./prediction_differences.csv', 'No Change'),
])
def test_determine_change_same_length(file_path_1, expected_result):
    """
     Test the determine_change function for removed themes.
    """
    # grab all rows with "differences" = True
    differences = True

    # labels placed in a list for comparison
    prediction_x = "label.1"
    prediction_y = "label.1"

    # Call the determine_change function with the extracted data
    result = determine_change(differences, prediction_x, prediction_y)
    assert result == expected_result


@pytest.mark.parametrize('file_path_1, expected_result', [
    ('./prediction_differences.csv', 'Sentiment Changed'),
])
def test_determine_change_sentiment(file_path_1, expected_result):
    """
     Test the determine_change function for removed themes.
    """
    # grab all rows with "differences" = True
    differences = True

    # labels placed in a list for comparison
    prediction_x = "label.1"
    prediction_y = "label.0"

    # Call the determine_change function with the extracted data
    result = determine_change(differences, prediction_x, prediction_y)
    assert result == expected_result
