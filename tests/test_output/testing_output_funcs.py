import pandas
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import pytest
from level_agreement.upload.file_upload import load_file
from level_agreement.output.output_file import predictions_metrics
from level_agreement.output.output_file import rename_columns


# Your test function
@pytest.mark.parametrize('file_path_1', ['./prediction_differences (1).csv'])
def test_total_prediction_metrics(file_path_1, prediction_column_x='prediction_x', prediction_column_y='prediction_y'):
    """
    Test function to validate the accuracy of total predictions calculated by the predictions_metrics function.

    Raises:
        AssertionError: If the total predictions in metrics_df do not match the length of the input DataFrame.

    """
    # Load DF
    df = load_file(file_path_1)

    # Call prediction metrics function and check if it raises an error
    with pytest.raises(AssertionError):
        metrics_df = predictions_metrics(df, prediction_column_x, prediction_column_y)
        assert metrics_df['Total Predictions'].iloc[0] != len(df)


@pytest.mark.parametrize('file_path_1', ['./prediction_differences (1).csv'])
def test_total_correct_prediction_metrics(file_path_1, prediction_column_x='prediction_x',
                                          prediction_column_y='prediction_y'):
    """
    Test function to validate the accuracy of total predictions
    calculated by the predictions_metrics function.
    """
    # Load DF
    df = load_file(file_path_1)

    # Call prediction metrics function
    metrics_df = predictions_metrics(df, prediction_column_x, prediction_column_y)

    # Get the expected 'Total Correct Predictions' from metrics_df row
    expected_total_correct_predictions = metrics_df['Total Correct Predictions'].iloc[0]

    with pytest.raises(AssertionError):
        # Compute total correct predictions using the differences in prediction columns
        differences_in_prediction = (df[prediction_column_y] != df[prediction_column_x]).sum()

        # Total Correct Prediction computation
        total_correct_computed_predictions = len(df) - differences_in_prediction

        # Assertion
        assert expected_total_correct_predictions != total_correct_computed_predictions


@pytest.mark.parametrize('file_path_1', ['./prediction_differences (1).csv'])
def test_total_prediction_differences(file_path_1, prediction_column_x='prediction_x',
                                      prediction_column_y='prediction_y'):
    """
    Test function to validate Total Prediction Differences metrics
    """
    # Load DF
    df = load_file(file_path_1)

    # Call prediction metrics function
    metrics_df = predictions_metrics(df, prediction_column_x, prediction_column_y)

    # Get the expected 'Total Prediction Differences' from metrics_df row
    expected_total_prediction_differences = metrics_df['Total Prediction Differences'].iloc[0]

    with pytest.raises(AssertionError):
        # Compute total correct predictions using the differences in prediction columns
        differences_in_prediction = (df[prediction_column_y] != df[prediction_column_x]).sum()

        # Assertion
        assert expected_total_prediction_differences != differences_in_prediction


@pytest.mark.parametrize('file_path_1', ['./prediction_differences (1).csv'])
def test_overall_accuracy_metrics(file_path_1, prediction_column_x='prediction_x', prediction_column_y='prediction_y'):
    """
    Test function to validate Overall Accuracy metrics
    """
    # Load DF
    df = load_file(file_path_1)

    # Call prediction metrics function
    metrics_df = predictions_metrics(df, prediction_column_x, prediction_column_y)

    # Compute the number of new predictions in 'prediction_y' compared to 'prediction_x'
    differences_in_prediction = (df[prediction_column_y] != df[prediction_column_x]).sum()

    # Get the 'Overall Accuracy' from metrics_df row
    expected_overall_accuracy = metrics_df['Overall Accuracy'].iloc[0]

    with pytest.raises(AssertionError):
        # Calculate overall metrics
        total_correct_predictions = len(df) - differences_in_prediction
        total_differences = len(df)

        # overall accuracy computation
        overall_accuracy = total_correct_predictions / total_differences

        # Assertion
        assert expected_overall_accuracy != overall_accuracy


@pytest.mark.parametrize('file_path_1', ['./prediction_differences (1).csv'])
def test_kappas_score_metrics(file_path_1, prediction_column_x='prediction_x', prediction_column_y='prediction_y'):
    """
    Test function to validate Kappas Score
    """
    # Load DF
    df = load_file(file_path_1)

    # Call prediction metrics function
    metrics_df = predictions_metrics(df, prediction_column_x, prediction_column_y)

    # Get the 'Kappas Score' from metrics_df
    expected_kappas_score = metrics_df['Kappas Score'].iloc[0]

    with pytest.raises(AssertionError):
        # Calculate Kappas Score
        kappa_score = cohen_kappa_score(df[prediction_column_y], df[prediction_column_x])
        # Assertion
        assert expected_kappas_score != kappa_score


@pytest.mark.parametrize('dataframe, ground_truth_file, comparison_file', [
    ('./prediction_differences (1).csv', './human_predictions_latest.csv', './model_predictions_latest.csv')
])
def test_rename_column(dataframe, ground_truth_file, comparison_file):

    # load files
    df = load_file(dataframe)
    df1 = load_file(ground_truth_file)
    df2 = load_file(comparison_file)

    # Call the rename_columns function with the parameters
    renamed_df = rename_columns(df, df1, df2)

    # Check if the columns are renamed correctly
    assert df1 in renamed_df
    assert df2 in renamed_df
    assert 'prediction_x' not in renamed_df
    assert 'prediction_y' not in renamed_df