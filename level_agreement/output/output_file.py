import pandas as pd
from sklearn.metrics import cohen_kappa_score
from pathlib import Path


def predictions_metrics(df, prediction_column_x='prediction_x', prediction_column_y='prediction_y'):
    """
    Analyze predictions and calculate overall metrics.

    Args:
        df (pd.DataFrame): DataFrame containing prediction data.
        prediction_column_x (str): The column for the first set of predictions.
        prediction_column_y (str): The column for the second set of predictions.

    Returns:
        dict: A dictionary containing overall metrics.
    """
    # Check if the DataFrame is not None
    if df is None:
        return {"Error": "DataFrame is None"}

    # Check if the 'comment' column exists in the DataFrame
    if 'comment' not in df.columns:
        return {"Error": "DataFrame does not contain 'comment' column"}

    # Compute total predictions
    total_comments = df['comment'].count()

    # Compute the number of new predictions in 'prediction_y' compared to 'prediction_x'
    differences_in_prediction = (df[prediction_column_y] != df[prediction_column_x]).sum()

    # calculate overall metrics
    total_differences = differences_in_prediction
    total_correct_predictions = len(df) - differences_in_prediction

    # matching predictions from total predictions
    overall_accuracy = total_correct_predictions / len(df)

    # Calculate Kappas Sore
    kappa_score = cohen_kappa_score(df[prediction_column_y], df[prediction_column_x])

    # Dictionary to store the overall metrics
    metrics = {
        "Total Predictions": total_comments,
        "Total Correct Predictions": total_correct_predictions,
        "Total Prediction Differences": total_differences,
        "Overall Accuracy": overall_accuracy,
        "Kappas Score": kappa_score,
    }

    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame([metrics])

    return metrics_df


def rename_columns(dataframe, ground_truth_file, comparison_file):
    """
    Rename the prediction columns to file's name
    """

    # Get the file names as strings with .name convention
    df_file_name = Path(ground_truth_file.name)
    comparison_file_name = Path(comparison_file.name)

    # Check if the columns "prediction_x" and "prediction_y" exist
    if "prediction_x" in dataframe.columns and "prediction_y" in dataframe.columns:
        # Rename the columns to the file names
        dataframe.rename(columns={"prediction_x": df_file_name, "prediction_y": comparison_file_name},
                         inplace=True)
    else:
        print("Columns 'prediction_x' and 'prediction_y' not found in the dataframe.")

    return dataframe
