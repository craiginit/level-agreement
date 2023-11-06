import pandas as pd
from level_agreement.validation.file_validation import validate_ids


def merge_df(file_path_1, file_path_2):
    """
    This function merges 2 DataFrames using pandas and returns a new DataFrame
    """

    df1, df2 = validate_ids(file_path_1, file_path_2)

    # Use pandas to merge df1 and df2 on id and comment
    merged_df = df1.merge(df2, on=["id", "comment"], how='inner')

    # Check to see if merge was successful
    if merged_df.empty:
        raise ValueError("Merge Not Successful, please check your input files")

    # Remove square brackets from prediction_x and prediction_y b
    merged_df['prediction_x'] = merged_df['prediction_x'].str.strip('[]').str.replace("'", "")
    merged_df['prediction_y'] = merged_df['prediction_y'].str.strip('[]').str.replace("'", "")

    return merged_df


def compare_prediction_columns(file_path_1, file_path_2):
    """
        Compares prediction columns between two data files and identifies differences.
        Args:
            file_path_1 (str): Path to the first data file.
            file_path_2 (str): Path to the second data file.

        Returns:
            pd.DataFrame: DataFrame with differences identified in the 'prediction_differences' column and
            type of action made.
        """

    # Merge the DataFrames using merge_df function
    merged_file = merge_df(file_path_1, file_path_2)

    # Calculate the differences and store them in a 'differences' column
    merged_file['differences'] = (merged_file["prediction_x"] != merged_file["prediction_y"])

    # Apply get_label_differences function to get the differences for each row
    merged_file['prediction_differences'] = merged_file.apply(
        lambda row: get_label_differences(row['prediction_x'], row['prediction_y']) if row['differences'] else "'NaN'",
        axis=1)

    # Apply determine_change function to create a new 'type' column
    merged_file['type'] = merged_file.apply(
        lambda row: determine_change(row['differences'], row['prediction_x'], row['prediction_y']),
        axis=1
    )

    # Logic to check if there are any differences
    if merged_file['differences'].any():
        return merged_file

    else:
        print("Great, Columns match!")


def get_label_differences(prediction_x, prediction_y):
    """
    Cleans and formats the label differences between two prediction strings.

    Args:
        prediction_x (str): First prediction string.
        prediction_y (str): Second prediction string.

    Returns:
        str: Formatted string containing differing labels in parentheses, separated by commas and spaces.
    """
    # Step 1: Cleaning the predictions
    cleaned_x = set(label.strip().lower() for label in prediction_x.split(','))
    cleaned_y = set(label.strip().lower() for label in prediction_y.split(','))

    # Find label differences using set subtraction: Labels in prediction_y but not in prediction_x
    differences_x_to_y = set(cleaned_x - cleaned_y)
    differences_y_to_x = set(cleaned_y - cleaned_x)

    # Determine if there are any differences (Added or Removed)
    if differences_y_to_x:
        formatted_differences = ", ".join(f"{label}" for label in differences_y_to_x)
        return formatted_differences
    elif differences_x_to_y:
        formatted_differences = ", ".join(f"{label}" for label in differences_x_to_y)
        return formatted_differences
    else:
        return 'NaN'


def determine_change(differences, prediction_x, prediction_y):
    """
        Determine the type of change between two sets of labels.

        Args:
            differences (bool): True if there are differences, False if not.
            prediction_x (str): First set of labels.
            prediction_y (str): Second set of labels.

        Returns:
            str: One of the following values: "Added," "Removed,", "Changed,", "Missing Predictions", "Sentiment Change" or "No Change."
        """
    # Split the 'prediction_x' and 'prediction_y' strings into lists of labels
    labels_x = [label.strip() for label in prediction_x.split(',')]
    labels_y = [label.strip() for label in prediction_y.split(',')]

    # Check if both 'prediction_x' and 'prediction_y' are empty
    if not prediction_x and not prediction_y:
        return "Both Predictions are Empty"
    # Check if 'prediction_x' is empty but 'prediction_y' is not & vice-versa
    elif not prediction_x and prediction_y:
        return f"Missing Labels in Prediction_X"
    elif prediction_x and not prediction_y:
        return f"Missing Labels in Prediction_Y"

    # Check if there are differences
    elif differences:
        # Compare the labels directly
        if set(labels_x) == set(labels_y):
            return "No Change"
        # Check if 'labels_y' is a subset of 'labels_x' (some labels were removed)
        elif set(labels_y).issubset(set(labels_x)):
            return "Theme(s) Removed"
        # Check if 'labels_x' is a subset of 'labels_y' (some labels were added)
        elif set(labels_x).issubset(set(labels_y)):
            return "Themes(s) Added"
        else:
            return "Sentiment Changed"
    else:
        return "No Change"
