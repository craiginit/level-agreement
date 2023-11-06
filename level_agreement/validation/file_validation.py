import pandas as pd


def validate_file_column(dataframe):
    """
    This function checks to see if columns are present
    arg:
         str() takes a file path for the file
    return:
         returns a dataframe
    """
    columns_to_check = ["id", "comment", "prediction"]
    for column in columns_to_check:
        if column not in dataframe.columns:
            raise ValueError(f'File is missing the {column} column')
    return dataframe


def validate_empty_file(dataframe):
    """
    This function validates whether a DataFrame
    is empty or not
    """

    # Check to see if DataFrame is empty
    if len(dataframe) > 0:
        return dataframe
    else:
        raise AssertionError("DataFrame is empty")


def validate_empty_rows(dataframe):
    """
    This function validates whether a DataFrame
    has NaN cells
    """
    columns_to_check = ["id", "comment", "prediction"]
    # Check to see if any of rows are n/a
    empty_cells = dataframe[columns_to_check].isna().any().any()

    if empty_cells:
        try:
            dataframe[columns_to_check] = dataframe[columns_to_check].fillna("[]")
            return dataframe
        except Exception as e:
            raise Exception("DataFrame has empty cells" + str(e))
    else:
        return dataframe


def validate_ids(dataframe1, dataframe2):
    # to test both ways to ensure that there are no ids unique to one specific df, all to check that all ids match
    if not dataframe1['id'].isin(dataframe2['id']).all() or not dataframe2['id'].isin(dataframe1['id']).all():
        # Stores the unique ids
        unique_ids_df1 = dataframe1[~dataframe1['id'].isin(dataframe2['id'])]
        unique_ids_df2 = dataframe2[~dataframe2['id'].isin(dataframe1['id'])]

        # Drop the unique IDs from each dataframe
        dataframe1 = dataframe1[~dataframe1['id'].isin(unique_ids_df1['id'])]
        dataframe2 = dataframe2[~dataframe2['id'].isin(unique_ids_df2['id'])]

        count_unique = len(unique_ids_df1) + len(unique_ids_df2)
        print(f"{count_unique} unique IDs were found in the files, these have been removed.")

    return dataframe1, dataframe2
