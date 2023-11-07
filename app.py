import streamlit as st
import pandas as pd
import logging


from level_agreement.validation.file_validation import validate_file_column, validate_empty_file, validate_empty_rows, \
    validate_ids
from level_agreement.merging.merging_df import merge_df, compare_prediction_columns
from level_agreement.output.output_file import predictions_metrics, rename_columns

# Set up logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    st.title("Agreement Checker")
    st.markdown(f""":sparkles: This tool compares two sets of processed data, 'X' serving as the ground truth and 'Y' as
     the comparison set, based on columns [id, comments, prediction]. It creates a merged file to assess label 
     differences. Utilizing computed metrics, it should be easier to identify the more accurate and aligned dataset 
     following our expectations. :sparkles:""")

    st.write('\n\n')

    ground_truth_file = None
    comparison_file = None

    if 'compare_data' not in st.session_state:
        st.session_state.compare_data = False

    ground_truth_file = st.sidebar.file_uploader("Prediction X: Ground-Truth Dataset", type=["csv"])
    comparison_file = st.sidebar.file_uploader("Prediction Y: Comparison Dataset", type=["csv"])
    compare_data_button = st.sidebar.button("Compare Data")

    if compare_data_button and ground_truth_file is not None and comparison_file is not None:
        # Returns both csv files as dataframes
        gt_df = pd.read_csv(ground_truth_file)
        comp_df = pd.read_csv(comparison_file)
        dataframes = [gt_df, comp_df]

        # To check if both dataframes are empty and contains any empty rows
        logging.info(f":hourglass: Validating files to check for empty file, empty rows and column names")
        st.write(f":hourglass: Validating files to check for empty file, empty rows and column names")
        for df in dataframes:
            empty_df_check = validate_empty_file(df)
            empty_rows_check = validate_empty_rows(df)
            file_column_check = validate_file_column(df)

        logging.info(f"Files Validated - All Good!")
        st.write(f":white_check_mark: Files Validated - All Good!")

        logging.info(f"Checking for mismatched IDs - both datasets need to have the same IDs present")
        st.write(f":hourglass: Checking for mismatched IDs - both datasets need to have the same IDs present")
        id_check = validate_ids(gt_df, comp_df)

        logging.info(f"Merging files - creating prediction_x and prediction_y columns")
        st.write(f":hourglass: Merging files - creating prediction_x and prediction_y columns")

        # Merging the two files into one - producing columns prediction_x and prediction_y
        merged_df = merge_df(gt_df, comp_df)

        logging.info(f"Retrieving differences in label predictions")
        st.write(f":hourglass: Retrieving differences in label predictions")
        logging.info(f":hourglass: Retrieving prediction metrics")
        st.write(f":hourglass: Retrieving prediction metrics")

        # Comparing the two dataframes to find the rows with differences in the prediction columns
        compare_files = compare_prediction_columns(gt_df, comp_df)

        # Grab prediction_differences column and compute metrics
        metrics_df = predictions_metrics(compare_files)

        # Rename prediction columns to file name
        renamed_columns = rename_columns(compare_files, ground_truth_file, comparison_file)

        # Display metrics table
        with st.expander("Metrics Table", expanded=False):
            st.table(metrics_df)

        logging.info(f"Process Complete - Please see prediction differences & metrics table below:")

        st.write(f":white_check_mark: Process Complete - Please see prediction differences below:")

        # Download metrics table
        metrics_results = metrics_df.to_csv(index=False)

        # Display Comparison table
        with st.expander("Comparison Table", expanded=False):
            st.table(compare_files)
        compare_file_results_csv = compare_files.to_csv(index=False)

        with pd.ExcelWriter('prediction_differences.xlsx') as writer:
            compare_files.to_excel(writer, index=False)

        st.download_button(
            label="Download Prediction Differences - CSV",
            data=compare_file_results_csv,
            file_name="prediction_differences.csv",
            mime="text/csv",
            key="download-csv",
        )

        st.download_button(
            label="Download Prediction Differences - XLS",
            data=open('prediction_differences.xlsx', 'rb'),
            file_name="prediction_differences.xls",
            mime="application/vnd.ms-excel",
            key="download-xls",
        )

        st.download_button(
            label="Download Metric Results - CSV",
            data=metrics_results,
            file_name="prediction_metrics.csv",
            mime="text/csv",
            key="download csv",
        )
