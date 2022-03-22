# Daniel Zhu
# March 4th, 2021
# Function to perform various pre-processing operations on multivariate systems serology data.
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path


# Import main class for testing:
# from Systems_Serology import SystemsSerology

class SystemsSerologyPreprocessing():
    def __init__(self, data, second_dataset=None, ignore_index=True, output_basename="processed_dataset"):
        '''
        This class will have one required argument if a class is instantiated; either the path to the dataset to
        perform analysis on (which will be read and converted to a dataframe) or the dataset itself, given as a dataframe.
        :param data: The path to the dataset or the dataset itself (given as a dataframe). If provided the path (a
        string argument), pd.read_csv() will be used to read from the path to a dataframe object.
        :param second_dataset: Optionally, the path to a second dataset or the dataset itself (if given as a
        dataframe).
        :param ignore_index: If True, use the first column as indices.
        :param output_basename: The base name of the file to save to (do not specify the drive/path or file
        extension, as this will lead to redundancy). Will default to "processed_dataset" if not given.
        '''
        self.index_col = 0 if ignore_index else None
        self.dataset = pd.read_csv(data, index_col=self.index_col) if isinstance(data, str) else data
        if second_dataset is not None:
            self.second_dataset = pd.read_csv(second_dataset, index_col=self.index_col) \
                if isinstance(data, str) else second_dataset
        else:
            self.second_dataset = None
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError('Dataset must be given as a dataframe or .csv file.')
        self.output_basename = str(output_basename)

    @staticmethod
    def save_dataset(output_basename, dataset):
        '''
        Save a processed dataset to a folder within the working directory labelled "Processed Data".
        :param output_basename: The base name of the file to save to (do not specify the drive/path or file extension,
        as this will lead to redundancy).
        :param dataset: Dataset to save.
        :return: None
        '''
        # Create the "Processed" folder if it doesn't already exist:
        processed_folder_path = os.path.join(str(Path().absolute()), "Processed_Data")
        if not os.path.exists(processed_folder_path):
            os.mkdir(processed_folder_path)
        output_path = os.path.join(processed_folder_path, output_basename + ".csv")
        dataset.to_csv(output_path)

    @staticmethod
    def remove_invalid_rows(data):
        '''
        Function to remove rows/samples that contain invalid (e.g. infinity) values.
        :return: The dataset after selected rows have been removed.
        '''
        data_dropped = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
        return data_dropped

    @staticmethod
    def remove_nonnumeric(data):
        '''
        Function to remove columns of a dataframe that contain non-numeric values.
        :param data: Dataframe to edit.
        :return: Dataframe once non-numeric columns have been removed.
        '''
        numeric_dtypes = ["int", "float", "double", "long"]
        for column in data.columns:
            if not any(type in str(data[column].dtype) for type in numeric_dtypes):
                data.drop(column, axis=1, inplace=True)
        return data

    def drop_na(self, axis=0, drop_invalid=True, drop_nonnumeric=False, save=False):
        '''
        Function to drop any rows (defaults to rows, can be changed to columns) that contain NaN values.
        :param axis: Specifies whether to drop the row or column containing the NaN(s). Defaults to 0 (for rows).
        :param drop_invalid_rows: In addition to NaN, will also search for inf and -inf, and drop if found.
        :param drop_nonnumeric: Select whether to also drop non-numeric columns. Defaults to False.
        :param save: Select whether to save the processed dataset.
        :return: The dataset after selected columns have been dropped.
        '''
        if drop_invalid:
            self.dataset = self.dataset.replace([np.inf, -np.inf], np.nan)
        self.dataset = self.dataset.dropna(axis=axis, how='all')
        if drop_nonnumeric:
            numeric_dtypes = ["int", "float", "double", "long"]
            for column in self.dataset.columns:
                if not any(type in str(self.dataset[column].dtype) for type in numeric_dtypes):
                    self.dataset.drop(column, axis=1, inplace=True)
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)
        return self.dataset

    def split_dataset_and_variable(self, variable_col_id):
        '''
        Function to separate a dataset from a a column originally in that dataset, if the two initially exist within
        the same dataframe. Assumes the dataset is of the form (samples x variables).
        :param data: Dataset, given as a dataframe.
        :param variable_col_id: Name (or index) of the column corresponding to the variable of interest.
        :return: The original dataset with groups removed, and a separate dataframe containing group information.
        '''
        try:
            variable = self.dataset.iloc[:, variable_col_id].values if isinstance(variable_col_id, int) else self.dataset[
                variable_col_id].values
            variable = pd.DataFrame(data=variable, index=self.dataset.index, columns=[variable_col_id])
        except:
            raise ValueError("Outcomes_col_id invalid (must be integer or string).")
        try:
            self.dataset = self.dataset.drop(self.dataset[variable_col_id], axis=1) \
                if isinstance(variable_col_id, int) else self.dataset.drop(variable_col_id, axis=1)
        except:
            raise ValueError("Column already separate from dataframe, exiting function.")
        return self.dataset, variable

    def split_data(self, split_colname):
        '''
        Function to split a dataframe into smaller dataframes, based on the unique values in the column for which
        split_colname is the ID.
        :param split_colname: Column ID for the column that will be the basis for splitting.
        :return: List of dataframes after splitting, and a list of identifiers indicating which groups each dataframe came
        from.
        '''
        # List variable to store split dataframes in:
        split_dfs = []
        split_col = self.dataset[split_colname]
        for group in set(split_col):
            group_df = self.dataset[split_col == group]
            split_dfs.append(group_df)

        # FIGURE OUT SAVING LATER.
        return split_dfs, set(split_col)

    def subset_data(self, groups_col, to_keep, save=False):
        '''
        Function to transform a dataframe into a smaller dataframe by keeping samples that match a specific group ID.
        :param groups_col: Column of the dataframe to use to search for group information.
        :param to_keep: List of strings containing group labels to keep.
        :param save: Select whether to save the processed dataset.
        :return: None
        '''
        if not isinstance(to_keep, list): to_keep = list(to_keep)
        dataset = self.dataset.copy()
        # dataset = dataset.loc[dataset[groups_col].isin(to_keep).values]    # maybe edit later to not need an exact
        # match:
        if len(to_keep) > 1:
            to_keep = to_keep.replace(",", "|")
        dataset = dataset.loc[dataset[groups_col].str.contains(to_keep[0], regex=True)]
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=dataset)

    def concatenate(self, save=False):
        '''
        Function to create a new dataframe by joining self.data to self.second_dataset.
        :param save: Select whether to save the concatenated dataset.
        :return: None.
        '''
        if self.second_dataset is None:
            raise ValueError("No second dataset is provided, nothing to concatenate.")
        # Drop invalid samples from both datasets:
        dataset, second_dataset = self.remove_invalid_rows(self.dataset), self.remove_invalid_rows(self.second_dataset)
        # Remove_invalid_rows to fix possible dimension mismatches.
        concatenated = self.remove_invalid_rows(pd.concat([dataset, second_dataset], axis=1))
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=concatenated)

    def move_columns(self, cols_to_move, save=False):
        '''
        Function to move columns from one dataframe (given by self.dataset) to another (given by
        self.second_dataset). Note that the indices of the two dataframes must be the same for this to work.
        :param cols_to_move: Identifiers for the columns to move.
        :param save: Select whether to save the processed dataset.
        :return: None
        '''
        if self.second_dataset is None:
            raise ValueError("No dataset exists from which to move columns. Please specify the path to this dataset or "
                             "provide the dataframe.")
        # cols_to_move = [col.replace("_", " ") for col in cols_to_move]

        # Subset the rows from the second dataset to include only the samples from the first dataset (in the case
        # that rows are imbalanced/changed somehow between the two):
        # Remove duplicate rows to avoid reindexing issues:
        self.second_dataset = self.second_dataset[~self.second_dataset.index.duplicated(keep='first')]
        cols = self.second_dataset.loc[self.dataset.index, cols_to_move]
        self.dataset[cols_to_move] = cols
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)

    def move_rows(self, rows_to_move, save=False):
        '''
        Function to move rows from one dataframe (given by self.dataset) to another (given by self.second_dataset).
        :param rows_to_move: Identifiers for the rows to move.
        :param save: Select whether to save the processed dataset.
        :return: None
        '''
        if self.second_dataset is None:
            raise ValueError("No dataset exists from which to move columns. Please specify the path to this dataset or "
                             "provide the dataframe.")
        rows_to_move = [row.replace("_", " ") for row in rows_to_move]
        rows = self.second_dataset.loc[rows_to_move, self.dataset.columns]
        self.dataset = self.dataset.append(rows)
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)

    def columns_remove_spaces(self, save=False):
        '''
        Function to remove spaces from the column names of a dataframe (as SystemsSerology can be integrated with
        argparse, but argparse does not handle spaces well).
        :param save: Select whether to save the processed dataset.
        :return: None
        '''
        # Replace whitespaces with "_", but delete leading whitespaces first (if they exist)- also might as well check for
        # closing whitespaces as well.
        self.dataset.columns = self.dataset.columns.str.lstrip()
        self.dataset.columns = self.dataset.columns.str.replace(" ", "_")
        # Do this for the second dataset as well, if applicable:
        if self.second_dataset is not None:
            self.second_dataset.columns = self.second_dataset.columns.str.lstrip()
            self.second_dataset.columns = self.second_dataset.columns.str.replace(" ", "_")
            self.save_dataset(output_basename="SystemsSerologySecondDataset", dataset=self.second_dataset)
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)

    def shorten_columns(self, to_remove, save=False):
        '''
        Function to process column names, shortening columns by removing specified substrings.
        :param to_remove: List of substrings to remove, if they occur in column names.
        :param save: Select whether to save the processed dataset.
        :return: None
        '''
        if not isinstance(to_remove, list):
            to_remove = [to_remove]
        for string in to_remove:
            self.dataset.columns = [col.replace(string, "") for col in self.dataset.columns]
            # Remove empty space:
            self.dataset.columns = [col.replace(" ", "") for col in self.dataset.columns]
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)

    def remove_negatives(self, save=False):
        '''
        Function to remove negative values from serological datasets (which don't make sense in a biological context
        before normalization).
        :param save: Select whether to save the processed dataset.
        :return: None
        '''
        # First select the non-numerical columns (object columns) and remove them from the dataframe, storing into a
        # separate variable:
        non_numerics = self.dataset.select_dtypes(include=['object'])
        self.dataset = self.dataset.select_dtypes(exclude=['object'])
        self.dataset[self.dataset < 0] = 0
        # Re-insert the non-numerical columns (note that this will position all of the non-numerical columns at the front
        # of the dataframe):
        self.dataset = pd.concat([non_numerics, self.dataset], axis=1)
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)

    def control_correction(self, control_column_ids=None, control_row_id=None, save=False):
        '''
        Function to correct data values by subtracting using a control column, or
        :param control_column_ids: Labels for control columns, given as a list. Can also be a list of strings,
        where these strings are unique elements of the control column names.
        :param control_row_id: Labels for control rows, given as a list. Can also specify "All" to indicate that each
        sample has its own control measurement- these can be found in the second dataframe, and control correction
        should be performed using element-by-element division.
        :param save: Select whether to save the processed dataset.
        :return: None
        '''
        # If control row is specified, the assumption is that each measurement should be divided by a row that
        # corresponds to PBS/DMSO measurements:
        if control_row_id is not None:
            control_row_id = control_row_id.replace("_", " ")

        if control_row_id is not None and control_row_id.casefold() != "all":
            self.dataset = self.dataset.loc[:, :].div(self.dataset.loc[control_row_id, :])
            self.dataset.drop(control_row_id, axis=0, inplace=True)
        elif control_row_id is not None and control_row_id.casefold() == "all":
            self.dataset = self.dataset / self.second_dataset

        if control_column_ids is not None:
            # If only one control column is given, convert to list:
            if not isinstance(control_column_ids, list):
                control_column_ids = [control_column_ids]
            # If control column IDs are given in the form of strings unique to control columns, check columns of the
            # dataset for these strings (note if strings are an exact match, this next line will also still retrieve
            # them).
            control_column_ids = [col for col in self.dataset.columns if any(key in col for key in control_column_ids)]

            # If control column ID is given as an index (integer), convert to the corresponding name:
            if isinstance(control_column_ids, int) or all(isinstance(elem, int) for elem in control_column_ids):
                control_column_ids = self.dataset.columns[control_column_ids].values

            for control_column_id in control_column_ids:
                # Split the control column ID on either " " (empty space) or "_" (depending on whether
                # columns_remove_spaces) has been run beforehand.
                control_colname_split = control_column_id.split(
                    " ") if " " in control_column_id else control_column_id.split("_")
                # Search for "Ig" or "FcG" in the control column to find the keyword (IgG1, IgG3, FcR, etc.), then search
                # the other columns for the keyword:
                keywords = ["ig", "fcg", "fcr", "c1q"]
                key = next(
                    string for string in control_colname_split if any(keyword in string.casefold() for keyword in keywords))

                # Remove the control column from the dataframe, but save it as its own list:
                control_col = self.dataset[control_column_id].values
                self.dataset.drop(control_column_id, axis=1, inplace=True)
                # Take the first entry (theoretically an antibody name) and search the rest of the columns for that
                # antibody name (presumably, this is a titer measurement and needs to be baseline corrected). If found,
                # element-by-element subtract the control column from that column.
                for col in self.dataset.columns:
                    if key in col:
                        self.dataset[col] = self.dataset[col] - control_col

        # Remove negative values (set to 0) that may have appeared as a result of this analysis:
        self.remove_negatives()
        # And save the processed dataset:
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)

    def log10_transform(self, luminex_only=True, save=False):
        '''
        The standard in this field is to log-10 transform Luminex data.
        :param luminex_only: Select whether to apply only to Luminex (immunoglobulin titers and Fc binding) data.
        :param save: Select whether to save the processed dataset.
        :return: None
        '''
        # Luminex measurements = Ig titers and FcR binding, so check for these:
        # If luminex_only, use keywords to identify columns to log-transform. If not, use all numerical columns.
        from pandas.api.types import is_numeric_dtype
        numerical_cols = [col.casefold() for col in self.dataset.columns if is_numeric_dtype(self.dataset[col])]
        keywords = ["ig", "fc", "adcd", "c1q"] if luminex_only else numerical_cols
        for column in self.dataset.columns:
            if self.dataset[column].dtype != 'object' and any(keyword in column.casefold() for keyword in keywords):
                self.dataset[column] = self.dataset[column].apply(lambda x: np.log10(x + 1))

        # Remove negative values (set to 0) that may have appeared as a result of this analysis:
        # self.remove_negatives()
        # And save the processed dataset:
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)

    def z_score(self, make_positive=False, save=False):
        '''
        Function to convert all of the numerical columns of a dataframe to z-scores.
        :param make_positive: Boolean to shift z-scored data to strictly positive values.
        :param save: Select whether to save the processed dataset.
        :return: The modified dataframe.
        '''
        # First filter non-numeric and NaN columns:
        dataset_dropped = self.drop_na(drop_nonnumeric=True)
        # Perform the z-scoring:
        z_scored_data = (dataset_dropped - dataset_dropped.mean()) / dataset_dropped.std(ddof=1)
        # If make_positive is True, shift each column based on the minimum z-score in the column, such that values
        # are all positive.
        if make_positive:
            z_scored_data = z_scored_data + np.abs(z_scored_data.min(axis=0))
        # Replace the dropped columns at the beginning of the dataframe:
        dropped_cols = list(set(self.dataset.columns).difference(z_scored_data.columns))
        z_scored_data = z_scored_data.join(self.dataset[dropped_cols])[self.dataset.columns]

        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=z_scored_data)
        return z_scored_data

    def minmax_scaling(self, save=False):
        '''
        Function to perform min-max feature scaling to bring values into the range [0,1].
        :param save: Select whether to save the processed dataset.
        :return: Scaled dataframe.
        '''
        data = self.dataset[:]
        # Drop non-numerical columns:
        nonnumerics = data.select_dtypes('object')
        data.drop(nonnumerics.columns, axis=1, inplace=True)
        normed_data = data.apply(lambda x: (x.astype(float) - min(x)) / (max(x) - min(x)), axis=0)
        # Add the dropped columns back to the front of the dataframe:
        normed_data = pd.concat([nonnumerics, normed_data], axis=1)
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=normed_data)
        return normed_data

    def extract_by_keywords(self, keywords, save=False):
        '''
        Function for selecting many features at once, by allowing the user to specify shared words/strings of the
        features.
        :param keywords: Keywords contained within the features to select.
        :param save: Defaults to False, specify whether to save the dataframe after editing.
        :return: The modified dataframe.
        '''
        if not isinstance(keywords, list): keywords = [keywords]
        self.dataset = self.dataset[[column for column in self.dataset.columns if
                                     any(keyword.casefold() in column.casefold() for keyword in keywords)]]

        # If save is True, save the dataset:
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)
        return self.dataset

    def delete_by_keywords(self, keywords, save=False):
        '''
        Function to delete many features from a dataset at once, given a list of keywords
        :param keywords: List of words (or strings that are not words). Any columns containing these keywords will be
        deleted from the dataframe.
        :param save: Select whether to save the processed dataset.
        :return: Altered dataframe, with columns removed.
        '''
        if not isinstance(keywords, list): keywords = [keywords]
        self.dataset.drop([column for column in self.dataset.columns if
                           any(keyword.casefold() in column.casefold() for keyword in keywords)], axis=1, inplace=True)

        # If save is True, save the dataset:
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)
        return self.dataset

    def remove_samples(self, keys, sample_col=None, save=False):
        '''
        Function to manually remove samples from a serological dataset, by specifying the name/identifier of the sample(s)
        to remove. This function is designed for removal of potential outliers.
        :param keys: Names of the samples to remove.
        :param sample_col: The column that samples are contained within. If not given, will default search to the
        dataframe indices.
        :param save: Select whether to save the processed dataset.
        :return: Altered dataframe, with select samples removed.
        '''
        if not isinstance(keys, list): keys = [keys]
        keys = [str(key) for key in keys]

        if sample_col is not None and sample_col.casefold() == "index":
            self.dataset = self.dataset[~self.dataset.index.astype(str).str.contains('|'.join(keys))]
        elif sample_col is not None and sample_col.casefold() != "index":
            # sample_col = sample_col.replace("_", " ")
            self.dataset = self.dataset[
                ~self.dataset.iloc[:, sample_col].astype(str).str.contains('|'.join(keys)) if isinstance(sample_col,
                    int) else ~self.dataset[sample_col].astype(str).str.contains('|'.join(keys))]
        else:  # Search for keywords in the indices.
            self.dataset = self.dataset[~self.dataset.index.astype(str).str.contains('|'.join(keys))]

        # If save is True, save the dataset:
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)
        return self.dataset

    def keep_samples(self, keys, sample_col=None, save=False):
        '''
        Function to keep only select samples from a serological dataset, by specifying the name/identifier of the
        sample(s) to keep. The rest of the columns are removed.
        :param keys: Names of the samples to keep.
        :param sample_col: The column that samples are contained within. If not given, will default search to the
        dataframe indices.
        :param save: Select whether to save the processed dataset.
        :return: Altered dataframe, with select samples removed.
        '''
        if not isinstance(keys, list): keys = [keys]
        keys = [str(key) for key in keys]

        if sample_col is not None and sample_col.casefold() == "index":
            self.dataset = self.dataset[self.dataset.index.astype(str).str.contains('|'.join(
                keys))]  # A bit complicated with all the casting, but I was trying to deal with any form that the keys
            # might come in...
        elif sample_col is not None and sample_col.casefold() != "index":
            # sample_col = sample_col.replace("_", " ")
            self.dataset = self.dataset[
                self.dataset.iloc[:, sample_col].astype(str).str.contains('|'.join(keys)) if isinstance(sample_col, int) else
                self.dataset[sample_col].astype(str).str.contains('|'.join(keys))]
        else:  # Search for keywords in the indices.
            self.dataset = self.dataset[self.dataset.index.astype(str).str.contains('|'.join(keys))]

        # If save is True, save the dataset:
        if save:
            self.save_dataset(output_basename=self.output_basename, dataset=self.dataset)
        return self.dataset

    def plspm_setup(self):
        '''
        Function to set up a dataframe for PLS-PM.
        :return: Dataframe after setup has been performed, containing z-scored and processed data.
        '''
        # Search for AUC columns:
        cols = self.dataset.filter(regex="AUC")
        # Invert the AUC columns to get a measure that's positively correlated w/ the other features rather than
        # negatively:
        cols = cols.apply(lambda x: 1 / x, axis=1)
        # Replace "inf" with 1:
        cols = cols.replace(np.inf, 1.0)
        self.dataset[cols.columns] = cols
        z_scored = self.z_score(save=True)
        return z_scored
