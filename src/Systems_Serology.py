# Daniel Zhu
# Feburary 14th, 2021
# Main function to perform various analyses of multivariate systems serology data.
from Systems_Serology_Preprocessing import SystemsSerologyPreprocessing
import matplotlib.pyplot as plt
from PlottingFunctions import shifted_colormap

plt.style.use("seaborn-talk")
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import os
from pathlib import Path

pd.set_option('display.max_rows', None)
import seaborn as sns
from statannot import add_stat_annotation
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import statistics
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score
import random
from itertools import combinations
from copy import copy

import argparse

# Set up argparse:
# Allow use of Boolean values:
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default="..\Datasets\Seattle2_exampleCOVIDset.csv", type=str,
                    help='Path to the dataset to analyze.')
parser.add_argument('--datapath2', type=str,
                    help='Path to a second systems serology dataset, to allow for comparisons to be made between the '
                         'two.')
# Argument to specify the task (from a list of options):
parser.add_argument('--task', default='Preprocessing', const='Preprocessing', nargs='?',
                    choices=['Preprocessing', 'Z_Score', 'Feature_Selection', 'Remove_Columns', 'Keep_Columns',
                             'Remove_Samples', 'PCA', 'PLS', 'Heatmap', 'Bars', 'Violins', 'Boxplots', 'Swarm', 'Strip',
                             'Flowers', 'Dotplots', 'Most_Predictive', 'Classification', 'Regression',
                             'Feature_Influence', 'Variable_Enrichment', 'Subset_Data', 'Move_Columns',
                             'Move_Rows', 'Concatenate', 'Copy', 'Shorten_Columns', 'Combine_Columns',
                             'Trajectory_Analysis'],
                    help='Specify the task that you would like to accomplish with this dataset. Default: %(default)s')
parser.add_argument('--labels', type=str,
                    help='Column identifier for the dataframe column that contains sample ID information.')
parser.add_argument('--group', type=str,
                    help='Column identifier for the dataframe column that contains group information.')
parser.add_argument('--var', nargs='+', action='extend', type=str,
                    help='Specify variables of interest; this is a multipurpose function to specify any variables to '
                         'regress, plot, serve as controls, etc. If spaces exist in the variable names, '
                         'please replace them with "_". Specify --var before each variable ID. If "all", '
                         'use all numerical features in the dataset for analysis. Note that not all functions support '
                         'the "all" assignment.')
# Argument to specify keywords/key strings, to be used to extract or delete a subset of features containing these
# keywords/strings.
parser.add_argument('--key', nargs='+', action='extend', type=str,
                    help='Generic args used to supply the function with specific words or strings that can be used '
                         'for further processing. Example usage: for feature selection, specify words or phrases '
                         'common to several features, which can be used to extract or delete these together. When '
                         'removing samples, these should be the names of the samples to remove.')
parser.add_argument('--string', type=str, help='Argument that is used to provide any additional '
                         'string arguments beyond those which can be covered by var, key, and title.')
parser.add_argument('--xlabel', type=str, help='Used to provide x-axis labels to certain plotting functions.')
parser.add_argument('--ylabel', type=str, help='Used to provide y-axis labels to certain plotting functions.')
parser.add_argument('--set_flag', type=str2bool, default=True, help='Argument used to set Boolean arguments to True '
                        'or False. Note that some functions will have a Boolean save argument that will always be set '
                        'True; use this argument to adjust any other Boolean arguments, however.')
parser.add_argument('--model_type', type=str, default='logistic', help='Specifies the type of regression to perform. \
    Options are linear, logistic and SVM.')
parser.add_argument('--num_lvs', default=2, type=int,
                    help='Number of latent variables or principal components to compute.')
parser.add_argument('--num_loadings', default=10, type=int, help='Number of highest loadings to view on PCA/PLS plots.')
parser.add_argument('--num_iterations', default=100, type=int,
                    help='Number of iterations for the Classification() function.')
parser.add_argument('--title', type=str, help='Title of the plot.')
parser.add_argument('--fontsize', default=24, type=int, help='Font size to use for plot titles.')
parser.add_argument('--alpha', default=0.01, type=float, help='Alpha value for LASSO feature selection.')
parser.add_argument('--cluster_flag', default=False, type=bool, help='Choose whether to apply Kmeans clustering to '
                                                                     'the data. If not, data will be grouped by group '
                                                                     'only.')
parser.add_argument('--num_clusters', type=int, help='Number of clusters to group PCA and PLS results into.')
parser.add_argument('--draw_ellipses', default=False, type=bool,
                    help='Set to True to draw 95% confidence ellipses on PLS or PCA plots.')
parser.add_argument('--draw_colorbar', default=False, type=bool,
                    help='Set to True to color code the PCA/PLS points by their values for variables given by the '
                         '--var argument.')
parser.add_argument('--nested_grouping', default=False, type=bool,
                    help='Set to True if features are multiple measurements of the same variable (e.g. a timecourse). '
                         'Used in comparison plotting.')
parser.add_argument('--heatmap_type', default='self', type=str, help='Kind of heatmap to construct. Options: self for '
                    'correlation within one dataset, cross for correlation between two, z for z-score heatmap.')

args = parser.parse_args()


class SystemsSerology():
    def __init__(self, data, second_dataset=None, labels_col_id=None, group_col_id=None, transpose=False):
        '''
        This class will have one required argument if a class is instantiated; either the path to the dataset to
        perform analysis on (which will be read and converted to a dataframe) or the dataset itself, given as a
        dataframe. Optionally, a column identifier for labels or columns can also be given if the data has not
        already been subsetted/processed, and any number of additional column identifier names can be specified to
        identify columns of interest for further analysis. Functions in this class will assume data is of shape (
        samples x variables).
        :param data: The path to the dataset or the dataset itself (given as a dataframe). If provided the path (a
        string argument), pd.read_csv() will be used to read from the path to a dataframe object.
        :param second_dataset: Can be used to optionally provide the path to another dataset (or another dataset,
        given as a dataframe), to be able to make comparisons between the two datasets.
        :param labels_col_id: The name of a column of the dataframe that identifies certain data points as belonging
        to particular samples (e.g. this can be a patient ID).
        :param group_col_id: The name of a column of the dataframe that contains information on some categorical group (
        e.g. survived vs. died), used for visualization purposes in PCA/PLS-DA.
        :param transpose: If data is not already in the form (samples x variables), this argument can be used to
        easily transpose it before further processing.
        '''
        self.dataset = pd.read_csv(data, index_col=0) if isinstance(data, str) else data
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError('Dataset must be given as a dataframe or .csv file.')
        if second_dataset is not None:
            self.second_dataset = pd.read_csv(second_dataset, index_col=0) if isinstance(second_dataset,
                                                                                         str) else second_dataset
            if not isinstance(self.second_dataset, pd.DataFrame):
                raise TypeError('Dataset must be given as a dataframe or .csv file.')
            # Filter second dataset so that it has the same indices as the first dataset:
            shared_samples = self.dataset.index & self.second_dataset.index
            self.second_dataset = self.second_dataset.loc[shared_samples, :]
        else:
            self.second_dataset = None
        # self.labels_col_id = labels_col_id.replace("_", " ") if labels_col_id is not None else labels_col_id
        self.labels_col_id = labels_col_id
        self.group_col_id = group_col_id
        # self.group_col_id = group_col_id.replace("_", " ") if group_col_id is not None else group_col_id
        if self.labels_col_id is not None and self.labels_col_id.casefold() == "index":
            self.index_labels = self.dataset.index
        # If the labels ID or the group ID are not found in the dataset, search the second dataset for them and then add
        # as a column:
        if self.labels_col_id is not None and self.labels_col_id not in self.dataset.columns and second_dataset is \
                not None:
            try:
                labels = self.second_dataset[self.labels_col_id]
                self.dataset[self.labels_col_id] = labels
            except Exception:
                raise ValueError("Labels column ID not found in either dataset provided.")
        if self.group_col_id is not None and self.group_col_id not in self.dataset.columns and second_dataset is not \
                None:
            try:
                groups = self.second_dataset[self.group_col_id]
                self.dataset[self.group_col_id] = groups
            except Exception:
                raise ValueError("Outcomes column ID not found in either dataset provided.")

        if transpose:
            self.dataset = self.dataset.transpose()
        # Set consistent colormap:
        # Ad26 dosedown colors:
        #self.colors = ["#b57edc", "#ffb347"]
        # v2:
        self.colors = ["#dda0dd", "#ffa500", "#9370db", "#7a7aff", "#ff91a4", "#0f52ba", "#ed2939",
                       "#ffe135", "#bb3385"]
        # Immunosuppression colors:
        #self.colors = ["#dcdcdc", "#4682b4", "#93ccea", "#696969", "#778899"]
        # COVID watch colors:
        #self.colors = ["#ffb3ba", "#ffdfba", "#ffffba", "#baffc9", "#bae1ff", "#e2baff", "#ffbae7"]
        # Also select a matching colormap:
        self.colormap = 'YlOrBr'
        #self.colors = 'rainbow'


    @staticmethod
    def split_dataset_and_variable(data, variable_col_id):
        '''
        Function to separate a dataset from a a column originally in that dataset, if the two initially exist within
        the same dataframe. Assumes the dataset is of the form (samples x variables).
        :param data: Dataset, given as a dataframe.
        :param variable_col_id: Name (or index) of the column corresponding to the variable of interest.
        :return: The original dataset with groups removed, and a separate dataframe containing group information.
        '''
        try:
            variable = data.iloc[:, variable_col_id].values if isinstance(variable_col_id, int) else data[
                variable_col_id].values
            variable = pd.DataFrame(data=variable, index=data.index, columns=[variable_col_id])
        except:
            raise ValueError("Outcomes_col_id invalid (must be integer or string).")
        try:
            data = data.drop(data[variable_col_id], axis=1) if isinstance(variable_col_id, int) else data.drop(
                variable_col_id, axis=1)
        except:
            raise ValueError("Column already separate from dataframe, exiting function.")
        return data, variable


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

    @staticmethod
    def remove_NaNcols(data, drop_invalid=False):
        '''
        Function to remove columns of a dataframe that contain NaN values.
        :param data: Dataframe to edit.
        :param drop_invalid: Select whether to also drop columns that contain invalid values (e.g. infinity). Note
        this doesn't work on string/object columns.
        :return: Dataframe once columns that contain NaN values have been removed.
        '''
        for column in data.columns:
            if data[column].isnull().values.any():
                data.drop(column, axis=1, inplace=True)
            if drop_invalid:
                if np.isinf(data[column]).values.any():
                    data.drop(column, axis=1, inplace=True)
        return data


    @staticmethod
    def remove_invalidrows(data):
        '''
        Function to remove rows that contain NaNs or inf values.
        :param data: Dataframe to edit.
        :return: Dataframe once rows that contain invalid values have been removed.
        '''
        data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
        return data


    @staticmethod
    def kmeans_clustering(data, n_clusters=None):
        '''
        This is a general K-means clustering function, but the intention is for it to be applicable to PLS/PCA data
        for systems serology, in order to potentially find samples of interest. This function will automate the
        process of finding the number of clusters, but can also be given a number of clusters to group datapoints
        into, if that is known a priori.
        :param data: Dataframe to perform k-means clustering on.
        :param n_clusters: Can be optionally used to specify a specific number of clusters. Defaults to None,
        and if None, will create clusters based on the elbow method.
        :return: Cluster labels.
        '''
        # If n_clusters is not specified, automatically find the optimal number of clusters:
        if n_clusters is None:
            # Maximum number of clusters to test:
            max_clusters = 10
            # Initialize variable to store the within-cluster-sum-of-squares (WCSS)- this is probably better known as
            # the elbow plot.
            wcss = []

            # Test different values for the number of clusters:
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=10, n_init=2)
                # NOTE: KMeans parameters: init = method for initialization; k-means++ selects initial cluster
                # centers in a way to speed up convergence. n_init = the number of times the k-means algorithm will
                # be run with different centroid seeds; the final results will be the best output. max_iter = maximum
                # number of iterations of the k-means algorithm for a single run.
                kmeans.fit(data)
                wcss.append(
                    kmeans.inertia_)  # KMeans.inertia is the sum of squared distances of samples to their closest
                # cluster center.

            # Find the "elbow", which can be approximated by the number of clusters for which the second derivative
            # is highest. The second derivative can be approximated using the central difference:
            secondDeriv = []
            # From range(2, len(wcss)-1) because it is likely the first element will have the highest central
            # difference (likely incorrect).
            for i in range(2, len(wcss) - 1):
                secondDeriv.append(wcss[i + 1] + wcss[i - 1] - 2 * wcss[i])
            # Return index of the max second derivative + 2 to account for indexing.
            n_clusters = secondDeriv.index(np.max(secondDeriv)) + 2

        # Define KMeans object to perform the clustering itself:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=500, n_init=25)
        # Apply the fit_predict method to the dataset to return which cluster each observation belongs to:
        fit_clusters = kmeans.fit_predict(data)
        # Return cluster assignments:
        return fit_clusters


    @staticmethod
    def group_features(dataset, group_keys):
        '''
        Function to rearrange the columns of a dataframe such that they are ordered based on user-specified group keys.
        :param dataset: The dataset to rearrange.
        :param group_keys: A list of strings to use to group columns.
        :return: The rearranged dataframe.
        '''
        if not isinstance(group_keys, list): group_keys = [group_keys]
        rearranged_df = pd.DataFrame()
        # For each key, subset the dataset to return columns containing those keys, and append to grow the rearranged
        # dataframe:
        for key in group_keys:
            cols = [col for col in dataset.columns if key in col]
            subset = dataset[cols]
            rearranged_df = pd.concat([rearranged_df, subset], axis=1)
        return rearranged_df

    @staticmethod
    def group_samples(dataset, groups):
        '''
        Function to rearrange the rows of a dataframe such that they are ordered based on specified group keys.
        :param dataset: The dataset to rearrange.
        :param groups: Either a string (to specify a column of the dataframe containing group information) or a
        series with matched indices to dataset.
        :return: The rearranged dataframe, and groups.
        '''
        # If string, extract the column from the dataframe corresponding to the ID.
        if isinstance(groups, str):
            groups = self.data[groups]
        rearranged_df = pd.DataFrame()
        # For each group, subset the data to return rows belonging to that group, and append to grow the rearranged
        # dataframe:
        for group in set(groups):
            subset = dataset[groups == group]
            rearranged_df = pd.concat([rearranged_df, subset], axis=0)
        # If given as a separate series, reorder the groups variable as well, to match the rearranged dataframe.
        if not isinstance(groups, str):
            groups = groups.reindex(rearranged_df.index)
        return rearranged_df, groups

    @staticmethod
    def combine_columns(dataset, cols, name_combined_col='Combined'):
        '''
        Function to combine two columns together. This is only really useful to be able to plot a larger group of
        data against smaller subsets of that same data, as the data is duplicated.
        :param dataset: Dataframe to apply this function to, or the path to a dataframe.
        :param cols: Two-item list specifying the labels of the columns to be combined.
        :param name_combined_col: Label to assign the new column formed by merging the specified two.
        :return: The modified dataframe.
        '''
        # Check whether dataset is given as a frame or as the path to a dataframe:
        dataset = pd.read_csv(dataset, index_col=0) if isinstance(dataset, str) else dataset
        # Check that cols actually contains only two items:
        if len(cols) != 2:
            raise ValueError("Merge operation can only be performed with two columns at a time.")
        # Find the columns indicated by the items in cols; these must be completely matching:
        col1_id, col2_id = cols[0], cols[1]
        # First, duplicate the dataset, then remove the column specified by col2_id from the original dataframe and
        # the column specified by col1_id from the copied dataset:
        temp_data = dataset.copy()
        dataset.drop(col2_id, axis=1, inplace=True)
        temp_data.drop(col1_id, axis=1, inplace=True)
        # Rename col1 in the original dataset and col2 in the duplicated dataset to what is specified by
        # name_combined_col:
        dataset.rename(columns={col1_id: name_combined_col}, inplace=True)
        temp_data.rename(columns={col2_id: name_combined_col}, inplace=True)
        # Combine columns by appending dataset and temp_data together (and reset index to avoid potential re-indexing
        # issues):
        combined_cols_df = pd.concat([dataset, temp_data], axis=0, ignore_index=True)
        # Save the dataset (instantiate SystemsSerologyPreprocessing, with data=dataset- this doesn't actually matter):
        SystemsSerologyPreprocessing(data=dataset).save_dataset(output_basename='SystemsSerologyCombinedCols',
                                                               dataset=combined_cols_df)



    def lasso_feature_selection(self, output_basename=None, alpha=1.0, **kwargs):
        '''
        Function to perform feature selection on a dataset, with evaluation after each round with PCA/PLS-DA.
        :param output_basename: The base name of the file to save to (do not specify the drive/path or file
        extension, as this will lead to redundancy). Will default to "processed_dataset" if not given.
        :param groups: Optional argument; allows user to pass a separate dataframe containing group information. In
        this case, use the class dataset variable as the data to select, without further pre-processing.
        :param alpha: parameter for Lasso() (constant that multiplies the L1 term), defaults to 1.0.
        :param kwargs: Additional arguments that can be provided to the linear regression function.
        :return: The dataframe after all feature selection has happened.
        '''
        if self.group_col_id is not None:
            data, groups = self.split_dataset_and_variable(self.dataset, self.group_col_id)
        else:
            print("Feature selection is performed using regression. Output identifier is None, so the values to "
                  "regress against have not been specified. Please re-instantiate class with group_col_id specified.")
            return

        # Remove non-numerical columns and columns containing NaNs (which cannot be used in regression):
        data = self.remove_nonnumeric(data)
        data = self.remove_NaNcols(data, drop_invalid=False)
        # Remove rows containing invalid values (such as infinity):
        data = self.remove_invalidrows(data)
        groups = groups.loc[data.index]

        # Apply standardscaler to z-score the data (linear models benefit from feature scaling):
        scaler = StandardScaler()
        scaler.fit(data)

        # Apply regularization to shrink coefficients less important to the prediction to zero, thereby removing them
        # from the model:
        feature_selector = SelectFromModel(estimator=Lasso(alpha=alpha,
                                                           **{key: value for key, value in kwargs.items() if
                                                              key in vars(Lasso()).keys()}))
        # If groups are categorical, will need to convert them to numerical values; note that this will only work for
        # binary categorical variables, as this function uses logistic regression.
        if not any(type in str(groups[groups.columns[0]].dtype) for type in ["int", "float", "double", "long"]):
            cat_to_num = groups.iloc[:, 0].astype('category').cat.codes
            if len(set(cat_to_num)) > 2:
                raise RuntimeError(
                    "Categorical features should include no more than two distinct states, as conversion and "
                    "subsequent linear regression will likely be inaccurate on multiple categorical values that may "
                    "not be linearly related.")
            groups = pd.DataFrame(data=cat_to_num, index=groups.index, columns=["Outcome"])

        feature_selector.fit(scaler.transform(data), groups.values)

        # Subset the data to keep only the features that were not removed:
        data = data.loc[:, feature_selector.get_support()]
        # It's likely the samples column and groups column were removed by the function that removes non-numeric
        # columns; retrieve those and add them back to the dataframe.
        if self.labels_col_id is not None:
            data.insert(0, column=self.labels_col_id, value=self.dataset[self.labels_col_id])
        data[self.group_col_id] = self.dataset[self.group_col_id]
        # Save the feature-selected data if the optional basename to save to is given. Will be saved to the
        # Processed_Data folder.
        if output_basename is not None:
            SystemsSerologyPreprocessing.save_dataset(output_basename=output_basename, dataset=data)
        return feature_selector.get_support(), data  # return both the "regularization mask" and the subsetted data.


    def latent_biplot(self, scores, coeffs, labels):
        '''
        Function to plot latent variable eigenvectors as a biplot, and additionally return the magnitude of each
        eigenvector.
        :param scores: The projected data (using PCA, PLS, etc.), given as a dataframe.
        :param coeffs: The eigenvectors (obtained using pca.components_, etc.), given as a dataframe.
        :param labels: Class labels (for plotting the data on top), given as a dataframe.
        :return: None.
        '''
        # Check to make sure all of the inputs are dataframes:
        if not all(isinstance(i, pd.DataFrame) for i in [scores, coeffs, labels]):
            print("All of scores, coeffs and labels must be given as dataframes with identifical indices.")
            sys.exit()
        n_features = coeffs.shape[0]    # number of variables to plot.
        fig, ax = plt.subplots(1,1, figsize=(10, 10), dpi=100)
        classes = sorted(set(labels.iloc[:,0]))
        palette = dict(zip(classes, self.colors))
        sns.scatterplot(x=scores.columns[0], y=scores.columns[1], hue=labels.iloc[:,0], palette=palette, data=scores,
                        edgecolor='k', s=150, ax=ax)
        for i, feat in enumerate(coeffs.index):
            feat = feat.split("_")[0]
            # Plot the variable scores as arrows (scale by the x- and y-limits for plotting purposes) (and 2 is an
            # arbitrary scaling coeff for now):
            plt.arrow(0, 0, coeffs.iloc[i,0]*ax.get_xlim()[1], coeffs.iloc[i,1]*ax.get_ylim()[1], color='k',
                      alpha=0.5, linestyle='-', linewidth=2)
            plt.text(coeffs.iloc[i,0]*ax.get_xlim()[1]*1.25, coeffs.iloc[i,1]*ax.get_ylim()[1]*1.05, feat,
                     color='k', ha='center', va='center', fontsize=10)
        ax.set_title("Loadings Biplot")
        ax.set_xlabel("Latent Variable 1")
        ax.set_ylabel("Latent Variable 2")
        plt.tight_layout()
        plt.show()


    def n_dimensional_PCA_view_2D(self, n_components=2, title=None, fontsize=24, draw_colorbar=False, colorbar_var=None,
                                  n_loadings=5, find_clusters=False, num_clusters=None, draw_ellipses=False):
        '''
        Function to perform PCA on a given dataset, and project to a 2-D space for visualization. This function
        assumes the dataset is of shape (samples x variables).
        :param n_components: Number of principal components to compute. Defaults to 2.
        :param title: Optional string argument that can be used to title the plot.
        :param fontsize: Font size to use for plot title.
        :param draw_colorbar: Optional; if set to True, add a gradient-based color scheme to the data points,
        given a variable of interest to base the gradient on (given by the next arg, colorbar_var).
        :param colorbar_var: Optional column identifier for a variable that can be used to create a color gradient
        for the points.
        :param n_loadings: Specify the number of the highest loadings to view in bar chart form. Defaults to 5.
        :param find_clusters: If True, cluster data points in the PLS representation together, and draw ellipses
        based on clustering. If False, draw ellipses based on the group variable.
        :param num_clusters: Allows for the user to set the number of clusters (if find_clusters is False,
        this argument will be unused).
        :param draw_ellipses: If True, will draw 95% confidence ellipses around each group group or cluster in PCA
        space.
        :return: Loadings.
        '''
        # Define the groups dataframe (different groups will be assigned different colors in the PCA plot):
        if self.group_col_id is not None:
            data, groups = self.split_dataset_and_variable(self.dataset, self.group_col_id)
        # If labels are not given at all, assign each point the same label:
        else:
            data = self.dataset[:]
            groups = pd.DataFrame(data=np.zeros(len(data)), index=data.index, columns=self.group_col_id)

        if self.labels_col_id is not None and self.labels_col_id.casefold() != "index":
            data, sample_IDs = self.split_dataset_and_variable(data, self.labels_col_id)
        elif self.labels_col_id is not None and self.labels_col_id.casefold() == "index":
            sample_IDs = self.dataset.index
        else:
            sample_IDs = None

        # Remove nonnumeric and NaN columns:
        data = self.remove_nonnumeric(data)
        data = self.remove_NaNcols(data)

        # PCA is affected by scale, so each of the features in the data needs to first be scaled. Use sklearn's
        # StandardScaler to standardize the features onto unit scale (mean = 0 and variance = 1) for each feature.
        features = StandardScaler().fit_transform(data)
        # Convert this back to a dataframe:
        features = pd.DataFrame(data=features, index=data.index, columns=data.columns)

        # PCA! Project original data down into a n-dimensional space.
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(features)

        # Generate a data frame using the transformed variables (for later reference):
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['Principal Component {}'.format(x + 1) for x in range(n_components)],
                                    index=data.index)
        principal_df.to_csv("./Processed_Data/SystemsSeroPCA.csv")

        # Calculate the 95% confidence ellipse bounds for the first two principal components for each group (which
        # will be plotted):
        # If find_clusters is True, perform k-means clustering on the data (NOT the PCA representation) and draw
        # ellipses for each cluster. If False, use the groups list to draw ellipses.
        if find_clusters:
            cluster_labels_df = pd.DataFrame(data=self.kmeans_clustering(data=principal_df, n_clusters=num_clusters),
                                             columns=["Value"], index=data.index)
            group_labels = cluster_labels_df["Value"]
            # Add the PCA cluster labels to the source dataframe for potential further use:
            principal_df["PCA Clusters"] = group_labels
            unique_groups = sorted(set(group_labels))
        else:
            group_labels = groups[self.group_col_id]
            unique_groups = sorted(set(group_labels))

        # An inner function to draw ellipses on a PCA plot.
        def draw_ellipses_PCA(color_list=None, facecolor=False):
            '''
            Function to draw ellipses on a PCA plot.
            :param color_list: Can optionally supply the colors that will be used to color the ellipses.
            :param facecolor: Set whether ellipses should be colored in or not.
            :return: List of ellipse objects.
            '''
            ellipses = []
            for index, group in enumerate(unique_groups):
                principal_df_subset = principal_df[group_labels == group]
                pc_1_vals = list(principal_df_subset.iloc[:, 0])
                pc_2_vals = list(principal_df_subset.iloc[:, 1])
                covariance = np.cov(pc_1_vals, pc_2_vals)
                # The eigenvalues of the covariance matrix represent the magnitude of the variance in the direction
                # of the principal components.
                lambda_, v = np.linalg.eig(covariance)
                # And take the square root of the variance metric to obtain the standard deviation for plotting.
                lambda_ = np.sqrt(lambda_)
                # Compute the angle (correct orientation of the ellipse):
                theta = np.degrees(np.arctan2(*v[:, 0][::-1]))

                # Ellipse covering two standard deviations:
                # xy = coordinates of the center, width = length of horizontal axis, height = length of vertical
                # axis, angle = rotation in degrees.
                if color_list is None:
                    # If more than four groups, use a random permutation of mcolors keys to determine the other colors.
                    CCS4_colors = list(mcolors.CSS4_COLORS.keys())
                    color_list = ["b", "purple", "r", "g", "orange", *random.sample(CCS4_colors, len(CCS4_colors))]
                # Ellipse covering two standard deviations:
                # xy = coordinates of the center, width = length of horizontal axis, height = length of vertical
                # axis, angle = rotation in degrees.
                ellipses.append(Ellipse(xy=(np.mean(pc_1_vals), np.mean(pc_2_vals)), width=lambda_[0] * 2 * 2,
                                        height=lambda_[1] * 2 * 2, angle=theta, fill=facecolor,
                                        edgecolor='k' if facecolor else color_list[index], facecolor=color_list[index],
                                        alpha=0.2 if facecolor else 1.0, linewidth=2.0))
            return ellipses


        if n_components > 1:
            # Set up the plot to visualize the principal components in 2D space:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_xlabel('Principal Component 1 ({:.2f}%)'.format(pca.explained_variance_ratio_[0] * 100), fontsize=16)
            ax.set_ylabel('Principal Component 2 ({:.2f}%)'.format(pca.explained_variance_ratio_[1] * 100), fontsize=16)
            if title is not None:
                ax.set_title(title.replace("_", " "), fontsize=fontsize)

            # Add text labels if applicable:
            if sample_IDs is not None:
                for n, *vals in enumerate(principal_df.values):
                    point_label = sample_IDs.values[n]
                    ax.annotate(point_label, xy=(vals[0][0], vals[0][1]), xytext=(8, -2.5), textcoords='offset points',
                                fontsize=8)  # vals[0][0], vals[0][1] are the first two principal components. xytext
                    # specifies label offset.

            palette = dict(zip(unique_groups, self.colors))
            sns.scatterplot(x="Principal Component 1", y="Principal Component 2", hue=groups.iloc[:, 0],
                            palette=palette, data=principal_df, edgecolor='k', s=150, ax=ax)
            legend = ax.legend(fontsize=16)

            # Plot 95% confidence ellipse on data:
            if draw_ellipses:
                ellipses = draw_ellipses_PCA(color_list=list(palette.values()), facecolor=True)
                for ellipse in ellipses:
                    # Copy the ellipse (can't put a single artist in more than one figure):
                    ellipse_copy = copy(ellipse)
                    ax.add_patch(ellipse_copy)
        plt.show()

        # PCA loadings:
        # Loadings represent the weight of each variable, e.g. how much it contributes to the corresponding principal
        # component.
        loadings = pca.components_
        loadings_df = pd.DataFrame(data=loadings,
                                   index=['Principal Component {}'.format(x + 1) for x in range(n_components)],
                                   columns=features.columns).transpose()
        loadings_df.index.name = None
        loadings_df.to_csv("./Processed_Data/SystemsSeroPCALoadings.csv")
        # Sort loadings in descending order by absolute value to see which variable contributes the most positively
        # to variance (e.g. is most differentially expressed):
        pc1_loadings = loadings_df.iloc[loadings_df['Principal Component 1'].abs().argsort()[::-1]]
        pc2_loadings = loadings_df.iloc[loadings_df['Principal Component 2'].abs().argsort()[::-1]]

        # Make a bar chart to look at the loadings (currently only on principal component 1, but maybe add option to
        # look at other PCs).
        for index, df in enumerate([pc1_loadings, pc2_loadings]):
            fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 6))
            ax.set_xlabel('Loading Weight', fontsize=16)
            # Add ax.set_xticklabels()
            ax.set_ylabel('Variable Name', fontsize=16)
            df.index = [name.replace("_", " ") for name in list(df.index)]
            ax.set_title('Principal Component Loadings', fontsize=20)
            # Take the first n loadings, where n = n_loadings parameter.
            if n_loadings > len(df): n_loadings = len(df)
            df = df[:n_loadings]
            sns.barplot(data=df, x='Principal Component {}'.format(index + 1), y=df.index, palette=self.colormap, ax=ax,
                        edgecolor=".2")
            plt.tight_layout()
            plt.show()

        return loadings_df, principal_df


    def PLS(self, n_components=2, var_for_regression=None, title=None, fontsize=24, n_loadings=5, find_clusters=False,
            num_clusters=None, draw_ellipses=False, key=None):
        '''
        Function to perform PLS-R/PLS-DA on a given dataset, and project to a 2-D space for visualization. This
        function assumes the dataset is of shape (samples x variables).
        :param n_components: Number of latent variables to compute. Defaults to 2.
        :param var_for_regression: Variable to regress on, given as the name of the variable. If not given,
        default to using the group column.
        :param title: Optional string argument that can be used to title the plot.
        :param fontsize: Font size to use for plot title.
        :param n_loadings: Number of PLS loadings to visualize on bar plot. Defaults to 5.
        :param find_clusters: If True, cluster data points in the PLS representation together, and draw ellipses
        based on clustering. If False, draw ellipses based on the group variable.
        :param num_clusters: Allows for the user to set the number of clusters (if find_clusters is False,
        this argument will be unused).
        :param draw_ellipses: If True, draw 99% confidence ellipses on PLS points.
        :param key: Optional argument, column ID for one of the variables in the dataset (must be matched case). If not
        None, the points will be colored by the value of this variable instead of by group.
        :return: Loadings.
        '''
        # Define each variable to regress on, and separate it from the dataframe:
        # List to store this information for each variable (for each variable to regress on, the rest of the
        # dataframe and the information associated with the variable will be stored in these two lists):
        data_list, variable_list = [], []
        # If the key argument is not None, will also need to create a list to store key information.
        if key is not None:
            hue_list = []
            for var in key:
                # Extract the column, preserve it as a dataframe:
                hue_list.append(pd.DataFrame(self.dataset[var], columns=[var]))
                #self.dataset.drop(var, axis=1, inplace=True)
        if var_for_regression is not None:
            # If "all" is given as the variable for regression, make plots for all variables in both datasets (or one
            # if a second isn't given):
            if var_for_regression[0].casefold() == "all":
                # print(set(self.dataset.columns | self.second_dataset.columns))
                var_for_regression = list(set(pd.Index.union(self.dataset.columns,
                                                             self.second_dataset.columns))) if self.second_dataset is\
                                                                                               not None else list(
                    self.dataset.columns)
            if not isinstance(var_for_regression, list): var_for_regression = [var_for_regression]
            for var in var_for_regression:
                # Define a new working variable to avoid altering the class variable:
                data = self.dataset[:]
                # If the variable is not in the dataset, check to see if another dataset was provided, and search
                # that dataset for the column in question. If the feature of interest is in that dataset, copy it to
                # the working dataset temporarily.
                if var not in data.columns and self.second_dataset is not None:
                    data[var] = self.second_dataset.loc[data.index, var]  # .loc[data.index, ] to deal with
                    # situations in which pre-processing has rendered the two datasets to be different in size.
                data, variable_for_regression = self.split_dataset_and_variable(data, var)
                data_list.append(data)
                variable_list.append(variable_for_regression)
        elif var_for_regression is None:
            if self.group_col_id is not None:
                data, variable_for_regression = self.split_dataset_and_variable(self.dataset, self.group_col_id)
                data_list.append(data)
                variable_list.append(variable_for_regression)
            # If a variable to regress on is not given at all, there is no point to PLS, as there is nothing to
            # regress on:
            else:
                raise ValueError(
                    "Error: PLS is a regression-based analysis, and no variable to regress on was given. Please "
                    "specify a variable to regress on using the group_col_id or var arguments; exiting program now.")

        # If sample IDs are given, place these into a list. First return is "_" because we don't care about the group
        # here, just the sample labels.
        if self.labels_col_id is not None and self.labels_col_id.casefold() != "index":
            _, sample_IDs = self.split_dataset_and_variable(self.dataset, self.labels_col_id)
        elif self.labels_col_id is not None and self.labels_col_id.casefold() == "index":
            sample_IDs = self.dataset.index
        else:
            sample_IDs = None

        # Define the groups dataframe if it is given (different groups will be assigned different colors in the PLS-DA
        # plot):
        if self.group_col_id is not None:
            _, groups = self.split_dataset_and_variable(self.dataset, self.group_col_id)
        # If labels are not given at all, assign each point the same label:
        else:
            self.group_col_id = "Outcome"
            groups = pd.DataFrame(data=np.zeros(len(self.dataset)), index=self.dataset.index,
                                  columns=[self.group_col_id])

        # Define variables to store the master lists of loadings dataframes and the master list of PLSR dataframes:
        loadings_df_list1, loadings_df_list2, plsr_df_list = [], [], []
        for index, variable in enumerate(variable_list):
            # Convert labels to numeric if they are categorical (but maintain them as a DataFrame) because PLSDA
            # requires a numerical response value to regress on.
            var_name = variable.columns[0]
            if variable.iloc[:, 0].dtype == 'object':
                numeric_labels = variable.groupby(variable[var_name], sort=False).ngroup()
            else:
                numeric_labels = variable

            # Remove nonnumeric and NaN columns:
            data = self.remove_nonnumeric(data_list[index])
            data = self.remove_NaNcols(data)

            # PLS-DA! Use n latent variables (LVs)/components. scale=True to apply StandardScaler():
            plsr = PLSRegression(n_components=n_components, scale=True)
            plsr.fit(data.values, numeric_labels.values)

            plsr_df = pd.DataFrame(data=plsr.x_scores_,
                                   columns=['Latent Variable {}'.format(x + 1) for x in range(n_components)],
                                   index=data.index)
            plsr_df_list.append(plsr_df)

            # PLSR variance explained:
            var_in_x = np.var(plsr.x_scores_, axis=0)
            frac_var_explained = var_in_x / sum(var_in_x)

            # Calculate the 95% confidence ellipse bounds for the first two principal components for each group (
            # which will be plotted):
            # If find_clusters is True, perform k-means clustering on the data (NOT the PLS representation) and draw
            # ellipses for each cluster. If False, use the groups list to draw ellipses.
            if find_clusters:
                cluster_labels_df = pd.DataFrame(data=self.kmeans_clustering(data=plsr_df, n_clusters=num_clusters),
                                                 columns=["Value"], index=data.index)
                group_labels = cluster_labels_df["Value"]
                # Add cluster labels to the PLSR dataframe for potential future use:
                plsr_df["PLSR clusters"] = group_labels
                unique_groups = sorted(set(group_labels))
            else:
                group_labels = groups[self.group_col_id]
                unique_groups = sorted(set(group_labels))

            def draw_ellipses_PLS(color_list=None, facecolor=False):
                '''
                Function to draw ellipses on a PCA plot.
                :param color_list: Can optionally supply the colors that will be used to color the ellipses.
                :param facecolor: Can be used to color in the ellipses.
                :return: List of ellipses.
                '''
                # List variable to store ellipse objects for each group:
                ellipses = []
                for idx, group in enumerate(unique_groups):
                    plsr_df_subset = plsr_df[group_labels == group]
                    lv_1_vals = list(plsr_df_subset.iloc[:, 0])
                    lv_2_vals = list(plsr_df_subset.iloc[:, 1])
                    covariance = np.cov(lv_1_vals, lv_2_vals)
                    # The eigenvalues of the covariance matrix represent the magnitude of the variance in the
                    # direction of the principal components.
                    lambda_, v = np.linalg.eig(covariance)
                    # And take the square root of the variance metric to obtain the standard deviation for plotting.
                    lambda_ = np.sqrt(lambda_)
                    # Compute the angle (correct orientation of the ellipse):
                    theta = np.degrees(np.arctan2(*v[:, 0][::-1]))

                    # Ellipse covering two standard deviations:
                    if color_list is None:
                        # If more than four groups, use a random permutation of mcolors keys to determine the other
                        # colors.
                        CCS4_colors = list(mcolors.CSS4_COLORS.keys())
                        color_list = ["b", "g", "r", "purple", "orange", *random.sample(CCS4_colors, len(CCS4_colors))]
                    # xy = coordinates of the center, width = length of horizontal axis, height = length of vertical
                    # axis, angle = rotation in degrees.
                    ellipses.append(Ellipse(xy=(np.mean(lv_1_vals), np.mean(lv_2_vals)), width=lambda_[0] * 2 * 2,
                                            height=lambda_[1] * 2 * 2, angle=theta, fill=facecolor,
                                            edgecolor='k' if facecolor else color_list[idx], facecolor=color_list[idx],
                                            alpha=0.25 if facecolor else 1.0, linewidth=1.0))
                return ellipses


            if n_components > 1:
                # Set up the plot to visualize the latent variables in 2D space:
                if var_for_regression is None:
                    var_for_regression = [self.group_col_id]
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.set_xlabel('Latent Variable 1 ({:.2f}%)'.format(frac_var_explained[0]*100), fontsize=16)
                ax.set_ylabel('Latent Variable 2 ({:.2f}%)'.format(frac_var_explained[1]*100), fontsize=16)
                if title is not None:
                    ax.set_title(title.replace("_", " "), fontsize=fontsize)
                # else:
                # ax.set_title('Partial Least Squares Regression. Variable Regressed On: {}'.format(
                # var_for_regression[index].replace("_", " ")), fontsize=fontsize)

                # Add text labels if applicable:
                if sample_IDs is not None:
                    for n, *vals in enumerate(plsr_df.values):
                        point_label = sample_IDs.values[n]
                        ax.annotate(point_label, xy=(vals[0][0], vals[0][1]), xytext=(7.5, -2.5),
                                    textcoords='offset points',
                                    fontsize=10)  # vals[0][0], vals[0][1] are the first latent variables. xytext
                        # specifies label offset.

                palette = dict(zip(unique_groups, self.colors))
                # NOTE: have the option to color the points on a gradient, using the values of a variable specified
                # by the key parameter.
                sns.scatterplot(x="Latent Variable 1", y="Latent Variable 2",
                                hue=hue_list[index].iloc[:, 0] if key is not None else groups.iloc[:, 0],
                                palette=palette if key is None else "plasma_r", data=plsr_df,
                                legend="full", edgecolor='k', s=400 if key is not None else 200, ax=ax)
                if key is not None:
                    ax.get_legend().remove()
                    norm = plt.Normalize(hue_list[index].min(), hue_list[index].max())
                    cbar = plt.cm.ScalarMappable(cmap="plasma_r", norm=norm)
                    cb = ax.figure.colorbar(cbar)
                    cb.set_label('Day 10 AUC of log(sgRNA in nasal swab)', labelpad=20)

                # Plot 95% confidence ellipse on data:
                if draw_ellipses:
                    ellipses = draw_ellipses_PLS(list(palette.values()), facecolor=True)
                    for ellipse in ellipses:
                        # Copy the ellipse (can't put a single artist in more than one figure):
                        ellipse_copy = copy(ellipse)
                        ax.add_patch(ellipse_copy)
            else:   # Plot the PLS scores as a scatterplot, with samples on the y-axis
                plsr_df['Group'] = self.dataset[self.group_col_id]
                palette = dict(zip(unique_groups, self.colors))
                # For some reason, catplot's default is a despined plot? Draw in borders:
                sns.set_style('dark', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
                sns.catplot(x="Latent Variable 1", y=plsr_df.index, data=plsr_df, s=20, hue="Group", palette=palette,
                            legend=False, linewidth=2)
                plt.tight_layout()

            # plt.setp(ax.get_legend().get_texts(), fontsize=16)
            plt.show()

            # PLSR loadings:
            # Loadings represent the weight of each variable, e.g. how much it contributes to the corresponding
            # principal component.
            loadings = plsr.x_loadings_
            loadings_df = pd.DataFrame(data=loadings, index=data.columns,
                                       columns=['Latent Variable {}'.format(x + 1) for x in range(n_components)])
            # Sort loadings in descending order to see which variable covaries the most with the response (do this
            # twice to look at the largest absolute-value loadings and also the largest positive loadings):
            # argsort sorts in ascending order; read from the "tail" end to sort descending:
            loadings_df_1 = loadings_df.iloc[loadings_df["Latent Variable 1"].abs().argsort()[::-1]]
            loadings_df_list1.append(loadings_df_1)
            loadings_df_2 = loadings_df.iloc[loadings_df["Latent Variable 2"].abs().argsort()[::-1]] if n_components \
                                                                                                        > 1 else None
            loadings_df_list2.append(loadings_df_2) if n_components > 1 else None
            loadings = [loadings_df_1] if loadings_df_2 is None else [loadings_df_1, loadings_df_2]

            # Make a bar chart to look at the loadings that are largest (positive and negative), and another bar
            # chart to look at those that are most positive.
            for index, df in enumerate(loadings):
                fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 6))
                ax.set_xlabel('Loading Weight', fontsize=16)
                # Add ax.set_xticklabels()
                ax.set_ylabel('Variable Name', fontsize=16)
                df.index = [name.replace("_", " ") for name in list(df.index)]
                ax.set_title('Partial Least Squares Loadings', fontsize=16)
                # Take the first n loadings, where n = n_loadings parameter.
                if n_loadings > len(df): n_loadings = len(df)
                df = df[:n_loadings]
                sns.barplot(data=df, x='Latent Variable {}'.format(index + 1), y=df.index, palette=self.colormap, ax=ax,
                            edgecolor=".2")
                plt.tight_layout()
                plt.show()
            plt.show()

            # Also plot the loadings on a biplot:
            self.latent_biplot(scores=plsr_df, coeffs=loadings_df, labels=groups)

        return loadings_df_list1, plsr_df_list


    def comparison_plots(self, features_to_plot=None, type="Boxplots", nested_grouping=False, x_labels=None,
                         y_labels=None, titles=None, annot=True, save_keyword=None):
        '''
        Master comparison plotting function, to draw violin plot(s), boxplot(s) or box plot(s) comparing values of a
        variable of interest between different groups (groups). Outcomes on x-axis, magnitude of the variable response
        on y-axis.
        :param features_to_plot: List of the names of features to make plots of.
        :param type: Kind of plots to generate. If not provided, will default to bar plots.
        :param nested_grouping: If nested grouping is desired (i.e. the group groups can be split further by another
        variable- e.g. measurements are taken over time), this argument can be set to True to indicate that this is
        the case.
        :param x_labels: Optional list, should either be length 1 (to apply universal x-axis label to all plots) or
        the same length as features_to_plot (to give a 1:1 x-axis label to each plot).
        :param y_labels: Optional list, should either be length 1 (to apply universal y-axis label to all plots) or
        the same length as features_to_plot (to give a 1:1 y-axis label to each plot).
        :param titles: Optional list, should either be length 1 (to apply universal title to all plots) or the same
        length as features_to_plot (to give a 1:1 title to each plot).
        :param annot: Flag to set whether statistical annotations display on the plot.
        :param save_keyword: Defaults to None, but can pass a string instead in the case that there are too many
        groups to visualize using add_stat_annotation. This string is added to the assigned basename (
        "SystemsSerologyComparisonSignificance") to create the final save name.
        :return: None, but will display a plot.
        '''
        sns.set_style("white")
        plt.rcParams["axes.labelsize"] = 17
        # If nested_grouping_vars is set to True, treat the features (columns) as all being measurements of the same
        # variable over different conditions (e.g. time, dose, etc.). Melt the dataframe to a form more amenable to
        # side-by-side plotting, and plot on the same graph:
        if nested_grouping is True:
            # Use all of the numerical columns in the plotting procedure:
            # Make a copy of self.dataset to avoid altering the dataset in-place:
            data = self.remove_invalidrows(self.dataset[:])
            features_to_plot = self.remove_NaNcols(self.remove_nonnumeric(data)).columns
            if (x_labels is not None and len([x_labels]) > 1) or (y_labels is not None and len([y_labels]) > 1) or (
                    titles is not None and len([titles]) > 1):
                print(
                    "If nested_grouping is True, one plot will be generated; only one label will be used for the "
                    "title, x- and y-axes. Exiting.")
                sys.exit(1)
            else:
                x_labels = x_labels if x_labels is not None else "Variable"
                y_labels = y_labels if y_labels is not None else "Units"
                title = titles if titles is not None else "Comparison Plot"

            df_to_plot = data[features_to_plot]
            df_to_plot["Outcomes"] = data[self.group_col_id]
            melted_cols = df_to_plot.melt(id_vars=["Outcomes"], value_vars=features_to_plot, value_name=y_labels,
                                          var_name=x_labels)
            # Instantiate the plot:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            if type == "Boxplots":
                sns.boxplot(data=melted_cols, x="Outcomes", y=y_labels, hue=x_labels, ax=ax, palette=self.colors)

                for patch in ax.artists:
                    r, g, b, a = patch.get_facecolor()
                    patch.set_facecolor((r, g, b, .3))
            elif type == "Violins":
                sns.violinplot(data=melted_cols, x="Outcomes", y=y_labels, hue=x_labels, ax=ax, palette=self.colors)
            plt.show()

        elif nested_grouping == False and features_to_plot is not None:
            if not isinstance(features_to_plot, list): features_to_plot = [features_to_plot]
            # Make a copy of self.dataset to avoid altering the dataset in-place:
            data = self.remove_invalidrows(self.dataset[:])

            # If "all" is given for features to plot, plot all numeric columns:
            if features_to_plot[0].casefold() == "all":
                features_to_plot = self.remove_NaNcols(self.remove_nonnumeric(data)).columns

            # Else make plots only for the specified variables:
            # If the variable is not in the dataset, check to see if another dataset was provided, and search that
            # dataset for the column in question. If the feature of interest is in that dataset, copy it to the
            # working dataset temporarily.
            # Make a copy of self.dataset to avoid altering the dataset in-place:
            data = self.remove_invalidrows(self.dataset[:])
            for var in features_to_plot:
                if var not in data.columns and self.second_dataset is not None:
                    data[var] = self.second_dataset.loc[
                        data.index, var]  # .loc[data.index, ] to deal with situations in which pre-processing has
                    # rendered the two datasets to be different in size.

            # Separate plot for each feature:
            for index, feature in enumerate(features_to_plot):
                feat_to_plot = pd.DataFrame(data.loc[:, feature])
                feat_to_plot["Outcomes"] = data[self.group_col_id]
                # For plotting purposes:
                min_value = data.loc[:, feature].min()
                max_value = data.loc[:, feature].max()
                dev = data.loc[:, feature].std() * len(set(feat_to_plot["Outcomes"]))

                # Instantiate the plot:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                # ax.set_title("Comparison of {}".format(feature.replace("_", " ")), fontsize=16)
                if type == "Boxplots":
                    sns.boxplot(data=feat_to_plot, x="Outcomes", y=feature, ax=ax, palette=self.colors, linewidth=1.75)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    # I find it more aesthetically pleasing for the boxplots to have some transparency:
                    for patch in ax.artists:
                        r, g, b, a = patch.get_facecolor()
                        patch.set_facecolor((r, g, b, .4))
                    # And swarmplot:
                    #sns.swarmplot(data=feat_to_plot, x="Outcomes", y=feature, ax=ax, palette=self.colors,
                                  #edgecolor='black', linewidth=2, size=10)
                elif type == "Violins":
                    sns.violinplot(data=feat_to_plot, x="Outcomes", y=feature, ax=ax, palette=self.colors)
                    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    # And swarmplot:
                    sns.swarmplot(data=feat_to_plot, x="Outcomes", y=feature, ax=ax, palette=self.colors,
                                  edgecolor='black', linewidth=1.25, size=8)
                elif type == "Swarm":
                    sns.swarmplot(data=feat_to_plot, x="Outcomes", y=feature, ax=ax, palette=self.colors,
                                  edgecolor='black', linewidth=1, size=10)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                elif type == "Strip":
                    sns.stripplot(data=feat_to_plot, x="Outcomes", y=feature, ax=ax, palette=self.colors,
                                  edgecolor='black', linewidth=1.5, size=10)
                    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

                if annot:
                    # Statistical testing annotations (don't do this if there are more than four groups; the annotations
                    # become too occlusive:
                    # Find all pairs of unique groups:
                    group_pairs_list = []
                    group_pairs_list.extend(list(combinations(set(feat_to_plot["Outcomes"]), 2)))
                    if len(set(feat_to_plot["Outcomes"])) < 5:
                        add_stat_annotation(ax=ax, data=feat_to_plot, x="Outcomes", y=feature, box_pairs=group_pairs_list,
                                            test="Mann-Whitney", line_offset=0.05, line_offset_to_box=0.05,
                                            text_format="full", loc="inside", fontsize=14, verbose=2)
                    else:
                        # Don't plot the statistical annotations, but return a dataframe containing statistical
                        # significance information.
                        significance_df = pd.DataFrame(columns=['Var1', 'Var2', 'MWW statistic', 'p-value', '<0.05?'])
                        for pair in group_pairs_list:
                            data_subset = data[data[self.group_col_id] == pair[0]][feature]
                            data_comparison = data[data[self.group_col_id] == pair[1]][feature]
                            u, p = mannwhitneyu(data_subset.values, data_comparison.values)
                            flag = True if p < 0.05 else False
                            significance_df = significance_df.append(
                                pd.DataFrame(np.array([pair[0], pair[1], u, p, flag]).reshape(-1, 5),
                                            columns=significance_df.columns)).reset_index(drop=True)
                        # Save significance_df:
                        processed_folder_path = os.path.join(str(Path().absolute()), "Processed_Data",
                                                             "Significance_Testing")
                        if save_keyword is None:
                            save_keyword = ""
                        if not os.path.exists(processed_folder_path):
                            os.mkdir(processed_folder_path)
                        output_basename = "SystemsSerologyComparisonSignificance"
                        output_path = os.path.join(processed_folder_path, output_basename + save_keyword + ".csv")
                        significance_df.to_csv(output_path)
                ax.set(xlabel='Groups', ylabel='{}'.format(feature.replace("_", " ")))
                # ax.set_ylim(min_value-data.loc[:, feature].std(), max_value + dev)
                ax.tick_params(labelsize=14)
                plt.tight_layout()
                plt.show()

                # ADD THE OPTION TO SAVE THESE FIGURES.
        else:
            print("No features were given to plot. Exiting.")
            return


    # Rose plots:
    # The magnitude of the plot will be given by percentile, so at least two different groups must be given.
    def flower_plots(self, features_to_exclude=None, group_keys=None, metric="percentile", fig_layout="subplots"):
        '''
        Function to draw polar/rose/nightingale plots to compare the values of the same variables in different
        conditions.
        :param features_to_exclude: For this function, the default behavior will be to include every feature on the
        plot. Column identifiers can optionally be given for the variables not to include in the plots.
        :param group_keys: Can optionally specify strings that will be used to group variables together (otherwise,
        each individual variable will be plotted as a different color).
        :param metric: Metric to plot on the polar plots. Options are "percentile", "median", or "mean".
        :param fig_layout: Choose between "subplots" (which will plot each group as its own subplot) or "together" (
        which will overlay all groups onto the same plot). No other valid options.
        :return: None, but will display a plot.
        '''
        # Casefold metric (to avoid errors resulting from case mistakes and nothing else):
        metric = metric.casefold()

        # Use self.group_col_id to define the different conditions.
        if self.group_col_id is None:
            raise ValueError("No column ID was given to split samples into groups; no comparison can be made. Exiting.")
        else:
            data = self.dataset[self.dataset.columns[
                ~self.dataset.columns.isin(features_to_exclude)]] if features_to_exclude is not None else self.dataset
            # Remove nonnumeric and NaN columns, as well as rows with invalid values (e.g. inf/-inf) in them:
            data = self.remove_invalidrows(self.remove_NaNcols(data))

            split_dfs, groups = SystemsSerologyPreprocessing(data=data, ignore_index=True).split_data(
                split_colname=self.group_col_id)
            # Remove nonnumeric columns:
            split_dfs = [self.remove_nonnumeric(df) for df in split_dfs]

            # Group variables together based on key words.
            # Initialize column containing group keys:
            var_groups_list = np.zeros(len(split_dfs[0].columns))  # all split_dfs should have the same features,
            # so any of them could be used here.
            # Separate list containing corresponding labels based on the group keys (for use in creating the legend).
            # NOTE: this is fairly crude for now and assumes the only things being measured are titer, Fc binding,
            # function.
            legend_labels_dict = {}
            # Default label will be Effector Function.
            legend_labels_dict[0] = "Effector Function"
            for key_idx, key in enumerate(group_keys):
                # If all keys are titers, Fc binding, etc., assign each feature/group of features specified by key to its
                # own group:
                if all("ig" in key.casefold() for key in group_keys) or all("fc" in key.casefold() for key in group_keys):
                    legend_labels_dict[key_idx + 1] = key
                elif "igg" in key.casefold():
                    # Use a dictionary to ensure the labels in var_groups_list match in the legend labels list.
                    legend_labels_dict[key_idx + 1] = "IgG Titer"
                elif "igm" in key.casefold():
                    legend_labels_dict[key_idx + 1] = "IgM Titer"
                elif "iga" in key.casefold():
                    legend_labels_dict[key_idx + 1] = "IgA Titer"
                elif "ig" in key.casefold():
                    legend_labels_dict[key_idx + 1] = "Antibody Titer"
                elif "fc" in key.casefold():
                    legend_labels_dict[key_idx + 1] = "FcR Binding"
                elif "ad" in key.casefold() or "nk" in key.casefold():
                    legend_labels_dict[key_idx + 1] = "Effector Function"
                elif "neut" in key.casefold() or "nab" in key.casefold():
                    legend_labels_dict[key_idx + 1] = "Neutralizing Titer"
                elif "cell" in key.casefold() or "elispot" in key.casefold():
                    legend_labels_dict[key_idx + 1] = "T cell"
                else:
                    legend_labels_dict[key_idx + 1] = key
                for index, column in enumerate(split_dfs[0].columns):
                    if key.casefold() in column.casefold():
                        var_groups_list[index] = key_idx + 1
                    else:
                        pass

            legend_labels_list = [legend_labels_dict[key] for key in var_groups_list]
            # Find the first occurrence of each legend label:
            first_occurrence = []
            for label in set(legend_labels_list):
                indices = [i for i, x in enumerate(legend_labels_list) if x == label]
                first_occurrence.append(indices[0])

            '''           
            # Crude way of replacing "left over" zeros with unique group labels:
            labels_list = list(range(len(group_keys) + 1, 100))
            for index, value in enumerate(var_groups_list):
                if value == 0:
                    var_groups_list[index] = labels_list[0]
                    labels_list.pop(0)'''

            # Modify columns bsaed on the input to metric:
            # New variable to store modified dataframes in:
            split_processed = []
            if metric == 'percentile':
                # Normalize the size of the bars/allow features on dissimilar scales to be plotted next to one another:
                # For each feature, find the maximum value for that feature for any sample:
                max_val = pd.concat(split_dfs).apply(np.max, axis=0)
                # For each feature, divide the values by the maximum value to find the percentile, and then find the mean
                # percentile for each feature.
                for index, df in enumerate(split_dfs):
                    for var in max_val.index:
                        df[var] = df[var].apply(lambda x: x / max_val[var])
                    split_processed.append(df.apply(np.mean, axis=0).to_frame(name="Value"))
            elif metric == 'median':
                for df in split_dfs:
                    split_processed.append(df.apply(np.median, axis=0).to_frame(name="Value"))
            elif metric == 'mean':
                for df in split_dfs:
                    split_processed.append(df.apply(np.mean, axis=0).to_frame(name="Value"))
            else:
                raise ValueError("Invalid argument to 'metric' argument. Options are percentile, mean, or median.")
            # For plotting purposes, find the max value in any dataframe:
            max_val = pd.concat(split_processed, axis=0).max().values[0]

            # Sort the values in each dataframe by group label and then by alphabetical order of index name:
            for df in split_processed:
                df["Var group label"] = var_groups_list
                df = df.sort_values("Var group label").groupby("Var group label", sort=False).apply(
                    lambda x: x.sort_index())

            # Plotting:
            # Set specs to "polar" for plotting; this requires {type: polar} dict to be given for each subplot. With
            # a variable number of subplots, do this using a list:
            specs_list = []
            for i in range(len(groups)):
                specs_list.append({"type": "polar"}.copy())

            if fig_layout.casefold() == "subplots":
                fig = make_subplots(rows=1, cols=len(groups), specs=[specs_list], subplot_titles=list(
                    groups), horizontal_spacing=0.12, vertical_spacing=0.1)  # for index in range(len(groups)):  #
                # fig.layout.annotations[index].update(y=0.8)
            else:
                fig = make_subplots(rows=1, cols=1)

            for index, df in enumerate(split_processed):
                vals_to_plot = df["Value"].values
                # One polar bar for each variable:
                num_slices = len(df)
                # Define the angle of the center of each slice (offset of 1.5 ensures that the right edge of the
                # first slice is at 0 degrees).
                theta = [(i) * 360 / num_slices for i in range(num_slices)]
                width = [360 / num_slices for _ in range(num_slices)]

                # Generate list of colors based on group keys:
                # NOTE: sometime later, can also try building a custom color palette. To do this, just define a list
                # of CSS colors:
                # e.g. ['#636EFA', '#EF553B'], etc.
                if fig_layout.casefold() == "subplots":
                    # Seaborn Pastel1 palette:
                    color_seq = self.colors  # color_seq = ["#CC99FF", "#FFCC99", "#99FF99", "#99CCFF", "#FF6666",
                    # "#FFB366", "#FFFF66",  # "B3FF66"] # etc.
                else:
                    palettes = [px.colors.qualitative.Pastel, px.colors.qualitative.Set1, px.colors.qualitative.Bold]
                    color_seq = palettes[index]

                # color_indices = range(0, len(color_seq), len(color_seq) // len(set(var_groups_list)))
                color_indices = range(0, len(set(var_groups_list)))
                unique_colors = [color_seq[i] for i in color_indices]
                # Assign each feature a color based on its grouping:
                colors_dict = dict(zip(set(df["Var group label"]), unique_colors))
                colors = list(map(colors_dict.get, df["Var group label"]))

                '''
                colors = np.zeros(len(var_groups_list))
                # Convert list of colors to list of strings:
                colors = ['{:f}'.format(x) for x in colors]
                for color_index, color in enumerate(unique_colors):
                    for group_index, group in enumerate(df["Var group label"]):
                        if group == color_index:
                            colors[group_index] = unique_colors[color_index]'''

                # For now, based on the name of the feature- extract feature name up to first space:
                ticks = [index.split("_")[0] for index in df.index]
                #ticks = [' '.join([x[0], x[1]]) for x in df.index.str.split("_")]
                # This is easy- tickvals are just equal to theta and don't need to be adjusted in this case.
                # When plotting multiple variables of the same category together (e.g. IgG4 S1 vs. RBD), need to adjust
                # the method:
                if len(set(ticks)) < len(df.index):
                    # Find all indices that are identical and get the median (round down if odd number of indices):
                    medians = []
                    for ticklabel in set(ticks):
                        repeated_indices = [i for i, value in enumerate(ticks) if value == ticklabel]
                        medians.append(np.floor(statistics.median(repeated_indices)).astype(int))
                    # Subset the ticks list and the thetaval list using the median values:
                    ticks = [ticks[median] for median in medians]
                    theta_subset = [theta[median] for median in medians]

                    colors_dict = dict(zip(set(df["Var group label"]), unique_colors))
                    colors = list(map(colors_dict.get, df["Var group label"]))
                else:
                    theta_subset = theta

                if fig_layout.casefold() == "subplots":
                    # NOTE: plotly treats all slices in the polar plot as a single trace; not ideal for assigning
                    # legends, etc. Need to make a separate barpolar object for each "group".
                    # Do not plot legend information for every trace; only show for the first occurrence of each group.
                    barpolar_plots = [go.Barpolar(r=[r], theta=[t], width=[w], name=n, marker_color=[c],
                                                  showlegend=True if i in first_occurrence and index == 0 else False,
                                                  marker_line_color='black') for r, t, w, (i, n), c in
                                      zip(vals_to_plot, theta, width, enumerate(legend_labels_list), colors)]
                    for idx, plot in enumerate(barpolar_plots):
                        fig.add_trace(plot, row=1, col=index + 1)
                else:
                    fig.add_trace(go.Barpolar(r=vals_to_plot, theta=theta, width=width, marker_color=colors))

                # Automate text position and font size based on the number of plots being generated:
                tick_fontsize, title_fontsize, legend_fontsize = np.linspace(20,12,num=5), np.linspace(32,24,num=5), \
                                                                 np.linspace(14,12,num=5)
                title_ypos = np.linspace(1, 0.6, num=5)
                plot_max_val = 0.7 if max_val < 0.7 else np.floor(max_val+1)
                fig.update_polars(radialaxis=dict(range=[0, plot_max_val]),
                                  angularaxis=dict(tickvals=theta_subset, ticktext=ticks,
                                                   tickfont_size=tick_fontsize[len(split_processed)-2]))
                fig.update_annotations(font_size=title_fontsize[len(split_processed)-2], y=title_ypos[len(
                    split_processed)-2])
                fig.update_layout(title_x=0.5, title_y=0.7,
                                  title_font_size=title_fontsize[len(split_processed)-2],
                                  legend_y=0.25, legend_font_size=legend_fontsize[len(split_processed)-2],
                                  margin_l=200, margin_r=100)
            fig.show()


    # Fancy heatmap:
    @staticmethod
    def heatmap(x, y, size, color=None):
        '''
        Function to make a heatmap using the scatterplot function, fancified and with different colors/sizes that
        can be specified.
        :param x: Some kind of list or array-like specifying what to plot/label on the x-axis.
        :param y: Some kind of list or array-like specifying what to plot/label on the y-axis.
        :param size: Some kind of list or array-like indicating the size of each point.
        :return: None (but will plot heatmap).
        '''
        fig, ax = plt.subplots(figsize=(1.5 * len(x.unique()), 1.2 * len(y.unique())))
        # First, map from the column names to integer coordinates (to define the locations to plot each point):
        x_labels = [v for v in sorted(x.unique())]
        y_labels = [v for v in sorted(y.unique())]
        x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
        y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}
        print(y_to_num)

        # With this size scale, the assumption is that the values are standardized in some way. Change this around if
        # this is not the case.
        size_scale = 6000 / len(y.unique())
        plot = ax.scatter(x=x.map(x_to_num),  # Use mapping for x
            y=y.map(y_to_num),  # Use mapping for y
            s=size * size_scale,  # Vector of dot sizes, proportional to size parameter
            c=color if color is not None else size,  # If not given, color by size.
            cmap='Reds', edgecolor='black', linewidth=1.5)

        # Center the points:
        ax.grid(False, 'major')
        # ax.grid(True, 'minor')
        # Show column labels on the axes:
        ax.set_xticks([x_to_num[v] for v in x_labels])
        ax.set_xticklabels([x.replace("_", " ") for x in x_labels], rotation=45, horizontalalignment='right')
        ax.set_yticks([y_to_num[v] for v in y_labels])
        ax.set_yticklabels([y.replace("_", " ") for y in y_labels])
        # A bunch of formatting to get this to look good...
        ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
        ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
        ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
        ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
        cbar = plt.colorbar(plot)
        cbar.set_label("Standardized Z-score", rotation=90, labelpad=20)
        plt.tight_layout()
        plt.show()

    # Dot plots:
    def dotplots(self, variables_of_interest=None, keywords=None):
        '''
        Function to plot dotplots of variables of interest- multiple dotplots will be generated, one for each group,
        and each row of each dotplot will correspond to an individual in that group.
        :param variables_of_interest: Individual variables of interest to plot. Defaults to None.
        :param keywords: If given, these keywords will be passed to self.variable_enrichment() to create a composite
        variable score. Defaults to None.
        :return:
        '''
        # Both variables of interest and keywords cannot be left as None:
        if variables_of_interest is None and keywords is None:
            print("No variables to plot. Please specify variables of interest or define keywords to group variables "
                  "together.")
            sys.exit()

        if variables_of_interest is not None:
            features_of_interest = variables_of_interest

        if keywords is not None:
            keywords = [kw.replace("_", " ") for kw in keywords]
            features_of_interest = [col for col in self.dataset.columns if
                                    any(keyword.casefold() in col.casefold() for keyword in keywords)]

        # Convert all features to percentiles by scaling down to 0-1:
        dataset = self.dataset.copy()
        for feature in dataset:
            if dataset[feature].dtype != 'object':
                # Percentiles:
                # dataset[feature] = dataset[feature].apply(lambda x: x / np.max(dataset[feature]))
                # Z-score:
                dataset[feature] = (dataset[feature] - dataset[feature].mean()) / dataset[feature].std(ddof=1)
                # For plotting purposes, now percentile rank the Z-scores:
                dataset[feature] = (dataset[feature] - np.min(dataset[feature])) / (
                            np.max(dataset[feature]) - np.min(dataset[feature]))

        # Subset by group:
        for group in set(dataset[self.group_col_id]):
            group_df = dataset[dataset[self.group_col_id] == group]
            group_df = group_df[features_of_interest]
            # Pivot the dataframe so that we can get a list for both x and y:
            if self.labels_col_id.casefold() == "index":
                melted_group_df = pd.melt(group_df.reset_index(), id_vars=dataset.index.name)
            else:
                melted_group_df = pd.melt()
            # Plot!
            self.heatmap(x=melted_group_df['variable'], y=melted_group_df[dataset.index.name],
                         size=melted_group_df['value'])


    # Function to draw heatmaps (can be self- or cross-correlation heatmaps, or z-score heatmaps):
    def heatmap_master(self, group_keys=None, type="self", title=None, other_secondary_label="Other"):
        '''
        Function to compute a correlations heatmap using all of the numerical features in a dataset.
        :param group_keys: Can be used to structure the heatmap such that similar columns are grouped together. A
        list of key strings; columns will be grouped based on partial match to elements of the list.
        :param type: Can be "self", "cross", "z" to specify the type of heatmap that should be generated (self and
        cross for correlation heatmaps, for between columns of one dataframe or between columns of two dataframes,
        respectively, or z for the processed values of one dataframe).
        :param title: Optional argument to give the heatmap a custom title. Defaults to None.
        :param other_secondary_label: For the secondary label axis- label to give features when the feature doesn't
        contain any of the key terms. Defaults to "other".
        :return: None.
        '''
        # Make a copy of self.dataset to avoid altering the dataset in-place:
        data = self.dataset[:]
        second_dataset = None
        if type not in ["self", "cross", "z"]:
            return ValueError("Invalid string passed to type argument. Options: self, cross, or z.")
        if type.casefold() == "cross" and self.second_dataset is None:
            print("Cross-correlation was specified, but a second dataset was not given.")
            sys.exit()
        elif type.casefold() == "cross":
            second_dataset = self.second_dataset[:]

        # Preprocessing (remove _ characters from input strings, group features based on whether they contain certain
        # keys, so long as group_keys is not None):
        # Save a copy of the dataset with all columns, not just numeric (to be able to retrieve label information in
        # the case that type is "z"):
        data_w_labels = data[:]
        data = self.remove_NaNcols(self.remove_nonnumeric(data))
        # Remove rows with invalid values (NaN, inf, or -inf):
        data = self.remove_invalidrows(data)
        if group_keys is not None:
            data = self.group_features(dataset=data, group_keys=group_keys)

        if other_secondary_label is None:
            other_secondary_label = "Other"
        other_secondary_label = other_secondary_label.replace("_", " ")
        if title is not None:
            title = title.replace("_", " ")

        # Labels for the primary axis:
        x = [col.replace("_", " ") for col in data.columns]
        # Labels for a secondary axis (if group keys are given, look for common serological terms (e.g. IgG, FcgR3,
        # etc.) in the names of the primary axis labels to assign an appropriate secondary axis label):
        secondary_labels = []
        key_terms = ["IgG1", "IgG2", "IgG3", "IgG4", "IgG", "IgA", "IgM", "ADCD", "ADCP", "ADNP", "NK", "FcgR2",
                     "FcgR3", "FcR2", "FcR3"]
        for column in data.columns:
            matched_key = next((key for key in key_terms if key.casefold() in column.casefold()),
                               other_secondary_label)
            matched_key = "Function" if any(
                str in matched_key for str in ["ADNP", "ADCP", "ADCD", "NK"]) else matched_key
            secondary_labels.append(matched_key)
        labels = [secondary_labels, x] if len(data.columns) > 3 else x

        # Repeat all of these pre-processing steps for the second dataset, if it exists:
        if second_dataset is not None:
            second_data = self.remove_NaNcols(self.remove_nonnumeric(second_dataset))
            if group_keys is not None:
                second_data = self.group_features(dataset=second_data, group_keys=group_keys)

            # Labels for the primary axis:
            y = [col.replace("_", " ") for col in second_data.columns]
            # Labels for a secondary axis (if group keys are given, look for common serological terms (e.g. IgG, FcgR3,
            # etc.) in the names of the primary axis labels to assign an appropriate secondary axis label):
            secondary_labels = []
            key_terms = ["IgG1", "IgG2", "IgG3", "IgG4", "IgG", "IgA", "IgM", "ADCD", "ADCP", "ADNP", "NK", "FcgR2",
                         "FcgR3", "FcR2", "FcR3"]
            for column in second_data.columns:
                matched_key = next((key for key in key_terms if key.casefold() in column.casefold()), other_secondary_label)
                matched_key = "Function" if any(
                    str in matched_key for str in ["ADNP", "ADCP", "ADCD", "NK"]) else matched_key
                secondary_labels.append(matched_key)
            second_data_labels = [secondary_labels, y] if len(second_data.columns) > 3 else y
            cross_corr = pd.concat([data, second_dataset], axis=1, keys=['df1', 'df2']).corr().loc['df2', 'df1']

        # If type is "self" or "cross", compute the correlations array.
        if type != "z":
            to_plot = data.corr() if second_dataset is None else cross_corr
        # Else perform processing needed to set up the dataframe for the heatmap; this means making sure the dataset is
        # z-scored by applying the z-scoring function. Then, if group_col_ID is not None, use these groups as a
        # secondary axis along the y. Else, don't include the secondary axis.
        else:
            to_plot = (data - data.mean()) / data.std()
            y = [idx.replace("_", " ") for idx in to_plot.index]
            if self.group_col_id is not None:
                groups = self.dataset[self.group_col_id]
                # Rearrange the rows of the dataframe based on their grouping, return rearranged data+group labels (
                # group labels will return as series):
                data, groups = self.group_samples(to_plot, groups=groups)
                # Z-scores in to_plot are no longer accurate, since the rows have been rearranged:
                to_plot = (data - data.mean()) / data.std()
                sample_labels = [groups.values, data.index]
            else:
                sample_labels = y

        # Plot heatmap using plotly:
        if type != 'z':
            fig = go.Figure(data=go.Heatmap(z=to_plot, x=labels, y=labels if second_dataset is None else
                second_data_labels, colorscale="reds", zmin=-1, zmax=1))
        else:
            fig = go.Figure(data=go.Heatmap(z=to_plot, x=labels, y=sample_labels, colorscale="reds"))
        fig.update_layout(title=dict(text="Heatmap" if title is None else title, \
                        font_size=18, x=0.55, y=0.95), height=1000, plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(tickfont_size=12, tickangle=90, automargin=True)
        fig.update_yaxes(tickfont_size=12, automargin=True)
        fig.show()





    def clustermap(self, title=None):
        '''
        Function to construct a heatmap comparing standardized features between .
        :param filler:
        :return:
        '''


    def most_predictive_features(self, features_of_interest=None, multiple_correction="bonferroni", title=None):
        '''
        A variety of functions to analyze which feature in the dataset is most predictive of group. Analyses include:
        analysis of differences in the average z-score of each feature between each group and the rest of the
        samples, log-fold change analysis of each feature between each group and the rest of the samples,
        and AUROC analysis using combinations of different features, using only those features to attempt to classify
        samples into groups (yes-in the group vs. no-not in the group).
        :param features_of_interest: Optional argument that allows for user specification of a list of column IDs,
        to find the most predictive variables from the features in this list.
        :param multiple_correction: String argument used to specify the type of multiple hypothesis correction to
        implement. Defaults to Bonferroni, options for now are Bonferroni or Holm-Bonferroni.
        :param title: The title of the plot.
        :return:
        '''
        if self.group_col_id is None:
            raise ValueError("No groups exist for which to compare average z-score. Exiting.")
        if features_of_interest is None:
            # If specific features are not given, use the entire dataset.
            features_of_interest = list(self.dataset.columns)
        groups = self.dataset[self.group_col_id]

        # Analysis 1: differences in average z-score:
        # First compute the z-scores for each feature for each sample:
        z_scores = pd.DataFrame()
        # Copy of the dataset to work with:
        data = self.dataset[:]
        data = data[features_of_interest]
        # Remove non-numeric and NaN columns:
        data = self.remove_NaNcols(self.remove_nonnumeric(data))
        # Z-score (formula: x - mean / stddev) (should I Z-score after subsetting? Or before?)
        for col in data.columns:
            z_scores[col] = (data[col] - data[col].mean()) / data[col].std(ddof=1)

        # Dictionary to store differences in average z-scores:
        z_score_difference_dict = {}
        for group in set(groups):
            # Split the dataframe into two separate dataframes: one for the group and one for the rest of the dataset
            # (the "field"). Apply the mean to get the average z-score for each feature. Subtract the field from the
            # group and take the absolute value to get a measure of the difference in average z-score:
            z_scores_group = z_scores[groups == group].apply(np.mean, axis=0)
            z_scores_field = z_scores[groups != group].apply(np.mean, axis=0)
            z_score_difference = abs(z_scores_group - z_scores_field)
            z_score_difference_dict[group] = z_score_difference

        # Display differences in average z-score using a heatmap:
        '''
        for key, df in z_score_difference_dict.items():
            x_labels = [index.replace("_", " ") for index in df.index]
            y_labels = title.replace("_", " ") if title is not None else "{} vs. other groups".format(key)
            fig = px.imshow([df.values], labels=dict(x="Feature", color="Difference in average z-score"), x=x_labels,
                            y=[y_labels], color_continuous_scale="Reds")

            fig.update_xaxes(tickangle=-45, title_font_size=32)
            fig.update_layout(xaxis=dict(tickfont_size=28), yaxis=dict(tickfont_size=28))
            fig.update_coloraxes(colorbar=dict(len=0.5), colorbar_title=dict(side="right", font_size=28))
            fig.show()'''

        # Analysis 2: log-fold difference, assessed with statistical testing and visualized using a volcano plot:
        # Note that log-fold differences are likely not normally distributed in our sample, so a Wilcoxon rank-sum (
        # aka Mann-Whitney U-test) should be used.
        # Two things to compute: ratio/log-fold change and p-value using a two-sided non-parametric significance test
        # (using the raw values from both groups).
        from scipy.stats import mannwhitneyu
        data = self.dataset[:]  # this is "resetting" the variable data (which is altered by the prev. analysis.
        # Might try to combine these two analyses later).
        data = data[features_of_interest]
        # Copy strictly for labeling and indexing purposes:
        data_copy = data[:]
        data_copy = self.remove_NaNcols(self.remove_nonnumeric(data_copy))

        # Dictionaries to store log-fold difference and p-values:
        log_diff_dict, p_val_dict = {}, {}
        # Adjust for multiple hypothesis correction to find the p-value to be considered significant:
        if multiple_correction == "bonferroni":
            bonferroni_alpha = -np.log10(0.05 / len(data.columns))
        elif "holm" in multiple_correction.casefold():
            # List of alpha values needed to satisfy Holm-Bonferroni significance criteria:
            holm_bonferroni_significance = -np.log10([0.05 / (len(data.columns)-rank+1) for rank in range(len(
                    data.columns))])

        # First, split dataset into group of interest and then the rest of the samples (the "field"):
        for group in set(groups):
            group_df = data[groups == group]
            field_df = data[groups != group]
            # Remove non-numeric and NaN columns:
            group_df, field_df = self.remove_NaNcols(self.remove_nonnumeric(group_df)), self.remove_NaNcols(
                self.remove_nonnumeric(field_df))
            # For each feature, compute the p-value:
            p_list = []
            for feature in group_df.columns:
                w, p = mannwhitneyu(group_df[feature], field_df[feature], alternative="two-sided")  # two-sided
                # because the group could be either less than or greater than the field.
                p_list.append(-np.log10(p))
            p_val_dict[group] = p_list

            # Compute the mean of each feature within each group, and use this information to compute the log-fold
            # difference:
            means_group = group_df.apply(np.mean, axis=0)
            means_field = field_df.apply(np.mean, axis=0)
            # Compute log-fold difference for the average values of each feature between the group of interest and
            # the field:
            log2_diff = np.log2(means_group.divide(means_field))
            # Store in the log-fold difference dictionary:
            log_diff_dict[group] = log2_diff

        # List of text labels for the scatterplot (don't label points that are in close proximity to other points):
        master_labels_list = []
        for p_vals, diff_vals in zip(p_val_dict.items(), log_diff_dict.items()):
            p_vals_copy, diff_vals_copy = list(p_vals[:][1]), list(diff_vals[:][1])
            labels = []
            # For each point, find the point that is closest to it:
            for p, d, label in zip(p_vals[1], diff_vals[1], list(data_copy.columns)):
                # Remove p and d from their respective lists (eventually, only one point that is close to any other
                # given points should be labeled). Do this before checking nearest neighbors because otherwise,
                # each point will count itself as its nearest neighbor:
                p_vals_copy.remove(p)
                diff_vals_copy.remove(d)
                # Compute the closest point to the point that was just removed:
                if len(p_vals_copy) > 0:  # since points are removed before the computation, check if the list is
                    # empty first:
                    closest_p = p_vals_copy[min(range(len(p_vals_copy)), key=lambda i: abs(p_vals_copy[i] - p))]
                    indices = [i for i, val in enumerate(p_vals_copy) if val == closest_p]
                    # Subset the fold-differences list to ensure the closest p and closest diff values are coming
                    # from the same point.
                    diff_vals_subset = list(map(diff_vals_copy.__getitem__, indices))
                    closest_diff = diff_vals_subset[
                        min(range(len(diff_vals_subset)), key=lambda i: abs(diff_vals_subset[i] - d))]
                    # If both the closest p and closest fold-difference are very close, then the two points are
                    # probably not far enough from one another to prevent overlap.
                    if abs(abs(closest_p) - abs(p)) < 0.1 and abs(abs(closest_diff) - abs(d)) < 0.1:
                        labels.append(None)
                        #labels.append(label.replace("_", " ").split(" ")[0])
                    else:
                        labels.append(label.replace("_", " ").split(" ")[0])
                else:
                    labels.append(label.replace("_", " ").split(" ")[0])
            # Adjust the labels based on whether the fold change was found to be significant, if the multiple
            # hypothesis method is set to Holm-Bonferroni or...(other methods yet to be implemented, e.g. Benjamini):
            # Sort the p-values in descending order, but keep track of the original index by also sorting indices:
            if "holm" in multiple_correction.casefold():
                sorted_p = sorted(p_vals[1], reverse=True)
                print(sorted_p)
                print(holm_bonferroni_significance)
                sorted_p_indices = np.argsort(p_vals[1])[::-1]
                for index, (hb_alpha, p) in enumerate(zip(holm_bonferroni_significance, sorted_p)):
                    if p > hb_alpha and labels[sorted_p_indices[index]] is not None:
                        labels[sorted_p_indices[index]] = labels[sorted_p_indices[index]] + "*"
            master_labels_list.append(labels)


        # Plot log2(fold-difference) vs. -log10(p-value) using a scatterplot:
        for p_vals, diff_vals, labels in zip(p_val_dict.items(), log_diff_dict.items(), master_labels_list):
            # Use p_vals[0] for keys, p_vals[1] for values (same for log differences).
            layout = go.Layout(plot_bgcolor='rgba(230, 235, 245, 1)')
            fig = go.Figure(data=go.Scatter(x=diff_vals[1], y=p_vals[1], mode='markers+text',
                marker=dict(size=25, color=p_vals[1], colorscale="Reds",
                            colorbar=dict(title="P-value (-log10 Scale)", title_font_size=16, titleside='right',
                                          thickness=35), line=dict(width=2)), text=labels, textposition="middle left",
                textfont=dict(size=16)

            ), layout=layout)

            #fig.add_hline(y=bonferroni_alpha, line_color='red')
            fig.update_layout(title="Volcano plot for {} compared to all other groups".format(p_vals[0].casefold()),
                              title_font_size=24, title_x=0.5, title_y=0.95, autosize=False, width=1000, height=1000,
                              )
            fig.update_xaxes(linecolor='black', title=r"$\log_{2}(Fold\ Change)$", title_font_size=24, ticks='outside',
                             tickfont_size=20, gridcolor='rgba(230, 235, 245, 1)')
            fig.update_yaxes(linecolor='black', title=r"$-\log_{10}(P_{value})$", title_font_size=24, ticks='outside',
                             tickfont_size=20, gridcolor='rgba(230, 235, 245, 1)')
            fig.update_coloraxes(colorbar=dict(len=0.5), colorbar_title=dict(side='right'))
            fig.show()


    def logistic_classification(self, predictors, cross_val_k=5, n_iterations=100):
        '''
        Function to build a logistic regression classifier to predict protected/not protected on the basis of some
        combination of serological features.
        :param predictors: Column IDs to the features to be used as predictors for the model.
        :param cross_val_k: Number of groups to split data into for cross-validation.
        :param n_iterations: Number of models to fit. The final AUROC measure will be an average of the result for
        each iteration. Defaults to 100.
        :return: Classification accuracy for each predictor.
        '''
        data = self.dataset[:]
        if not isinstance(predictors, list): predictors = [predictors]
        targets = data[self.group_col_id]
        # If targets is categorical, change labels to numeric. If there are more than two unique groups, raise an error.
        if not np.issubdtype(targets.dtype, np.number):
            targets = targets.groupby(data[self.group_col_id], sort=False).ngroup()
            if len(set(targets)) > 2:
                raise ValueError("Logistic regression is not suited for multi-class classification. Please try again.")
        # After assigning the targets/labels column, separate it from the dataset:
        data = data.drop(self.group_col_id, axis=1)
        targets = targets.values

        if predictors[0].casefold() == "all":
            # In the case that predictors is "All", delete non-numerical columns (which can't be used for regression):
            data = self.remove_nonnumeric(self.remove_NaNcols(data))
            predictors = list(data.columns)

        # Define the classifier and the cross-validation framework:
        classifier = LogisticRegression()
        # Try with K-fold and leave-one-out:
        cfv = KFold(n_splits=5, shuffle=True)
        # cfv = LeaveOneOut()

        # Instantiate list to store AUROC values in for each predictor; AUROC values will be an average of a number
        # of fitting-predicting iterations (given by the n_iterations argument):
        AUROC_averages_list = []
        # Instantiate a dataframe to keep track of all AUROCs values- potentially for plotting purposes later):
        iteration_AUROCS_df = pd.DataFrame()
        for predictor in predictors:
            # Use only the singular predictor to create a logistic regression model:
            data_subset = data[predictor].values
            # Z-score values so that all features operate on the same scale:
            data_subset = (data_subset - data_subset.mean()) / data_subset.std(ddof=1)
            # Instantiate list to store the AUROC for each iteration before taking the mean:
            iteration_AUROCS = []
            for iteration in range(n_iterations):
                # Lists to store the actual values (0 or 1) and the predictions of the logistic regression model,
                # respectively.
                actual_y = []
                preds = []
                for train, test in cfv.split(X=data_subset, y=targets):
                    # Reshape the data to match the sklearn convention (which expects a 2D array
                    # - here, all the data will be 1D because the features are being tested one at a time).
                    train_data = data_subset[train].reshape(-1, 1)
                    test_data = data_subset[test].reshape(-1, 1)
                    actual_y.extend(targets[test])
                    # Predict_proba() also returns an array; just take the [1] index to get the value.
                    preds.extend(classifier.fit(train_data, targets[train]).predict_proba(test_data)[:, 1])
                fpr, tpr, _ = roc_curve(actual_y, preds)
                # Append the AUROC for this iteration:
                iteration_AUROCS.append(auc(fpr, tpr))
            # Store the complete list of AUROCs for each iteration for each predictor:
            iteration_AUROCS_df[predictor] = iteration_AUROCS
            # Take the mean over all iterations for each predictor and store that in a separate list:
            AUROC_averages_list.append(np.mean(iteration_AUROCS))
        # Convert AUROC averages list to a dataframe:
        AUROCs_averages_df = pd.DataFrame(data=[AUROC_averages_list], columns=predictors)
        # Re-order dataframe columns in order of most predictive feature (for plotting purposes):
        iteration_AUROCS_df = iteration_AUROCS_df[
            list(AUROCs_averages_df.sort_values(by=0, ascending=False, axis=1).columns)]

        # Barplot of features and their AUROC scores:
        fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 6))
        ax.set_xlabel('Average AUROC', fontsize=16)
        # Add ax.set_xticklabels()
        ax.set_ylabel('Variable Name', fontsize=16)
        ax.tick_params(axis='y', labelsize=12)
        iteration_AUROCS_df.columns = [name.replace("_", " ") for name in list(iteration_AUROCS_df.columns)]
        ax.set_title('Most Predictive Features by AUROC Analysis', fontsize=16)
        sns.barplot(data=iteration_AUROCS_df, orient='h', palette=self.colormap, ax=ax, edgecolor=".1", capsize=0.25,
                    errcolor='black')
        plt.tight_layout()
        plt.show()


    def protection_regression(self, predictors, n_iterations=100, model_type='logistic', plot=True):
        '''
        Function to construct regression models for serological data.
        :param predictors: Column IDs for the features to be used as predictors in the models.
        :param n_iterations: Number of models to fit. The final regression coefficients will be an average of the
        results from each run (is this needed for regression/will there be variance in these coeffs)?
        :param model_type: A string to specify the type of the regression model. Options: "linear", "logistic",
        or "SVM".
        :param plot: Set to False to skip plotting of the coefficients.
        :return: A dataframe of regression coefficients.
        '''
        # Import all of the possible regressors:
        from sklearn.svm import SVR

        data = self.dataset[:]
        if not isinstance(predictors, list): predictors = [predictors]
        targets = data[self.group_col_id]
        # If a logistic regression is specified and targets is categorical, change labels to numeric. If there are more
        # than two unique groups, raise an error.
        if model_type == 'logistic' and not np.issubdtype(targets.dtype, np.number):
            targets = targets.groupby(data[self.group_col_id], sort=False).ngroup()
            if len(set(targets)) > 2:
                raise ValueError("Logistic regression is not suited for multi-class classification. Please try again.")
        # After assigning the targets/labels column, separate it from the dataset:
        data = data.drop(self.group_col_id, axis=1)
        targets = targets.values

        if predictors[0].casefold() == "all":
            # In the case that predictors is "All", delete non-numerical columns (which can't be used for regression):
            data = self.remove_nonnumeric(self.remove_NaNcols(data))
            predictors = list(data.columns)

        # Define the regressor (cross-validation, etc. is not necessary because we aren't trying to make predictions
        # using the regressor, just analyze the data we have):
        regressors = {
            'linear': LinearRegression(),
            'logistic': LogisticRegression(solver='liblinear', random_state=np.random.randint(0, 42)),
            # NOTE: liblinear for small datasets, like most serology datasets are.
            'svm': SVR()
        }
        mod = regressors[model_type.casefold()]

        # If the data to fit only contains one predictor, reshape to an nx1 array (sklearn expects two-dimensional
        # inputs).
        if data.values.ndim == 1:
            data = data.reshape(-1,1)
        # Z-score values so that all features operate on the same scale:
        data = data.apply(lambda x: (x - x.mean()) / x.std(ddof=1), axis=0)
        # Preprocess data to an ndarray and Z-score before fitting the model:
        data = data.values
        # Fit the model:
        fitted_mod = mod.fit(data, targets)
        # Note that the coefficient sign depends on which categorical distinction was assigned "0" and which was
        # assigned "1" above.
        abs_coefs = np.abs(fitted_mod.coef_)
        # Create a dataframe using the coefficient values:
        coefs_df = pd.DataFrame(abs_coefs if len(abs_coefs)==1 else [abs_coefs], columns=predictors)
        # If logistic regression, convert the coefficients to odds ratios:
        if model_type == 'logistic':
            coefs_df = coefs_df.apply(np.exp, axis=1)
        # Sort the dataframe:
        coefs_df = coefs_df.sort_values(by=0, axis=1, ascending=False)

        # Plot if the plot flag is True:
        if plot:
            fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 6))
            ax.set_xlabel('Odds Ratio' if model_type == 'logistic' else 'Regression Coefficient Magnitude', fontsize=16)
            ax.set_ylabel('Variable Name', fontsize=16)
            ax.tick_params(axis='y', labelsize=12)
            coefs_df.columns = [name.replace("_", " ") for name in list(coefs_df.columns)]
            sns.barplot(data=coefs_df, orient='h', palette=self.colormap, ax=ax, edgecolor=".1")
            plt.tight_layout()
            plt.show()



    def feature_influence(self, predictors, cross_val_k=5, n_iterations=100, title=None):
        '''
        Function to determine the relative importance of features by building a random forest classifier to predict
        which group given sample(s) are in on the basis of some combination of serological features. This will be
        accomplished in two ways: determining which sole predictors are the best at classifying, and then starting
        from the entire feature set, selectively removing one feature at a time and measuring the impact on the
        accuracy (while here, will also return the impurity-based feature importances).
        :param predictors: Column IDs to the features to be used as predictors for the model.
        :param cross_val_k: Number of groups to split data into for cross-validation. Note: this should max out at
        the number of samples in the class w/ the lowest number of samples.
        :param n_iterations: Number of models to fit. The final AUROC measure will be an average of the result for
        each iteration. Defaults to 100.
        :param title: Optional argument to supply a title for the plots that are generated by this function.
        :return: None (but will plot a barplot).
        '''
        import graphviz
        from sklearn.tree import export_graphviz

        data = self.dataset[:]
        if not isinstance(predictors, list): predictors = [predictors]
        targets = data[self.group_col_id]
        # If targets is categorical, change labels to numeric.
        if not np.issubdtype(targets.dtype, np.number):
            targets = targets.groupby(data[self.group_col_id], sort=False).ngroup()
        # After assigning the targets/labels column, separate it from the dataset:
        data = data.drop(self.group_col_id, axis=1)
        targets = targets.values

        if predictors[0].casefold() == "all":
            # In the case that predictors is "All", delete non-numerical columns (which can't be used for regression):
            data = self.remove_nonnumeric(self.remove_NaNcols(data))
            predictors = list(data.columns)
        data_np = data.values

        # Initialize the random forest with default parameters:
        classifier = RandomForestClassifier()
        # Perform a grid search and train on the entire dataset to find the best hyperparameters, to be used for any
        # other trees that are created:
        param_grid = {'n_estimators': range(1, 30), 'max_depth': range(2, 6)}
        grid_search = GridSearchCV(classifier, param_grid, cv=3)
        grid_search.fit(data_np, targets)
        # Best model and best parameters:
        best_forest = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(best_params)

        # Initialize another random forest with the parameters found by the grid search, as well as the
        # cross-validation scheme:
        classifier = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params[
            'max_depth'])
        cfv = StratifiedKFold(n_splits=3, shuffle=True)

        # For each iteration in n_iterations, train a random forest classifier and store the accuracy. Also list the
        # feature importances for the iteration, and to get overall feature importance, tally how many times each
        # feature comes up in the top 5 (10?) features.
        # Instantiate lists to store the accuracy for each iteration before taking the mean:
        iteration_acc = []
        # Also instantiate a dataframe to store feature importances in:
        feature_importance_df = pd.DataFrame(columns=data.columns)
        for iteration in range(n_iterations):
            # Lists to store the actual values and the predictions of the random forest, respectively.
            actual_y = []
            preds = []
            for train, test in cfv.split(X=data_np, y=targets):
                train_data = data_np[train]
                test_data = data_np[test]
                actual_y.extend(targets[test])
                # predict() returns a list, take the second column to get the predicted outputs.
                preds.extend(classifier.fit(train_data, targets[train]).predict(test_data))
                # Return feature importance for this test set:
                iteration_feature_importances = classifier.feature_importances_
                # Add this to the dataframe:
                feature_importance_df.loc[len(feature_importance_df)] = iteration_feature_importances
            iteration_acc.append(accuracy_score(actual_y, preds))
        # Take the average importance value for each feature, then rank in order of importance:
        feature_importance_df = pd.DataFrame(feature_importance_df.apply(np.mean, axis=0).values.reshape(1, -1),
            columns=data.columns)
        feature_importance_df = feature_importance_df[
            list(feature_importance_df.sort_values(by=0, ascending=False, axis=1).columns)]
        # Plot the average feature importances as a barplot:
        fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 6))
        ax.set_xlabel('Variable Name', fontsize=16)
        ax.set_ylabel('Average Gini Importance', fontsize=16)
        ax.tick_params(axis='x', labelsize=12, labelrotation=45)
        feature_importance_df.columns = [name.replace("_", " ") for name in list(feature_importance_df.columns)]
        ax.set_title(title.replace("_", " ") if title is not None else 'Most Predictive Features', fontsize=16)
        sns.barplot(data=feature_importance_df, orient='v', palette='Reds_r', ax=ax, edgecolor=".1", capsize=0.25,
                    errcolor='black', linewidth=1.5)
        plt.tight_layout()
        plt.show()
        print("Average iteration accuracy: {:.3f}".format(np.mean(iteration_acc)))
        # Visualize the random forest:
        tree_Graphviz = export_graphviz(classifier.base_estimator_.fit(data_np, targets), filled=True,
                                        feature_names=data.columns)
        graph = graphviz.Source(tree_Graphviz)
        graph.view()


    def variable_enrichment(self, keywords=None, enrichment_id="Variable_enrichment", plot=False, save=True):
        '''
        Function to compute the variable enrichment of a group of variables from a systems serology dataset; this is
        given by the mean of the Z-scores for all specified variables (given as keywords).
        :param keywords: Optional argument allowing for a list of variable column IDs to be provided, to be used in
        variable enrichment calculations. If None, will default to using functional features (e.g. ADCP,
        NK degran MIP1B, etc.).
        :param enrichment_id: String specifying the column title that the enrichment variable will be stored in.
        :param plot: If True, display the differences in the variable enrichment between groups using a violin plot.
        :param save: If True, save the result to a .csv file.
        :return: Numerical variable enrichment score.
        '''
        if isinstance(enrichment_id, list): enrichment_id = enrichment_id[0]
        default_flag = False  # set to True if keywords is None.
        if keywords is None:
            # Compute functional enrichment if keywords are not given.
            keywords = ["ADCP", "ADNP", "ADCD", "NK"]
            default_flag = True
        if not isinstance(keywords, list): keywords = [keywords]
        features_of_interest = self.dataset[
            [col for col in self.dataset.columns if any(keyword.casefold() in col.casefold() for keyword in keywords)]]
        enrichment = np.mean(
            [(features_of_interest[col] - features_of_interest[col].mean()) / features_of_interest[col].std(ddof=1) for
             col in features_of_interest.columns], axis=0)
        if default_flag:
            self.dataset["Functional_Enrichment_(averaged_z-score)"] = enrichment
        else:
            self.dataset[enrichment_id] = enrichment
        if plot:
            self.comparison_plots(features_to_plot="Functional_Enrichment_(averaged_z-score)" if default_flag else
            enrichment_id, type="Violins")
        if save:
            SystemsSerologyPreprocessing(data=self.dataset).save_dataset(
                output_basename="SystemsSerologyVariableEnrichment", dataset=self.dataset)
        return enrichment



if __name__ == "__main__":
    if args.task == "Preprocessing":
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, second_dataset=args.datapath2,
        ignore_index=True, output_basename="SystemsSerology")
        preprocessing.columns_remove_spaces(save=True)
        preprocessing.drop_na(save=True)
        if args.var is not None or args.string is not None:
            preprocessing.control_correction(control_column_ids=args.var, control_row_id=args.string,
                                             save=True)
        # More options:
        #preprocessing.z_score(save=True)
        #preprocessing.log10_transform(save=True, luminex_only=args.set_flag)
    # data = pd.read_csv(os.path.join(str(Path().absolute()), "Processed_Data\SystemsSerology.csv"),
    # index_col=0)
    if args.task == "Copy":
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, ignore_index=True,
                                                     output_basename="SystemsSerologyCopy")
        preprocessing.save_dataset(preprocessing.output_basename, preprocessing.dataset)
    if args.task == "Z_Score":
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, ignore_index=True,
                                                     output_basename="SystemsSerologyZscored")
        preprocessing.z_score(save=True, make_positive=args.set_flag)
    if args.task == "Subset_Data":
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, ignore_index=True,
                                                     output_basename="SystemsSerologySubset")
        preprocessing.subset_data(groups_col=args.group, to_keep=args.key, save=True)
    if args.task == "Shorten_Columns":
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, ignore_index=True,
                                                     output_basename="SystemsSerology")
        preprocessing.shorten_columns(to_remove=args.key, save=True)
    if args.task == "Combine_Columns":
        combined_colname = args.string if args.string is not None else "Combined"
        SystemsSerology(data=args.datapath).combine_columns(dataset=args.datapath, cols=args.key,
                                                   name_combined_col=combined_colname)
    if args.task == "Remove_Columns":
        if args.key is None:
            raise ValueError("No keys were provided, so manual feature deletion could not be performed.")
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, ignore_index=True,
                                                     output_basename="SystemsSerologyManualFeatureSelection")
        manual_feature_selected = preprocessing.delete_by_keywords(keywords=args.key, save=True)
        print("Remaining Features: ", list(manual_feature_selected.columns))
    if args.task == "Keep_Columns":
        if args.key is None:
            raise ValueError("No keys were provided, so manual feature extraction could not be performed.")
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, ignore_index=True,
                                                     output_basename="SystemsSerologyManualFeatureSelection")
        manual_feature_selected = preprocessing.extract_by_keywords(keywords=args.key, save=True)
        print("Remaining Features: ", manual_feature_selected.columns)
    if args.task == "Move_Columns":
        if args.key is None:
            raise ValueError("No keys were provided, so no columns can be moved.")
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, second_dataset=args.datapath2,
                            ignore_index=True, output_basename="SystemsSerology")
        # os.path.basename(os.path.normpath(args.datapath)) to save to a file with the same base name as args.datapath.
        cols_moved = preprocessing.move_columns(cols_to_move=args.key, save=True)
    if args.task == "Move_Rows":
        if args.key is None:
            raise ValueError("No row IDs were provided, so no rows can be moved.")
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, second_dataset=args.datapath2,
                            ignore_index=True, output_basename="SystemsSerology")
        rows_moved = preprocessing.move_rows(rows_to_move=args.key, save=True)
    if args.task == "Remove_Samples":
        if args.key is None:
            raise ValueError("No keys were provided to specify which samples should be removed.")
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, ignore_index=True,
                                                     output_basename="SystemsSerologySamplesRemoved")
        samples_removed = preprocessing.remove_samples(keys=args.key, sample_col=args.labels, save=True)
    if args.task == "Concatenate":
        preprocessing = SystemsSerologyPreprocessing(data=args.datapath, second_dataset=args.datapath2,
                                                     ignore_index=True, output_basename="SystemsSerologyConcatenated")
        concatenated = preprocessing.concatenate(save=True)


    analysis = SystemsSerology(data=args.datapath, second_dataset=args.datapath2, group_col_id=args.group,
                               labels_col_id=args.labels)
    if args.task == "Feature_Selection":
        mask, feat_sel = analysis.lasso_feature_selection(alpha=args.alpha,
                                                          output_basename="SystemsSerologyFeatureSelected")
        print("Remaining features: ", feat_sel.columns)
    if args.task == "PCA":
        loadings, pc_df = analysis.n_dimensional_PCA_view_2D(n_components=args.num_lvs, title=args.title,
                                                             fontsize=args.fontsize, colorbar_var=args.var,
                                                             n_loadings=args.num_loadings,
                                                             find_clusters=args.cluster_flag,
                                                             num_clusters=args.num_clusters,
                                                             draw_ellipses=args.draw_ellipses,
                                                             draw_colorbar=args.draw_colorbar)
        print(loadings)
        print(pc_df)
    if args.task == "PLS":
        loadings, pls_data = analysis.PLS(n_components=args.num_lvs, var_for_regression=args.var, title=args.title,
                                          fontsize=args.fontsize, n_loadings=args.num_loadings,
                                          find_clusters=args.cluster_flag, num_clusters=args.num_clusters,
                                          draw_ellipses=args.draw_ellipses, key=args.key)
        print(loadings)
    if args.task == "Most_Predictive":
        analysis.most_predictive_features(features_of_interest=args.var, title=args.title,
                                          multiple_correction=args.string)
    if args.task == "Violins" or args.task == "Boxplots" or args.task == "Swarm" or args.task == "Strip":
        analysis.comparison_plots(features_to_plot=args.var, type=args.task, nested_grouping=args.nested_grouping,
                                  save_keyword=args.string, annot=args.set_flag)
    if args.task == "Bars":
        analysis.comparison_plots(features_to_plot=args.var, type=args.task, nested_grouping=args.nested_grouping)
    if args.task == "Flowers":
        analysis.flower_plots(features_to_exclude=args.var, group_keys=args.key, fig_layout='subplots')
    if args.task == "Dotplots":
        analysis.dotplots(variables_of_interest=args.var, keywords=args.key)
    if args.task == "Heatmap":
        analysis.heatmap_master(group_keys=args.key, type=args.heatmap_type, title=args.title,
                                      other_secondary_label=args.string)
    if args.task == "Classification":
        analysis.logistic_classification(predictors=args.var, n_iterations=args.num_iterations)
    if args.task == "Regression":
        analysis.protection_regression(predictors=args.var, model_type=args.model_type, plot=True)
    if args.task == "Feature_Influence":
        analysis.feature_influence(predictors=args.var, n_iterations=args.num_iterations, title=args.title)
    if args.task == "Variable_Enrichment":
        analysis.variable_enrichment(keywords=args.key, enrichment_id=args.var, plot=True, save=True)

    # Slightly different formulation for trajectory analysis:
    if args.task == "Trajectory_Analysis":
        TA = SystemsSerologyTrajectoryAnalysis(data=args.datapath, group_col_id=args.group, labels_col_id=args.labels)

        TA.trajectory_lineplot(time_col_id=args.key, variables_to_plot=args.var, x_label=args.xlabel,
                               y_label=args.ylabel)