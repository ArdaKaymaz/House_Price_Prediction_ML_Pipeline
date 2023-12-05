import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def dataframe_reading(file_path):
    dataframe = pd.read_csv(file_path)
    return dataframe


def check_data(dataframe):
    """
    Display key information about a DataFrame, including shape, head, tail, missing values, and descriptive statistics.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to be checked.

    Returns:
    None

    Example:
    >>> check_data(df)
    """
    print(20 * "-" + "Information".center(20) + 20 * "-")
    print(dataframe.info())
    print(20 * "-" + "Data Shape".center(20) + 20 * "-")
    print(dataframe.shape)
    print("\n" + 20 * "-" + "The First 5 Data".center(20) + 20 * "-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print((dataframe.isnull().sum()).sort_values(ascending=False))
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)


def string_to_numerical(dataframe, overwrite=False):
    """
    Convert mistyped columns in a DataFrame to float type.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame containing columns to be converted.
    - overwrite (bool, optional): If True, overwrite the original DataFrame with converted values.

    Returns:
    None

    Prints:
    - The number of mistyped columns found and the corresponding column names.
    - If overwrite is True, also prints the number of columns converted and the names of mistyped columns.
    """
    numeric_cols = []
    converted_cols = {}

    for col in dataframe.columns:
        try:
            if dataframe[col].dtype != "O":
                numeric_cols.append(col)
            elif col not in numeric_cols:
                converted_cols[col] = pd.to_numeric(dataframe[col].replace(" ", ""))
        except ValueError:
            pass

    if overwrite:
        for col_name, converted_values in converted_cols.items():
            dataframe[col_name] = converted_values
        print(f"0 mistyped column(s) needs to be converted to float type.\n"
              f"{len(converted_cols.keys())} mistyped column(s) converted to float type.\n"
              f"Mistyped columns: {list(converted_cols.keys())}")
    else:
        print(f"{len(converted_cols)} mistyped column(s) needs to be converted to float type.")
        if len(converted_cols) > 0:
            print(f"Mistyped columns: {list(converted_cols.keys())}")
        else:
            print(f"No mistyped columns found.")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Identify and return categorical, numerical, and cardinal variables along with their quantities.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame from which variable names are extracted.
    cat_th : int or float, optional
        Threshold value for identifying variables that are numeric but categorical (default is 10).
    car_th : int or float, optional
        Threshold value for identifying variables that are categorical but cardinal (default is 20).

    Returns
    -------
    cat_cols : list
        List of categorical variables.
    num_cols : list
        List of numerical variables.
    cat_but_car : list
        List of cardinal variables with a categorical aspect.

    Notes
    -----
    cat_cols + num_cols + cat_but_car = total variables
    num_but_cat are included in cat_cols.

    Examples
    --------
    >>> cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=8, car_th=15)
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


def num_summary(dataframe, numerical_col, plot=False):
    """
    Generate summary statistics for a numerical column in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the data.
    - numerical_col (str): The name of the numerical column for which to generate summary statistics.
    - plot (bool, optional): Whether to plot a histogram of the numerical column (default is False).

    Returns:
    None

    Example:
    >>> num_summary(df, 'Age', plot=True)
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    result = dataframe[numerical_col].describe(quantiles)
    print(result, end="\n\n")

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def cat_summary(dataframe, col_name, plot=False):
    """
    Generate summary statistics for a categorical column in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the data.
    - col_name (str): The name of the categorical column for which to generate summary statistics.
    - plot (bool, optional): Whether to plot a countplot of the categorical column (default is False).

    Returns:
    None

    Example:
    >>> cat_summary(df, 'Category', plot=True)
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=45)
        plt.show(block=True)


def target_summary(dataframe, target, column):
    """
    Generate a summary of a column grouped by the target variable in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the data.
    - target (str): The name of the target variable.
    - column (str): The name of the column for which to generate the summary.

    Returns:
    None

    Example:
    >>> target_summary(df, 'Target', 'Category')
    """
    if dataframe[column].dtype == 'O':
        result = dataframe.groupby([target, column])[column].count().reset_index(name='count')
        print(result.transpose(), end="\n\n")

    else:
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        result = dataframe.groupby(target).agg({column: 'mean'}).reset_index()
        print(result.transpose(), end="\n\n")
    print("\n###################################\n\n")


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    """
    Identify highly correlated columns in a DataFrame based on a correlation threshold.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the data.
    - plot (bool, optional): Whether to plot a heatmap of the correlation matrix (default is False).
    - corr_th (float, optional): The correlation threshold to identify highly correlated columns (default is 0.70).

    Returns:
    - list: A list of column names to be dropped due to high correlation.

    Example:
    >>> drop_columns = high_correlated_cols(df, plot=True, corr_th=0.75)
    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list
