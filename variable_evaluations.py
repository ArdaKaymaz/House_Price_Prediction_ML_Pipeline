import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



def plot_importance(model, features, model_name, num=20, save=False):
    """

    Plot feature importances for a given model.

        Parameters:
        - model (object): The trained model with a `feature_importances_` attribute.
        - features (DataFrame): The DataFrame containing the features used in the model.
        - model_name (str): A string specifying the name of the model for plot title.
        - num (int, optional): Number of top features to display (default is 20).
        - save (bool, optional): Whether to save the plot as a PNG file (default is False).

        Returns:
        - DataFrame: A DataFrame containing feature importances.

        Example:
        >>> model = RandomForestRegressor()
        >>> model.fit(X_train, y_train)
        >>> plot_importance(model, X_train, "Random Forest")

    """
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title(f"{model_name} Features")
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig(f"{model_name}_importances.png")

    return feature_imp


def plot_importance_for_func(models, features, num=20, save=False):
    """
    Plot feature importances for a list of models.

    Parameters:
    - models (list): A list of trained models, each with a `feature_importances_` attribute.
    - features (DataFrame): The DataFrame containing the features used in the models.
    - num (int, optional): Number of top features to display (default is 20).
    - save (bool, optional): Whether to save individual plots for each model (default is False).

    Returns:
    None

    Example:
    >>> models = [RandomForestRegressor(), DecisionTreeRegressor()]
    >>> for model in models:
    ...     model.fit(X_train, y_train)
    ...
    >>> plot_importance_for_func(models, X_train)
    """
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)

    for model in models:
        if hasattr(model, 'feature_importances_'):
            feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
            sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
            plt.title(f"{type(model).__name__} Features")
            plt.tight_layout()
            plt.show()

            if save:
                plt.savefig(f"{type(model).__name__} importances.png")
        else:
            print(f"Model {type(model).__name__} does not have feature_importances_. Skipping...")


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    """
    Plot validation curve for a model's performance over a range of hyperparameter values.

    Parameters:
    - model (object): The machine learning model for which the validation curve is plotted.
    - X (array-like or pd.DataFrame): Input features.
    - y (array-like): Target variable.
    - param_name (str): Name of the hyperparameter to vary.
    - param_range (array-like): Range of values for the hyperparameter.
    - scoring (str, optional): Scoring metric for evaluation (default is 'roc_auc').
    - cv (int, optional): Number of cross-validation folds (default is 10).

    Returns:
    None

    Example:
    >>> model = RandomForestClassifier()
    >>> param_name = 'n_estimators'
    >>> param_range = [10, 20, 30, 40, 50]
    >>> val_curve_params(model, X_train, y_train, param_name, param_range)
    """
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)








