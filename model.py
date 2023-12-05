import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from house_prediction import variable_evaluations as ve


def evaluate_models(X, y, plot_imp=False, save=False, num=20, random_state=7):
    """
    Evaluate and compare the performance of various regression models.

    Parameters:
    - X (array-like or pd.DataFrame): Input features.
    - y (array-like): Target variable.
    - plot_imp (bool, optional): Whether to plot feature importances for each model (default is False).
    - save (bool, optional): Whether to save feature importance plots (default is False).
    - num (int, optional): Number of top features to display in importance plots (default is 20).
    - random_state (int, optional): Seed for random number generation (default is 7).

    Returns:
    - dict: A dictionary containing the performance metrics (RMSE) for each model.
    - dict: A dictionary containing the names of the fitted models.

    Example:
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> performance, models_names = evaluate_models(X_train, y_train, plot_imp=True, save=True)
    """

    global fitted_models

    models = {
        'LR': LinearRegression(),
        'KNN': KNeighborsRegressor(),
        'CART': DecisionTreeRegressor(random_state=random_state),
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
        'XGBoost': XGBRegressor(random_state=random_state),
        'LightGBM': LGBMRegressor(random_state=random_state)
    }

    models_names = {}
    performance = {}

    for model_name, model in models.items():
        performance[model_name] = \
            {
              "RMSE":  np.mean(np.sqrt(-cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error", )))
            }
        models_names[model_name] = {model}

        fitted_models = [model.fit(X, y) for model_name, model in models.items()]

    if plot_imp:
        ve.plot_importance_for_func(fitted_models, X, num=num, save=save)

    return performance, models_names
