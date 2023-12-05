from house_prediction_M import data_reading_and_exploring as de
from house_prediction_M import feature_engineering as fe
from house_prediction_M import model as md
from house_prediction_M import variable_evaluations as ve
import numpy as np
import pandas as pd


### Data Reading and Exploring ###

df = de.dataframe_reading("house_prediction/datasets/train.csv")

de.check_data(df)

de.string_to_numerical(df)

cat_cols, num_cols, cat_but_car = de.grab_col_names(df, car_th=25)

for col in num_cols:
    de.num_summary(df, col)

for col in cat_cols:
    de.cat_summary(df, col)

cat_num_cols = cat_cols + num_cols

# for col in cat_num_cols:
#     de.target_summary(df[cat_num_cols], "SalePrice", col)


### Feature Engineering ###

for col in num_cols:
    print(col, fe.check_outlier(df, col, q1=0.25, q3=0.75))

for col in num_cols:
    fe.replace_with_thresholds(df, col, q1=0.25, q3=0.75)

fe.missing_values_table(df)

df = fe.quick_missing_imp(df)

for col in num_cols:
    df = fe.scaling_func(df, col, "robust")

df = fe.one_hot_encoder(df, cat_cols)


### Modelling ###

y = df["SalePrice"]

X = df.drop(["Id", "SalePrice"], axis=1)


# Dead model #

performance, models = md.evaluate_models(X, y, plot_imp=True, num=20, save=False, random_state=7)

# To evaluate the model after the feature extraction process, proceed to further. #

""" ***Robust***                    ***Min-Max***                   ***Standart***
RMSE: 0.3737 (LR)     RMSE: 10812306643.5973 (LR)           RMSE: 39828988381.9135 (LR) 
RMSE: 0.6032 (KNN)            RMSE: 0.0548 (KNN)                    RMSE: 0.4742 (KNN) 
RMSE: 0.496 (CART)            RMSE: 0.0589 (CART)                   RMSE: 0.5529 (CART)
RMSE: 0.3616 (RF)             RMSE: 0.042 (RF)                      RMSE: 0.3837 (RF) 
RMSE: 0.3156 (GBM)            RMSE: 0.0367 (GBM)                    RMSE: 0.3355 (GBM) 
RMSE: 0.3389 (XGBoost)        RMSE: 0.0396 (XGBoost)                RMSE: 0.3584 (XGBoost)
RMSE: 0.3447 (LightGBM)       RMSE: 0.0401 (LightGBM)               RMSE: 0.3643 (LightGBM)
RMSE: 0.3447 (0.01 0.99 Outlier (LightGBM))
RMSE: 0.3349 (0.05 0.95 Outlier (LightGBM))
RMSE: 0.2491 (0.25 0.75 Outlier (LightGBM))
RMSE: 20875.612053444867 (0.25 0.75 Outlier, without scaling (LightGBM))
"""

### After Feature Extraction ###

# Read The Dataset Again #

df = de.dataframe_reading("house_prediction/datasets/train.csv")


# Grabbing Columns Before Feature Extraction #

cat_cols, num_cols, cat_but_car = de.grab_col_names(df, car_th=25)


# Feature Engineering #

for col in num_cols:
    print(col, fe.check_outlier(df, col, q1=0.25, q3=0.75))

for col in num_cols:
    fe.replace_with_thresholds(df, col, q1=0.25, q3=0.75)

fe.missing_values_table(df)

df = fe.quick_missing_imp(df)

# for col in num_cols:
#     df = fe.scaling_func(df, col, "robust")


# Feature Extraction #

df["NEW_1st*GrLiv"] = df["1stFlrSF"] * df["GrLivArea"]

df["NEW_Garage*GrLiv"] = (df["GarageArea"] * df["GrLivArea"])

# Total Floor
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]

# Total Finished Basement Area
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2

# Porch Area
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df[
    "3SsnPorch"] + df.WoodDeckSF

# Total House Area
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF

df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF

# Lot Ratio
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea

df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea

df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea

# MasVnrArea
df["NEW_MasVnrRatio"] = df.MasVnrArea / df.NEW_TotalHouseArea

# Dif Area
df["NEW_DifArea"] = (df.LotArea - df[
    "1stFlrSF"] - df.GarageArea - df.NEW_PorchArea - df.WoodDeckSF)

df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"]

df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt

df["NEW_HouseAge"] = df.YrSold - df.YearBuilt

df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd

df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt

df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd)

df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt


# Grabbing Columns After Feature Extraction #

cat_cols, num_cols, cat_but_car = de.grab_col_names(df, car_th=25)


# Encoding #

df = fe.one_hot_encoder(df, cat_cols)


# Modelling After Feature Extraction #

y = df["SalePrice"]

X = df.drop(["Id", "SalePrice"], axis=1)

performance, models = md.evaluate_models(X, y, plot_imp=False, num=30, save=True, random_state=7)


# Feature Selection #

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

lgbm_model = LGBMRegressor(random_state=7).fit(X, y)

important_features = ve.plot_importance(model=lgbm_model, features=X, model_name="LightGBM", num=40, save=True)

pd.set_option("display.max_rows", None)

important_features.sort_values(by="Value", ascending=False)

selected_features = important_features.loc[important_features['Value'] >= 5, 'Feature'] # 5 is the best threshold value

X = df[selected_features]

performance, models = md.evaluate_models(X, y, plot_imp=False, num=30, save=True, random_state=7)

"""
All features                   
{'LR': {'RMSE': 6640650.47276791},
 'KNN': {'RMSE': 0.3640315934296902},
 'CART': {'RMSE': 0.40066007822794425},
 'Random Forest': {'RMSE': 0.2700645129677207},
 'Gradient Boosting': {'RMSE': 0.2539589558934124},
 'XGBoost': {'RMSE': 0.271796322295383},
 'LightGBM': {'RMSE': 0.251341992798846}}
"""

"""
# Threshold = >=5(gini, feature importance), without scaling   # Threshold = >=5(gini, feature importance), with scaling
# {'LR': {'RMSE': 20774.78358671364},                          # {'LR': {'RMSE': 0.26190349363144183},
#  'KNN': {'RMSE': 39254.66746608178},                         #  'KNN': {'RMSE': 0.35899516307579254},
#  'CART': {'RMSE': 31213.08333622585},                        #  'CART': {'RMSE': 0.4076390202499912},
#  'Random Forest': {'RMSE': 21372.61849861148},               #  'Random Forest': {'RMSE': 0.2673577841737005},
#  'Gradient Boosting': {'RMSE': 20212.891314111697},          #  'Gradient Boosting': {'RMSE': 0.24739721537144796},
#  'XGBoost': {'RMSE': 21609.351866943653},                    #  'XGBoost': {'RMSE': 0.26138508148665796},
#  'LightGBM': {'RMSE': 20435.29736331899}}                    #  'LightGBM': {'RMSE': 0.2501721285786639}}
 """

# y.mean()
# y.std()

# LightGBM is faster also has more accurate results than most of the others.


### Hyperparameter Optimization ###

lgbm_model = LGBMRegressor(random_state=7)

lgbm_model.get_params()

lgbm_random_params = \
    {
        "learning_rate": np.random.uniform(0.01, 0.1, 10),
        "n_estimators": np.random.randint(1000, 10000, 10),
        "colsample_bytree": np.random.uniform(0.5, 1, 10),
        "min_child_samples": np.random.randint(10, 50, 20),
        "num_leaves": np.random.randint(10, 75, 15)
    }

lgbm_random = RandomizedSearchCV(estimator=lgbm_model,
                                 param_distributions=lgbm_random_params,
                                 cv=5,
                                 verbose=True,
                                 n_jobs=-1).fit(X, y)

print(lgbm_random.best_params_)

lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.02, 0.03881494490727592, 0.04],
               "n_estimators": [2016],
               "colsample_bytree": [0.9084045004168388],
               "min_child_samples": [12],
               "num_leaves": [14]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_random.best_params_, random_state=7).fit(X, y)

cv_results = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=5, scoring="neg_mean_squared_error")))

print(f"RMSE = {cv_results}")

# RMSE = 19808.25389624279


import joblib

joblib.dump(lgbm_final, "other_files/LightGBM_Final.pkl")


### Test Dataframe Prediction ###

# Read The Dataset Again #

test_df = de.dataframe_reading("house_prediction/datasets/test.csv")


# Test Dataset Grabbing Columns Before Feature Extraction #

train_cat_cols, train_num_cols, train_cat_but_car = de.grab_col_names(test_df, car_th=25)


# Feature Engineering #

fe.missing_values_table(test_df)

test_df = fe.quick_missing_imp_test(test_df)

# for col in num_cols:
#     test_df = fe.scaling_func(test_df, col, "robust")


# Feature Extraction #

test_df["NEW_1st*GrLiv"] = test_df["1stFlrSF"] * test_df["GrLivArea"]

test_df["NEW_Garage*GrLiv"] = (test_df["GarageArea"] * test_df["GrLivArea"])

# Total Floor
test_df["NEW_TotalFlrSF"] = test_df["1stFlrSF"] + test_df["2ndFlrSF"]

# Total Finished Basement Area
test_df["NEW_TotalBsmtFin"] = test_df.BsmtFinSF1 + test_df.BsmtFinSF2

# Porch Area
test_df["NEW_PorchArea"] = test_df.OpenPorchSF + test_df.EnclosedPorch + test_df.ScreenPorch + test_df[
    "3SsnPorch"] + test_df.WoodDeckSF

# Total House Area
test_df["NEW_TotalHouseArea"] = test_df.NEW_TotalFlrSF + test_df.TotalBsmtSF

test_df["NEW_TotalSqFeet"] = test_df.GrLivArea + test_df.TotalBsmtSF

# Lot Ratio
test_df["NEW_LotRatio"] = test_df.GrLivArea / test_df.LotArea

test_df["NEW_RatioArea"] = test_df.NEW_TotalHouseArea / test_df.LotArea

test_df["NEW_GarageLotRatio"] = test_df.GarageArea / test_df.LotArea

# MasVnrArea
test_df["NEW_MasVnrRatio"] = test_df.MasVnrArea / test_df.NEW_TotalHouseArea

# Dif Area
test_df["NEW_DifArea"] = (test_df.LotArea - test_df[
    "1stFlrSF"] - test_df.GarageArea - test_df.NEW_PorchArea - test_df.WoodDeckSF)

test_df["NEW_OverallGrade"] = test_df["OverallQual"] * test_df["OverallCond"]

test_df["NEW_Restoration"] = test_df.YearRemodAdd - test_df.YearBuilt

test_df["NEW_HouseAge"] = test_df.YrSold - test_df.YearBuilt

test_df["NEW_RestorationAge"] = test_df.YrSold - test_df.YearRemodAdd

test_df["NEW_GarageAge"] = test_df.GarageYrBlt - test_df.YearBuilt

test_df["NEW_GarageRestorationAge"] = np.abs(test_df.GarageYrBlt - test_df.YearRemodAdd)

test_df["NEW_GarageSold"] = test_df.YrSold - test_df.GarageYrBlt


# Test Dataset Grabbing Columns After Feature Extraction #

train_cat_cols, train_num_cols, train_cat_but_car = de.grab_col_names(test_df, car_th=25)


# Encoding #

test_df = fe.one_hot_encoder(test_df, cat_cols)


# As the result of encoding process, there are a few mismatches between train_df and test_df

missing_columns = set(selected_features) - set(test_df.columns)
print(f"Missing columns: {missing_columns}")

garage_columns = test_df.columns[test_df.columns.str.contains("Garage")]
bsmt_columns = test_df.columns[test_df.columns.str.contains("Bsmt")]

test_df.rename(columns={"BsmtFullBath_1.0": "BsmtFullBath_1"}, inplace=True)
test_df.rename(columns={"GarageCars_2.0": "GarageCars_2"}, inplace=True)
test_df.rename(columns={"GarageCars_3.0": "GarageCars_3"}, inplace=True)


# Test Dataframe Prediction #

cat_cols, num_cols, cat_but_car = de.grab_col_names(test_df[selected_features], car_th=25)

test_df["SalePrice"] = lgbm_final.predict(test_df[selected_features])

pd.reset_option("display.max_rows")

test_df[["Id", "SalePrice"]]

sample = de.dataframe_reading("house_prediction/datasets/sample_submission.csv")
