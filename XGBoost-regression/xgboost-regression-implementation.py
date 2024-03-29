import pandas as pd
import numpy as np
import xgboost
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 30)

boston = load_boston()

# The boston variable itself is a dictionary, hence we can check its keys
print(boston.keys())

# To check the shape of the dataset
print("Dataset shape: ".format(boston.data.shape))

boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['PRICE'] = boston.target
print(boston_df.describe())

# Dividing the dataframe into features, labels
X, y = boston_df.iloc[:, :-1], boston_df.iloc[:, -1]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply xgbregressor on the dataset
xgb_reg = xgboost.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                               max_depth=5, reg_alpha=10, n_estimators=10)

xgb_reg.fit(X_train, y_train)
predictions = xgb_reg.predict(X_test)

# Calculate root mean squared error
print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test, predictions))))

# Converting the features, labels to Dmatrix structure
boston_dmatrix = xgboost.DMatrix(data=X, label=y)

# Using K-fold cross validation
params = {"objective": "reg:squarederror", "colsample_bytree": 0.3, "learning_rate": 0.1, "max_depth": 5,
          "reg_alpha": 10}
xgb_reg_cv = xgboost.cv(dtrain=boston_dmatrix, params=params, nfold=3, num_boost_round=400, early_stopping_rounds=10,
                        metrics="rmse", as_pandas=True, seed=42)

# Calculate root mean squared error
print((xgb_reg_cv["test-rmse-mean"]).tail(1))

# Visualize the feature importance
xg_reg = xgboost.train(params=params, dtrain=boston_dmatrix, num_boost_round=10)
xgboost.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show(block=True)