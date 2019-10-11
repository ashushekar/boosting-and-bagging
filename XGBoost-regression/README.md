# XGBoost on a Regression Problem

## Dataset Used
The dataset is taken from the UCI Machine Learning Repository and is also present in sklearn's **datasets** module. It 
has 14 explanatory variables describing various aspects of residential homes in Boston, the challenge is to predict the 
median value of owner-occupied homes per $1000s.

```python
from sklearn.datasets import load_boston
boston = load_boston() 
```

To check description of the dataset: 
```python
print(boston.DESCR)
```
```sh
Boston house prices dataset
---------------------------

**Data Set Characteristics:**  

    :Number of Instances: 506 

    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.   
     
.. topic:: References

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.

```

### Dataset to DataFrame

Convert **boston.data** to dataframe using pandas

```python# for features
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
# for target
boston_df['PRICE'] = boston.target
```

Check for NULL values in any of the features. If present, then we might have to clean the dataframe.
```python
print(boston_df.info())
```
```sh
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 506 entries, 0 to 505
Data columns (total 14 columns):
CRIM       506 non-null float64
ZN         506 non-null float64
INDUS      506 non-null float64
CHAS       506 non-null float64
NOX        506 non-null float64
RM         506 non-null float64
AGE        506 non-null float64
DIS        506 non-null float64
RAD        506 non-null float64
TAX        506 non-null float64
PTRATIO    506 non-null float64
B          506 non-null float64
LSTAT      506 non-null float64
PRICE      506 non-null float64
dtypes: float64(14)
memory usage: 55.5 KB
None
```

## Importing XGBoost and using it
If you plan to use XGBoost on a dataset which has categorical features you may want to consider applying some encoding 
(like one-hot encoding) to such features before training the model. Also, if you have some missing values such as NA in 
the dataset you may or may not do a separate treatment for them, because XGBoost is capable of handling missing values 
internally. You can check out this link if you wish to know more on this.

Without delving into more exploratory analysis and feature engineering, we will now focus on applying the algorithm to 
train the model on this data.

We will build the model using Trees as base learners (which are the default base learners) using XGBoost's scikit-learn 
compatible API. Along the way, we will also see some of the common tuning parameters which XGBoost provides in order to 
improve the model's performance, and using the root mean squared error (RMSE) performance metric to check the performance
of the trained model on the test set. Root mean Squared error is the square root of the mean of the squared differences 
between the actual and the predicted values. As usual, we will start by importing the library xgboost and other important 
libraries that you will be using for building the model.

```python
# Import xgboost 
import xgboost

# divide the dataframe into X, y
X, y = boston_df.iloc[:, :-1], boston_df.iloc[:, -1]
```

Convert the dataset into optimized data structure called as **Dmatrix**. XGBoost supports this structure and gives good
performance.
```python 
boston_dmatrix = xgboost.DMatrix(data=X, label=y)
```
But, we will use later this property. 

### Tuning XGBoost Hyper-parameters

At this point, before building the model, we should be aware of the tuning parameters that XGBoost provides. There are a
many tuning parameters for tree-based learners in XGBoost.

1. learning_rate    :   step size shrinkage used to prevent overfitting. Range is [0,1]
2. max_depth        :   determines how deeply each tree is allowed to grow during any boosting round.
3. subsample        :   percentage of samples used per tree. Low value can lead to underfitting.
4. colsample_bytree :   percentage of features used per tree. High value can lead to overfitting.
5. n_estimators     :   number of trees you want to build.
6. objective        :   determines the loss function to be used like reg:linear for regression problems, reg:logistic 
                        for classification problems with only decision, binary:logistic for classification problems with
                        probability.
                  
XGBoost also supports regularization parameters to penalize models as they become more complex and reduce them to simple
(parsimonious) models.

1. gamma            :   controls whether a given node will split based on the expected reduction in loss after the 
                        split. A higher value leads to fewer splits. Supported only for tree-based learners.
2. alpha            :   L1 regularization on leaf weights. A large value leads to more regularization.
3. lambda           :   L2 regularization on leaf weights and is smoother than L1 regularization.

It's also worth mentioning that though we are using trees as base learners, we can also use XGBoost's relatively less 
popular linear base learners and one other tree learner known as dart. All we have to do is set the booster parameter to
either **gbtree**(_default_), **gblinear** or **dart**.

### Split and Apply XGBRegressor
```python
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

The next step is to instantiate an XGBoost _regressor_ object by calling the **XGBRegressor()** class from the XGBoost 
library with the hyper-parameters passed as arguments. For classification problems, use **XGBClassifier()** class.

After that, fit the _regressor_ to the training set and make predictions on the test set using the familiar **.fit()** 
and **.predict()** methods.
```python
# Apply xgbregressor on the dataset
xgb_reg = xgboost.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                               max_depth=5, reg_alpha=10, n_estimators=10)

xgb_reg.fit(X_train, y_train)
predictions = xgb_reg.predict(X_test)
```

Now let is compute the RMSE
```python
print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test, predictions))))
```
```sh
RMSE: 9.106604511148243
```
We can see that your RMSE for the price prediction came out to be around 9.1 per 1000$.

### k-fold Cross Validation using XGBoost
In order to build more robust models, it is common to do a k-fold cross validation where all the entries in the original 
training dataset are used for both training as well as validation. Also, each entry is used for validation just once. 
XGBoost supports k-fold cross validation via the cv() method. All we have to do is specify the _nfolds_ parameter, which 
is the number of cross validation sets we want to build.

1. num_boost_round      :   denotes the number of trees you build (analogous to n_estimators)
2. metrics              :   tells the evaluation metrics to be watched during CV
3. as_pandas            :   to return the results in a pandas DataFrame.
4. early_stopping_rounds:   finishes training of the model early if the hold-out metric ("rmse" in our case) does not 
                            improve for a given number of rounds.
5. seed                 :   for reproducibility of results.

This time we will create a hyper-parameter dictionary params which holds all the hyper-parameters and their values as 
key-value pairs but will exclude the _n_estimators_ from the hyper-parameter dictionary using **num_boost_rounds**.

We will use these parameters to build a 3-fold cross validation model by invoking XGBoost's cv() method and store the 
results in a cv_results DataFrame. Note that here we are using the Dmatrix object which we created before.

```python
# Using K-fold cross validation
params = {"objective": "reg:squarederror", "colsample_bytree": 0.3, "learning_rate": 0.1, "max_depth": 5,
          "reg_alpha": 10}
xgb_reg_cv = xgboost.cv(dtrain=boston_dmatrix, params=params, nfold=3, num_boost_round=400, early_stopping_rounds=10,
                        metrics="rmse", as_pandas=True, seed=42)

# Calculate root mean squared error
print((xgb_reg_cv["test-rmse-mean"]).tail(1))
```

```sh
245    3.593899
Name: test-rmse-mean, dtype: float64
```

We can see that your RMSE for the price prediction has reduced as compared to last time and came out to be around 3.59 
per 1000$. We can reach an even lower RMSE for a different set of hyper-parameters.

### Visualize Feature importance
Once we train a model using the XGBoost learning API, count the number of times each feature is split on across all 
boosting rounds (trees) in the model, and then visualizing the result as a bar graph, with the features ordered 
according to how many times they appear. XGBoost has a plot_importance() function that allows us to do exactly this.

```python
# Visualize the feature importance
xg_reg = xgboost.train(params=params, dtrain=boston_dmatrix, num_boost_round=10)
xgboost.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show(block=True)
```

![F-score](https://user-images.githubusercontent.com/35737777/66647513-07c5f300-ec21-11e9-846d-3348e7f179d3.png)
