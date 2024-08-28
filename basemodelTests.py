import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

df = pd.read_csv("data/train.csv")
# X = df[['7_days', '14_days', '21_days', '28_days']].to_numpy()
X = df.drop(['target'], axis=1).to_numpy()
# X = df[['14_days', '28_days']].to_numpy()
print(df.columns)
print(X.shape)
y = df['target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print("7_days error: ", mean_absolute_error(X_test[:,0], y_test))
print("14_days error: ", mean_absolute_error(X_test[:,0], y_test))
# 14_days error:  6.214285714285714

# print("21_days error: ", mean_absolute_error(X_test[:,2], y_test))
print("28_days error: ", mean_absolute_error(X_test	[:,1], y_test))
# 28_days error:  6.328571428571428
print()

reg = LinearRegression().fit(X_train, y_train)
print("Linear Regression Error: ", mean_absolute_error(reg.predict(X_test), y_test))
# Linear Regression Error:  3.533498228890859

clf = svm.SVR(kernel="poly")
clf.fit(X_train, y_train)
print("SVR Error: ", mean_absolute_error(clf.predict(X_test), y_test))
# SVR Error:  3.2810288025060097

regr = RandomForestRegressor()
regr.fit(X_train, y_train)
print("Random Forest Regressor Error: ", mean_absolute_error(regr.predict(X_test), y_test))
# Random Forest Regressor Error:  3.445769774867062

adaregr = AdaBoostRegressor()
adaregr.fit(X_train, y_train)
print("AdaBoost Regressor Error: ", mean_absolute_error(adaregr.predict(X_test), y_test))
# AdaBoost Regressor Error:  3.533002079378498

reg = GradientBoostingRegressor()
reg.fit(X_train, y_train)
print("Gradient Boost Regressor Error: ", mean_absolute_error(reg.predict(X_test), y_test))
# Gradient Boost Regressor Error:  3.4389168301985387

model = xgb.XGBRegressor()
model.fit(X_train, y_train)
print("XGB Regressor Error: ", mean_absolute_error(model.predict(X_test), y_test))
# XGB Regressor Error:  3.4965758459908622