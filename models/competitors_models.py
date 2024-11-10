import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from utilities.competitor_util import create_csv_competitors
import os.path
import skops.io as sio

class C_VeloCaster():
    def __init__(self, df, numDays=14):
        self.df = df
        self.numDays = numDays
        self.today = self.df['date'].max()
        self._cdf = None

    def train(self):
        startDate = self.today - timedelta(days=self.numDays)

        tempDf = self.df[self.df['date'] > startDate]

        days = ['day0', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6']
        trainDf = tempDf[days].sum().reset_index()
        trainDf.columns = ['day', 'count']

        trainDf['percentage'] = trainDf['count'] / trainDf['count'][0]

        self._cdf = trainDf['percentage'].to_numpy()

    def getCdf(self):
        return self._cdf

    def forecast(self):
        # If not trained, then train
        if self._cdf is None:
            self.train()

        futureBookings = np.zeros(6)
        for i in range(6):
            futureBookings[i] = self.df['day'+str(i+1)].iloc[-1]
            futureBookings[i] /= self._cdf[i+1]
            futureBookings[i] = int(futureBookings[i])

        return futureBookings

class C_HybridCaster():
    def __init__(self, df, brand='BingeNBash', forecaster=C_VeloCaster, baseModel=RandomForestRegressor()):
        self.df = df
        self.today = self.df['date'].max().date()
        self.days = [7, 14, 21, 28]
        self.brand = brand
        self.dataset_path = f'data/{brand}_train.csv'

        self.forecasters = [None] * len(self.days)
        for i in range(len(self.days)):
            self.forecasters[i] = forecaster(self.df, numDays=self.days[i])
            self.forecasters[i].train()

        # if not os.path.isfile(self.dataset_path) or self.today.weekday()==0:
        dataset = create_csv_competitors(df, forecaster, filename=self.dataset_path, saveFile=False)

        dataset.dropna(inplace=True)

        self.baseModel = baseModel
        X = dataset.drop(['target'], axis=1).to_numpy()
        y = dataset['target'].to_numpy()
        self.baseModel.fit(X, y)

    def train(self):
        pass

    def forecast(self):
        X = np.empty((6, 6))
        for i in range(len(self.days)):
            X[i] = self.forecasters[i].forecast()

        # n_th days array
        X[len(self.days)] = np.array([n for n in range(1, 7)])

        # weekday array
        X[len(self.days)+1] = np.array([int(self.today.weekday()+n+1)%7 for n in range(1, 7)])

        final_y_pred = self.baseModel.predict(X.T)

        for i in range(len(final_y_pred)):
            final_y_pred[i] = int(final_y_pred[i])

        return final_y_pred

if __name__ == "__main__":
    from utilities import load_competitors_df, cross_val_error_competitors, create_csv_competitors
    df = load_competitors_df('../competitors_data.xlsx')
    # print(C_VeloCaster(df).forecast())
    print(cross_val_error_competitors(C_HybridCaster, df, None, 50))
    # create_csv_competitors(df, C_VeloCaster)
