import numpy as np
import pandas as pd
from datetime import timedelta
from random import randint

def mean_absolute_error(y_true, y_pred):
	output_errors = np.average(abs(y_true - y_pred), axis=0)
	return output_errors

def get_cv_errors(estimator, payDates, bookDates, numDays=None, cv=5):
	df = pd.DataFrame(columns = ['payDates', 'bookDates'])
	df['payDates'] = payDates
	df['bookDates'] = bookDates
	minDate = df['payDates'].min().date() + timedelta(days=63)
	maxDate = df['payDates'].max().date() - timedelta(days=8)
	errors = np.zeros(cv)

	for i in range(cv):
		randomDate = minDate + timedelta(days=randint(0, int((maxDate-minDate).days)))
		tempDf = df[df['payDates'] <= pd.to_datetime(randomDate)]
		if numDays is None:
			model = estimator(tempDf['payDates'], tempDf['bookDates'])
		else:
			model = estimator(tempDf['payDates'], tempDf['bookDates'], numDays=numDays)
		y_pred = model.forecast()
		y_true = df[(df['bookDates'] > pd.to_datetime(randomDate)) & (df['bookDates'] < pd.to_datetime(randomDate+timedelta(days=8)))]
		y_true = y_true['bookDates'].value_counts().reset_index().sort_values(by='bookDates')
		y_true = y_true['count'].to_numpy()
		errors[i] = mean_absolute_error(y_true, y_pred)

	return errors

def cross_val_error(estimator, payDates, bookDates, numDays=None, cv=5):
	return np.average(get_cv_errors(estimator, payDates, bookDates, numDays, cv))

def create_csv(payDates, bookDates, Forecaster, days:list = []):
	if not days:
		days = [7, 14, 21, 28]
	df = pd.DataFrame(columns = ['payDates', 'bookDates'])
	df['payDates'] = payDates
	df['bookDates'] = bookDates
	minDate = df['payDates'].min().date() + timedelta(days=63)
	maxDate = df['payDates'].max().date() - timedelta(days=8)
	cols = [f"{day}_days" for day in days]
	cols.append("target")
	result = pd.DataFrame(columns = cols)
	ls = [np.array([])] * (len(days)+1)

	for _ in range(100):
		randomDate = minDate + timedelta(days=randint(0, int((maxDate-minDate).days)))

		df1 = df[(df['payDates'] <= pd.to_datetime(randomDate))]

		# Retrieving actual bookings of the week.
		y = df[(df['bookDates'] > pd.to_datetime(randomDate)) & (df['bookDates']<pd.to_datetime(randomDate+timedelta(days=8)))]
		y = y['bookDates'].value_counts().reset_index().sort_values(by='bookDates')
		y_true = y['count'].to_numpy()

		ls[len(days)] = np.concatenate([ls[len(days)], (y_true)])

		for i in range(len(days)):
			forecaster = Forecaster(df1['payDates'], df1['bookDates'], numDays=days[i])
			forecaster.train()
			y_pred = forecaster.forecast()

			ls[i] = np.concatenate([ls[i], (y_pred)])
				
	for i in range(len(ls)):
		result[cols[i]] = ls[i]
	result.to_csv("data/train.csv")