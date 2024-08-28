import numpy as np
import pandas as pd
from datetime import timedelta
from random import randint

def load_dataset(path='data/raw_data_bookings.csv'):
	df = pd.read_csv(path, encoding = 'unicode_escape', low_memory=False, lineterminator='\n')

	# Filter records which are Cancelled or Failed.
	df = df[(df['isBooked']==True) & (df['bookingStatus']=='Booked')]

	# Regular expression for extracting date from date time format
	regex = r'(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}\.\d{3}Z'

	# Replace the timestamp with the extracted date using regex.
	df['pay_date'] = df['payment_acknowledgement_time'].replace(regex, r'\1', regex=True)

	# Converting dtypes of dateOfBooking and pay_date columns to datetime type.
	df = df[['dateOfBooking', 'pay_date']]
	df['dateOfBooking'] = pd.to_datetime(df['dateOfBooking'], format='%d-%m-%Y')
	df['pay_date'] = pd.to_datetime(df['pay_date'], format='%Y-%m-%d')

	# Calculating cross validation error using mean_absolute_error metrics on 100 different subsets.
	df = df[(df['dateOfBooking'] >= pd.to_datetime('2024-01-01', format = '%Y-%m-%d'))]

	return df

def mean_absolute_error(y_true, y_pred):
	output_errors = np.average(abs(y_true - y_pred), axis=0)
	return output_errors

def get_cv_errors(estimator, payDates, bookDates, numDays=None, cv=5):
	df = pd.DataFrame(columns = ['payDates', 'bookDates'])
	df['payDates'] = payDates
	df['bookDates'] = bookDates
	date = df['payDates'].max().date() - timedelta(days=cv+15)
	errors = np.zeros(cv)

	for i in range(cv):
		currentDate = date + timedelta(days=i)
		tempDf = df[df['payDates'] <= pd.to_datetime(currentDate)]
		if numDays is None:
			freshDataset = True if i==0 else False
			model = estimator(tempDf['payDates'], tempDf['bookDates'], freshDataset=freshDataset)

		else:
			model = estimator(tempDf['payDates'], tempDf['bookDates'], numDays=numDays)
		y_pred = model.forecast()
		y_true = df[(df['bookDates'] > pd.to_datetime(currentDate)) & (df['bookDates'] < pd.to_datetime(currentDate+timedelta(days=8)))]
		y_true = y_true['bookDates'].value_counts().reset_index().sort_values(by='bookDates')
		y_true = y_true['count'].to_numpy()
		errors[i] = mean_absolute_error(y_true, y_pred)

	return errors

def cross_val_error(estimator, payDates, bookDates, numDays=None, cv=5):
	return np.average(get_cv_errors(estimator, payDates, bookDates, numDays, cv))

def create_csv(payDates, bookDates, Forecaster, days:list = [], filename='data/train.csv', saveFile=True):
	if not days:
		days = [7, 14, 21, 28]
	
	# dataframe
	df = pd.DataFrame(columns = ['payDates', 'bookDates'])
	df['payDates'] = payDates
	df['bookDates'] = bookDates

	# df date limits
	today = df['payDates'].max().date()
	minDate = df['bookDates'].min().date() + timedelta(days=30)
	maxDate = df['payDates'].max().date() - timedelta(days=7)

	# initializing columns
	cols = [f"{day}_days" for day in days]
	oth = ["nth_day", "week_day", "target"]
	cols.extend(oth)
	result = pd.DataFrame(columns = cols)

	# 3d list for df
	ls = np.empty((len(cols), int((maxDate-minDate).days), 7), dtype=object)

	i = 0
	while minDate < maxDate:
		df1 = df[(df['payDates'] <= pd.to_datetime(minDate))]

		# Retrieving actual bookings of the week.
		y = df[(df['bookDates'] > pd.to_datetime(minDate)) & (df['bookDates']<pd.to_datetime(minDate+timedelta(days=8)))]
		y = y['bookDates'].value_counts().reset_index().sort_values(by='bookDates')
		y_true = y['count'].to_numpy()

		ls[len(cols)-1, i] = y_true

		# n-th day values.
		ls[len(days), i] = np.array([n for n in range(1, 8)])
		# weekday values.
		ls[len(days)+1, i] = np.array([int(minDate.weekday()+n+1)%7 for n in range(1, 8)])

		for j in range(len(days)):
			forecaster = Forecaster(df1['payDates'], df1['bookDates'], numDays=days[j])
			forecaster.train()
			y_pred = forecaster.forecast()

			ls[j, i] = y_pred

		minDate += timedelta(days=1)
		i += 1
				
	for i in range(0, len(ls)):
		result[cols[i]] = ls[i].flatten()

	if saveFile:
		result.to_csv(filename, index=False)


	result.reset_index(drop=True, inplace=True)
	return result

def update_csv(payDates, bookDates, Forecaster, numDays=7, file='data/train.csv'):
	# dataframe
	df = pd.DataFrame(columns = ['payDates', 'bookDates'])
	df['payDates'] = payDates
	df['bookDates'] = bookDates

	# df date limits
	today = df['payDates'].max().date()
	minDate = today - timedelta(days=(numDays+30+8))
	df = df[(df['bookDates'] > pd.to_datetime(minDate))]

	update_df = create_csv(df['payDates'], df['bookDates'], Forecaster, saveFile=False)
	df = pd.read_csv(file)
	final_df = pd.concat([df, update_df], ignore_index=True)
	final_df.to_csv(file, index=False)

	return final_df

def nth_day_errors(estimator, payDates, bookDates, numDays=None, cv=5):
	df = pd.DataFrame(columns = ['payDates', 'bookDates'])
	df['payDates'] = payDates
	df['bookDates'] = bookDates
	date = df['payDates'].max().date() - timedelta(days=cv+15)
	errors = np.empty((cv, 7))

	for i in range(cv):
		currentDate = date + timedelta(days=i)
		tempDf = df[df['payDates'] <= pd.to_datetime(currentDate)]
		if numDays is None:
			freshDataset = True if i==0 else False
			model = estimator(tempDf['payDates'], tempDf['bookDates'], freshDataset=freshDataset)

		else:
			model = estimator(tempDf['payDates'], tempDf['bookDates'], numDays=numDays)
		y_pred = model.forecast()
		y_true = df[(df['bookDates'] > pd.to_datetime(currentDate)) & (df['bookDates'] < pd.to_datetime(currentDate+timedelta(days=8)))]
		y_true = y_true['bookDates'].value_counts().reset_index().sort_values(by='bookDates')
		y_true = y_true['count'].to_numpy()
		errors[i] = abs(y_true - y_pred)

	return errors.T