import numpy as np
import pandas as pd
from datetime import date, timedelta
from random import randint

class VeloCaster():
	""" Velocity Forecaster.
	
	Used to forecast bookings based on velocity which is provided by already booked slots.

	Parameters
	----------
	payDates : {pandas.Series}
		Series of Dates on which the customer booked.
	bookDates : {pandas.Series}
		Series of Dates on which the slot was booked.
	numDays : {int}, default = 14
		Number of past days used for calculating cummulative distribution function.
	
	Examples
	----------
	>>> from utilies import VeloCaster
	>>> model = VeloCaster(df['payDates'], df['bookDates'])
	>>> model.train()
	>>> y_pred = model.forecast()
	"""
	def __init__(self, payDates, bookDates, numDays=14):
		self.df = pd.DataFrame(columns = ['payDates', 'bookDates'])
		self.df['payDates'] = payDates
		self.df['bookDates'] = bookDates
		self.df['gap'] = bookDates - payDates
		self.numDays = numDays
		# Automatically set today as max value in payDates.
		self.today = self.df['payDates'].max().date()
		self._cdf = None
	
	def train(self):
		# startDate = today - numDays
		startDate = self.today - timedelta(days=self.numDays)

		# Filtering df to have bookDates in range (startDate, today].
		tempDf = self.df[(self.df['bookDates'] > pd.to_datetime(startDate)) & (self.df['bookDates'] <= pd.to_datetime(self.today))]

		# Group counting based on gap values.
		trainDf = tempDf['gap'].value_counts().reset_index()
		trainDf.sort_values(by='gap', inplace = True, ascending = False)
		trainDf.reset_index(inplace=True)
		
		# Calculating percentage of bookings on ith day before slot booked date.
		trainDf['bookPerc'] = trainDf['count'] / trainDf['count'].sum()
		
		# max_days = maximum no of gap between booked date and pay date
		max_days = trainDf['gap'][0].days+1
		self._cdf = np.zeros(max_days)
		
		# Setting last element of Cummulative Distribution Function as percentage of bookings on max_days'th day.
		self._cdf[max_days-1] = trainDf['bookPerc'][0]
		i, j = max_days - 2, 1
		
		# Calculating Cummulative Distribution Function.
		while i >= 0:
			day = trainDf['gap'][j].days
			if i > day:
				self._cdf[i] = self._cdf[i+1]
			else:
				self._cdf[i] = self._cdf[i+1] + trainDf['bookPerc'][j]
				j += 1
			i -= 1

	def getCdf(self):
		return self._cdf
	
	def forecast(self):
		# If not trained, then train.
		if self._cdf is None:
			self.train()

		futureBookings = np.zeros(7)
		for i in range(7):
			# Retrieving Number of bookings booked on i'th day after today.
			futureBookings[i] = self.df[self.df['bookDates'] == pd.to_datetime(self.today + timedelta(days=i+1))].shape[0]
			
			# Predicting i'th day booking by dividing current i'th day bookings by CDF of i'th day.
			futureBookings[i] /= self._cdf[i+1]

		return futureBookings

class WeekVeloCaster():
	""" Week Velocity Forecaster.
	
	Used to forecast bookings based on velocity which is provided by already booked slots
	catogorized by weekdays.

	Parameters
	----------
	payDates : {pandas.Series}
		Series of Dates in which the customer booked.
	bookDates : {pandas.Series}
		Series of Dates in which the slot was booked.
	numDays : {int}, default = 63
		Number of past days used for calculating cummulative distribution function.
	
	Examples
	----------
	>>> from utilies import WeekVeloCaster
	>>> model = WeekVeloCaster(df['payDates'], df['bookDates'])
	>>> model.train()
	>>> y_pred = model.forecast()
	"""
	def __init__(self, payDates, bookDates, numDays=63):
		self.df = pd.DataFrame(columns = ['payDates', 'bookDates'])
		self.df['payDates'] = payDates
		self.df['bookDates'] = bookDates
		self.df['gap'] = bookDates - payDates
		self.df['weekday'] = self.df['bookDates'].dt.dayofweek
		self.numDays = numDays
		# Automatically set today as max value in payDates.
		self.today = self.df['payDates'].max().date()
		self._weekCdf = None
	
	def train(self):
		# startDate = today - numDays
		startDate = self.today - timedelta(days=self.numDays)

		# Filtering df to have bookDates in range (startDate, today].
		tempDf = self.df[(self.df['bookDates'] > pd.to_datetime(startDate)) & (self.df['bookDates'] <= pd.to_datetime(self.today))]

		# Creating seperate dataframe for each weekday.
		weekDf = [None] * 7

		# For each day of the week.
		for i in range(7):
			# Group counting based on weekday and gap values.
			weekDf[i] = tempDf[tempDf['weekday']==i]['gap'].value_counts().reset_index()
			weekDf[i].sort_values(by='gap', ascending=False, inplace=True)

			# Calculating percentage of bookings on nth day before slot booked date.
			weekDf[i]['bookPerc'] = weekDf[i]['count'] / weekDf[i]['count'].sum()
			weekDf[i].reset_index(inplace=True)

		# Creating seperate cdfs' for each weekday.
		self._weekCdf = [None] * 7

		# For each day of the week.
		for t, week in zip(range(7), weekDf):
			# max_days = maximum no of gap between booked date and pay date
			max_days = week['gap'][0].days+1
			cdf = np.zeros(max_days)

			# Setting last element of Cummulative Distribution Function as percentage of bookings on max_days'th day.
			cdf[max_days-1] = week['bookPerc'][0]
			i, j = max_days-2, 1

			# Calculating Cummulative Distribution Function for t'th weekday.
			while i >= 0:
				day = week['gap'][j].days
				if i > day:
					cdf[i] = cdf[i+1]
				else:
					cdf[i] = cdf[i+1] + week['bookPerc'][j]
					j += 1
				i -= 1
			self._weekCdf[t] = cdf
	
	def getCdf(self):
		return self._weekCdf
	
	def forecast(self):
		# If not trained, then train.
		if self._weekCdf == None:
			self.train()

		futureBookings = np.zeros(7)
		for i in range(7):
			# Retrieving Number of bookings booked on i'th day after today.
			futureBookings[i] = self.df[self.df['bookDates'] == pd.to_datetime(self.today + timedelta(days=i+1))].shape[0]
			
			# Predicting i'th day booking by dividing current i'th day bookings by CDF of i'th day.
			futureBookings[i] /= self._weekCdf[(self.today.weekday()+i+1)%7][i+1]

		return futureBookings

def mean_absolute_error(y_true, y_pred):
	output_errors = np.average(abs(y_true - y_pred), axis=0)
	return output_errors

def cross_val_error(estimator, payDates, bookDates, cv=5):
	df = pd.DataFrame(columns = ['payDates', 'bookDates'])
	df['payDates'] = payDates
	df['bookDates'] = bookDates
	minDate = df['payDates'].min().date() + timedelta(days=63)
	maxDate = df['payDates'].max().date() - timedelta(days=8)
	totalError = 0

	for _ in range(cv):
		randomDate = minDate + timedelta(days=randint(0, int((maxDate-minDate).days)))
		tempDf = df[df['payDates'] <= pd.to_datetime(randomDate)]
		model = estimator(tempDf['payDates'], tempDf['bookDates'])
		y_pred = model.forecast()
		y_true = df[(df['bookDates'] > pd.to_datetime(randomDate)) & (df['bookDates'] < pd.to_datetime(randomDate+timedelta(days=8)))]
		y_true = y_true['bookDates'].value_counts().reset_index().sort_values(by='bookDates')
		y_true = y_true['count'].to_numpy()
		totalError += mean_absolute_error(y_true, y_pred)

	return totalError/cv
