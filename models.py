import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from utilities import create_csv
import os.path
import skops.io as sio

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
	>>> from models import VeloCaster
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
			futureBookings[i] = int(futureBookings[i])
			if(futureBookings[i]>60): futureBookings[i] = 60

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
	>>> from models import WeekVeloCaster
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

			# deleting index column
			week.drop('index', inplace=True, axis=1)

			# inserting 0 days gap row if there isn't one.
			if(week['gap'].iloc[-1].days != 0):
				week.loc[week.shape[0]] = [self.today-self.today, 0, 0]

			# Setting last element of Cummulative Distribution Function as percentage of bookings on max_days'th day.
			cdf[max_days-1] = week['bookPerc'][0]
			i, j = max_days-2, 1

			# Calculating Cummulative Distribution Function for t'th weekday.
			while i >= 0:
				try:
					day = week['gap'][j].days
				except:
					day = 0
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
			futureBookings[i] = int(futureBookings[i])
			if(futureBookings[i]>60): futureBookings[i] = 60
		return futureBookings

class HybridCaster():
	""" Hybrid Forecaster.
	
	Used to forecast bookings using both forecaster and a baseModel.
	We sample a dataset based on two forecaster with numDays=14 and numDays=28 signifying short and longterm forecasters.
	Using the sampled dataset we train a baseModel.
	To forecast next week, we get two forecaster outputs, short and longterm output, we forecast the final results by passing 
	this short and long term output to the basemodel.
	Hybrid Forecaster Mean Absolute error is approx 50% less than VeloCaster and WeekVeloCaster. 

	Parameters
	----------
	payDates : {pandas.Series}
		Series of Dates in which the customer booked.
	bookDates : {pandas.Series}
		Series of Dates in which the slot was booked.
	numDays : {int}, default = 63
		Number of past days used for calculating cummulative distribution function.
	forecaster : {VeloCaster or WeekVeloCaster}, default = VeloCaster
		Forecaster used for short and long term outputs.
	baseModel : {sklearn regression model}, default = RandomForestRegressor()
		baseModel used for processing short and long term outputs and forecast more accurate output.
	useExistingModel : {bool}, defualt = True
		whether to use existing model saved in models/model.skops or train a new model.
	dataset: {pandas.DataFrame}, default = None
		dataset on which the baseModel is trained upon, if None data/train.csv is used.
	updateDataset: {bool}, default = False
		whether to update the dataset in data/train.csv.
	
	Examples
	----------
	>>> from models import HybridCaster
	>>> model = HybridCaster(df['payDates'], df['bookDates'])
	>>> y_pred = model.forecast()
	"""
	def __init__(self, payDates, bookDates, forecaster=VeloCaster, baseModel=RandomForestRegressor(), useExistingModel=True, dataset=None, updateDataset=False):		
		self.df = pd.DataFrame(columns = ['payDates', 'bookDates'])
		self.df['payDates'] = payDates
		self.df['bookDates'] = bookDates
		self.df['gap'] = bookDates - payDates
		# Automatically set today as max value in payDates.
		self.today = self.df['payDates'].max().date()

		# 14 days for short term forecasting and 28 days for long term forecasting.
		self.forecaster_14 = forecaster(self.df['payDates'], self.df['bookDates'], numDays=14)
		self.forecaster_28 = forecaster(self.df['payDates'], self.df['bookDates'], numDays=28)
		self.forecaster_14.train()
		self.forecaster_28.train()

		# if updateDataset or there is no train.csv in data/ folder, create/update the data/train.csv
		if updateDataset or not os.path.isfile('data/train.csv'):
			create_csv(payDates, bookDates, Forecaster=VeloCaster)
		# if dataset is None, use data/train.csv
		if dataset is None:
			dataset = pd.read_csv('data/train.csv')
		
		# if useExistingModel and there is a model persistance available in models/ folder, load that model.
		if useExistingModel and os.path.isfile('models/model.skops'):
			unknown_types = sio.get_untrusted_types(file = "models/model.skops")
			self.baseModel = sio.load("models/model.skops", trusted=unknown_types)
		# else use the baseModel provided in the argument and train using dataset.
		else:
			self.baseModel = baseModel
			X = dataset[['14_days', '28_days']].to_numpy()
			y = dataset['target'].to_numpy()
			self.baseModel.fit(X, y)
			obj = sio.dump(self.baseModel, "models/model.skops")
	
	# Not needed.
	def train(self):
		pass

	def forecast(self):
		# Getting short term and long term outputs.
		y_pred_14 = self.forecaster_14.forecast()
		y_pred_28 = self.forecaster_28.forecast()
		
		# Arranging output and getting predictions using baseModel.
		X = np.array([[y_pred_14[i], y_pred_28[i]] for i in range(len(y_pred_14))])
		final_y_pred = self.baseModel.predict(X)

		# limiting the booking to integer and maximum slots.
		for i in range(len(final_y_pred)):
			final_y_pred[i] = int(final_y_pred[i])
			if(final_y_pred[i]>60): final_y_pred[i] = 60			

		return final_y_pred