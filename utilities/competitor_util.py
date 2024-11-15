import numpy as np
import pandas as pd
from datetime import timedelta

def load_competitors_df(path='data/competitors_data.xlsv', brand=None):
	if type(brand) is str:
		brand = [brand]

	df = pd.read_excel(path, usecols="A:D,M:S", parse_dates=['Date of recording'])
	df.columns = ['date', 'city', 'brand', 'location', 'day0', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6']
	days = ['day0', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6']
	df = df[df['brand'].isin(brand)]
	df = df.groupby(['date'])[days].sum().reset_index()
	return df

def check_dates_exist(unique_dates, date, numDays=7):
	for j in range(numDays):
		tempDate = date + timedelta(days=j)
		if np.datetime64(tempDate) not in unique_dates:
			return False
	return True

def mean_absolute_error(y_true, y_pred):
	output_errors = np.average(abs(y_true - y_pred), axis=0)
	return output_errors

def get_cv_errors_competitors(estimator, df, numDays=None, cv=5):
	date = df['date'].max().date() - timedelta(days=cv+15)
	unique_dates = df['date'].unique()
	errors = np.zeros(cv)
	i, day = 0, 0

	while i < cv:
		currentDate = date + timedelta(days=day)
		# print(type(currentDate))

		# Removing missing dates
		# flag = False
		# for j in range(7):
		# 	tempDate = currentDate + timedelta(days=j)
		# 	if np.datetime64(tempDate) not in unique_dates:
		# 		flag = True
		# 		break
		# if flag:
		# 	day += 1
		# 	continue
		if not check_dates_exist(unique_dates, currentDate, 7):
			day += 1
			continue

		tempDf = df[df['date'] <= pd.to_datetime(currentDate)]

		if numDays is None:
			model = estimator(tempDf)
		else:
			model = estimator(tempDf, numDays)
		y_pred = model.forecast()
		y_true = df[(df['date'] > pd.to_datetime(currentDate)) & (df['date'] < pd.to_datetime(currentDate + timedelta(days=7)))]
		y_true = y_true['day0'].to_numpy()
		errors[i] = mean_absolute_error(y_true, y_pred)
		# if errors[i] > 15:
		# 	print("More than 15 error: ")
		# 	print("df: ", tempDf.iloc[-1].to_list())
		# 	print("date: ", currentDate)
		# 	print("error: ", errors[i])
		# 	print("cdf: ", model.getCdf())
		# 	print("y_true: ", y_true)
		# 	print("y_pred: ", y_pred)
		# 	print("----------------------------------")

		day += 1
		i += 1

	# print(errors)
	return errors

def cross_val_error_competitors(estimator, df, numDays=None, cv=5):
	return np.average(get_cv_errors_competitors(estimator, df, numDays, cv))

def create_csv_competitors(df, Forecaster, filename = 'data/competitors_train.csv', saveFile=True):
	days = [7, 14, 21, 28]
	unique_dates = df['date'].unique()

	today = df['date'].max().date()
	minDate = df['date'].min().date() + timedelta(days=30)
	maxDate = df['date'].max().date() - timedelta(days=7)
	# print(maxDate)

	cols = [f"{day}_days" for day in days]
	oth = ["nth_day", "week_day", "target"]
	cols.extend(oth)
	result = pd.DataFrame(columns = cols)

	#3d list for df
	ls = np.empty((len(cols), int((maxDate-minDate).days), 6), dtype=object)

	i = 0
	while minDate < maxDate:
		if not check_dates_exist(unique_dates, minDate, 7):
			minDate += timedelta(days=1)
			continue

		df1 = df[df['date'] <= pd.to_datetime(minDate)]
		y_true = df[(df['date'] > pd.to_datetime(minDate)) & (df['date'] < pd.to_datetime(minDate + timedelta(days=7)))]
		y_true = y_true['day0'].to_numpy()

		ls[len(cols)-1, i] = y_true

		# n-th day values.
		ls[len(days), i] = np.array([n for n in range(1, 7)])
		# weekday values.
		ls[len(days)+1, i] = np.array([int(minDate.weekday()+n+1)%7 for n in range(1, 7)])

		for j in range(len(days)):
			forecaster = Forecaster(df1, numDays=days[j])
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

def nth_day_errors_competitors(estimator, df, brand='temp', numDays=None, cv=5):
	date = df['date'].max().date() - timedelta(days=cv+15)
	errors = np.empty((cv, 6))
	unique_dates = df['date'].unique()
	count = 0

	for i in range(cv):
		currentDate = date + timedelta(days=i)
		if not check_dates_exist(unique_dates, currentDate, 7):
			errors[i] = np.zeros(6)
			continue
		tempDf = df[df['date'] <= pd.to_datetime(currentDate)]
		if numDays is None:
			model = estimator(tempDf, brand)
		else:
			model = estimator(tempDf, numDays=numDays)
		y_pred = model.forecast()
		y_true = df[(df['date'] > pd.to_datetime(currentDate)) & (df['date'] < pd.to_datetime(currentDate + timedelta(days=7)))]
		y_true = y_true['day0'].to_numpy()
		booked_slots = df[(df['date'] == pd.to_datetime(currentDate))]
		booked_slots = booked_slots[['day'+str(x) for x in range(1, 7)]].to_numpy()
		errors[i] = abs(y_true - y_pred)

		# if errors[i].max() > 20:
		# 	count += 1
		# 	print(currentDate)
		# 	print("y_true",  y_true)
		# 	print("y_pred", y_pred)
		# 	print("errors: ", errors[i])
		# 	print("booked", booked_slots)
		# 	print("------------------------------------\n")
		# print(errors[i])
	errors = errors[~np.all(errors == 0, axis=1)]
	# print("total size: ", errors.shape[0])
	# print("> 20 error size: ", count)
	return errors.T

def data_point_error_report(estimator, df, brand='temp', numDays=None, cv=5, saveFile=True):
	date = df['date'].max().date() - timedelta(days=cv + 15)
	# errors = np.empty((cv, 6))
	unique_dates = df['date'].unique()

	columns = ['Date_of_prediction']
	main_cols = ['Actual', 'Forecasted', 'Slots_Booked']
	days = ['+1', '+2', '+3', '+4', '+5', '+6']

	for cols in main_cols:
		columns.extend([cols+day for day in days])
	columns.append('Error')
	result = pd.DataFrame(columns=columns)

	for i in range(cv):
		currentDate = date + timedelta(days=i)
		if not check_dates_exist(unique_dates, currentDate, 7):
			continue

		tempDf = df[df['date'] <= pd.to_datetime(currentDate)]
		if numDays is None:
			model = estimator(tempDf, brand)
		else:
			model = estimator(tempDf, numDays=numDays)
		y_pred = model.forecast()
		y_true = df[
			(df['date'] > pd.to_datetime(currentDate)) & (df['date'] < pd.to_datetime(currentDate + timedelta(days=7)))]
		y_true = y_true['day0'].to_numpy()
		booked_slots = df[(df['date'] == pd.to_datetime(currentDate))]
		booked_slots = booked_slots[['day' + str(x) for x in range(1, 7)]].to_numpy()

		error = mean_absolute_error(y_true, y_pred)

		y_pred = y_pred.tolist()
		y_true= y_true.tolist()
		booked_slots = booked_slots.flatten().tolist()

		row = [str(currentDate)]
		row.extend(y_pred)
		row.extend(y_true)
		row.extend(booked_slots)
		row.append(error)
		# print(row)

		result.loc[len(result)] = row

	if saveFile:
		result.to_csv('data/data_point_error_report.csv', index=False)

	return result.reset_index(drop=True)

def get_all_brands(path=r'data/competitors_data.xlsx'):
	brands = pd.read_excel(path, usecols='C')
	brands.columns = ['brand']
	brands = brands['brand'].unique()
	return brands

def all_brands_data_point_error(estimator, brands, path='data/competitors_data.xlsx', cv=5):
	main_brands = brands
	all_brands = get_all_brands(path)
	other_brands = list(filter(lambda x: x not in main_brands, all_brands))
	result = pd.DataFrame()

	for brand in main_brands:
		df = load_competitors_df(path, brand)
		report = data_point_error_report(estimator, df, brand, cv=cv, saveFile=False)
		report['brand'] = brand
		result = pd.concat([result, report])

	# Others
	df = load_competitors_df(path, other_brands)
	report = data_point_error_report(estimator, df, 'others', cv=cv)
	report['brand'] = 'others'
	result = pd.concat([result, report])

	result = result[['Date_of_prediction', 'brand'] +
					[col for col in result.columns if col != 'brand' and col != 'Date_of_prediction']]

	result.sort_values(by=['Date_of_prediction'], inplace=True)

	result.to_csv('data/all_brands_data_point_error.csv', index=False)