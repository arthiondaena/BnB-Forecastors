from utilities import VeloCastor, WeekVeloCastor, cross_val_error
import pandas as pd
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('raw_data_bookings.csv', encoding = 'unicode_escape', low_memory=False, lineterminator='\n')
df = df[(df['isBooked']==True) & (df['bookingStatus']=='Booked')]

# Regular expression for extracting date from date time format
regex = r'(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}\.\d{3}Z'

# Replace the timestamp with the extracted date using regex.
df['pay_date'] = df['payment_acknowledgement_time'].replace(regex, r'\1', regex=True)

# Converting dtypes of dateOfBooking and pay_date columns to datetime type.
prob_df = df[['dateOfBooking', 'pay_date']]
prob_df['dateOfBooking'] = pd.to_datetime(prob_df['dateOfBooking'], format='%d-%m-%Y')
prob_df['pay_date'] = pd.to_datetime(prob_df['pay_date'], format='%Y-%m-%d')

today = date.fromisoformat('2024-07-14')

prob_df1 = prob_df[(prob_df['pay_date'] <= pd.to_datetime(today))]

# Retrieving actual bookings of the week.
y = prob_df[(prob_df['dateOfBooking'] > pd.to_datetime(today)) & (prob_df['dateOfBooking']<pd.to_datetime(today+timedelta(days=8)))]
y = y['dateOfBooking'].value_counts().reset_index().sort_values(by='dateOfBooking')
y_true = y['count'].to_numpy()
print("Bookings of next week: ", y_true)

# Forecasting using VeloCastor.
velocastor = VeloCastor(prob_df1['pay_date'], prob_df1['dateOfBooking'])
velocastor.train()
y_pred_velo = velocastor.forecast()
print("Forecasting using VeloCastor: ", y_pred_velo)

# Forecasting using WeekVelocastor.
weekvelocastor = WeekVeloCastor(prob_df1['pay_date'], prob_df1['dateOfBooking'])
weekvelocastor.train()
y_pred_week = weekvelocastor.forecast()
print("Forecasting using WeekVeloCastor: ", y_pred_week)

# Plotting graph as saving it as plot.png.
plt.figure(figsize=(12,4))
plt.plot(y['dateOfBooking'], y['count'], label='y_true')
plt.plot(y['dateOfBooking'], y_pred_velo, label='VeloCastor')
plt.plot(y['dateOfBooking'], y_pred_week, label='WeekVeloCastor')
plt.legend()
plt.savefig("plot.png")

# Calculating cross validation error using mean_absolute_error metrics on 20 different subsets.
print("\n----------------------------------------------------------------------------------\n")
prob_df1 = prob_df[(prob_df['pay_date'] >= pd.to_datetime('2024-01-01', format = '%Y-%m-%d'))]

print("cv error on 20 subsets: ")
print("Mean Absolute error of Velocastor: ", cross_val_error(VeloCastor, prob_df1['pay_date'], prob_df1['dateOfBooking'], cv=20))
print("Mean Absolute error of WeekVelocastor: ", cross_val_error(WeekVeloCastor, prob_df1['pay_date'], prob_df1['dateOfBooking'], cv=20))