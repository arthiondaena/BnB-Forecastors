from models import *
from utilities import load_dataset
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

prob_df = load_dataset('../raw_data_bookings.csv')

today = date.fromisoformat('2024-07-14')

prob_df1 = prob_df[(prob_df['pay_date'] <= pd.to_datetime(today))]

# Retrieving actual bookings of the week.
y = prob_df[(prob_df['dateOfBooking'] > pd.to_datetime(today)) & (prob_df['dateOfBooking']<pd.to_datetime(today+timedelta(days=8)))]
y = y['dateOfBooking'].value_counts().reset_index().sort_values(by='dateOfBooking')
y_true = y['count'].to_numpy()
print("Bookings of next week: ", y_true)
# Bookings of next week:  [45 41 43 47 39 48 54]

# Forecasting using VeloCaster.
veloCaster = VeloCaster(prob_df1['pay_date'], prob_df1['dateOfBooking'])
veloCaster.train()
y_pred_velo = veloCaster.forecast()
print("Forecasting using VeloCaster: ", y_pred_velo)
# Forecasting using VeloCaster:  [39. 34. 28. 48. 50. 58. 60.]

# Forecasting using WeekVeloCaster.
weekveloCaster = WeekVeloCaster(prob_df1['pay_date'], prob_df1['dateOfBooking'])
weekveloCaster.train()
y_pred_week = weekveloCaster.forecast()
print("Forecasting using WeekVeloCaster: ", y_pred_week)
# Forecasting using WeekVeloCaster:  [38. 32. 23. 43. 40. 48. 58.]

# Forecasting using HybridCaster.
hybrid = HybridCaster(prob_df1['pay_date'], prob_df1['dateOfBooking'])
hybrid.train()
y_pred_hybrid = hybrid.forecast()
print("Forecasting using HybridCaster: ", y_pred_hybrid)
# Forecasting using HybridCaster:  [46. 47. 48. 39. 49. 48. 49.]

# Plotting graph as saving it as plot.png.
plt.figure(figsize=(12,4))
plt.plot(y['dateOfBooking'], y['count'], label='y_true')
plt.plot(y['dateOfBooking'], y_pred_velo, label='VeloCaster')
plt.plot(y['dateOfBooking'], y_pred_week, label='WeekVeloCaster')
plt.plot(y['dateOfBooking'], y_pred_hybrid, label='HybridCaster')
plt.legend()
plt.savefig("graphs/comparison_3.png")
plt.clf()