from models import WeekVeloCaster
from utilities import load_dataset

import warnings
warnings.filterwarnings("ignore")

df = load_dataset('../raw_data_bookings.csv')

weekVeloCaster = WeekVeloCaster(df['pay_date'], df['dateOfBooking'])
weekVeloCaster.train()
y_pred = weekVeloCaster.forecast()

print("Forecasting using WeekVeloCaster: ", y_pred)
# Forecasting using WeekVeloCaster:  [49. 51. 44. 48. 31. 42. 55.]