from models import VeloCaster
from utilities import load_dataset

import warnings
warnings.filterwarnings("ignore")

df = load_dataset('../raw_data_bookings.csv')

veloCaster = VeloCaster(df['pay_date'], df['dateOfBooking'])
veloCaster.train()
y_pred = veloCaster.forecast()

print("Forecasting using VeloCaster: ", y_pred)
# Forecasting using VeloCaster:  [52. 59. 54. 44. 29. 43. 55.]