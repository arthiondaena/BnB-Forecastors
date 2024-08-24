from models import HybridCaster
from utilities import load_dataset

import warnings
warnings.filterwarnings("ignore")

df = load_dataset('../raw_data_bookings.csv')

hybridCaster = HybridCaster(df['pay_date'], df['dateOfBooking'])
hybridCaster.train()
y_pred = hybridCaster.forecast()

print("Forecasting using HybridCaster: ", y_pred)
# Forecasting using HybridCaster:  [49. 56. 54. 47. 49. 47. 56.]