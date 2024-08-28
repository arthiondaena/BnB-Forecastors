from utilities import *
from models import *
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm 

import warnings
warnings.filterwarnings("ignore")

prob_df1 = load_dataset('../raw_data_bookings.csv')

# print("cv error on 100 subsets: ")
print("Mean Absolute error of VeloCaster_14: ", cross_val_error(VeloCaster, prob_df1['pay_date'], prob_df1['dateOfBooking'], numDays=14, cv=50))
# # Mean Absolute error of VeloCaster:  6.138571428571429
print("Mean Absolute error of VeloCaster_28: ", cross_val_error(VeloCaster, prob_df1['pay_date'], prob_df1['dateOfBooking'], numDays=28, cv=50))
# Mean Absolute error of WeekVeloCaster:  6.24142857142857
# print("Mean Absolute error of HybridCaster: ", cross_val_error(HybridCaster, prob_df1['pay_date'], prob_df1['dateOfBooking'], cv=50))
# Mean Absolute error of HybridCaster:  3.717142857142857

# # Plotting how the errors are distributed.
# velo_errors = get_cv_errors(VeloCaster, prob_df1['pay_date'], prob_df1['dateOfBooking'], 14, cv=50)
# velo_errors.sort()
# velo_mean = statistics.mean(velo_errors)
# velo_sd = statistics.stdev(velo_errors)

# week_errors = get_cv_errors(VeloCaster, prob_df1['pay_date'], prob_df1['dateOfBooking'], 28, cv=50)
# week_errors.sort()
# week_mean = statistics.mean(week_errors)
# week_sd = statistics.stdev(week_errors)

# hybrid_errors = get_cv_errors(HybridCaster, prob_df1['pay_date'], prob_df1['dateOfBooking'], cv=50)
# hybrid_errors.sort()
# hybrid_mean = statistics.mean(hybrid_errors)
# hybrid_sd = statistics.stdev(hybrid_errors)

# plt.plot(velo_errors, norm.pdf(velo_errors, velo_mean, velo_sd), label='VeloCaster_14')
# plt.plot(week_errors, norm.pdf(week_errors, week_mean, week_sd), label='VeloCaster_28')
# plt.plot(hybrid_errors, norm.pdf(hybrid_errors, hybrid_mean, hybrid_sd), label='HybridCaster')
# plt.legend()
# plt.savefig("graphs/errors_normal_14_28.png")
# plt.clf()

plt.figure(figsize=(12,4))
nth_day = nth_day_errors(HybridCaster, prob_df1['pay_date'], prob_df1['dateOfBooking'], cv=60)
for i in range(nth_day.shape[0]):
	nth_day[i].sort()
	nth_mean = statistics.mean(nth_day[i])
	nth_sd = statistics.stdev(nth_day[i])
	plt.plot(nth_day[i], norm.pdf(nth_day[i], nth_mean, nth_sd), label = f'day n+{i+1}')
plt.legend()
plt.savefig('graphs/errors_nth_day.png')
plt.clf()