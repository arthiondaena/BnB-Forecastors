from utilities.competitor_util import *
from models.competitors_models import *
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm
from build import get_all_brands

brands = get_all_brands('../competitors_data.xlsx')
df = load_competitors_df('../competitors_data.xlsx', brand='BingeNBash')
# plt.figure(figsize=(12, 4))
# df.to_csv('data/bingeNbash.csv')
#
# nth_day = nth_day_errors_competitors(C_HybridCaster, df, brand='BingeNBash', cv = 75)
# for i in range(nth_day.shape[0]):
# 	nth_day[i].sort()
# 	nth_mean = statistics.mean(nth_day[i])
# 	nth_sd = statistics.stdev(nth_day[i])
# 	plt.plot(nth_day[i], norm.pdf(nth_day[i], nth_mean, nth_sd), label = f'day n+{i+1}')
# plt.legend()
# plt.show()
# plt.savefig('graphs/errors_nth_day_competitors.png')
# plt.clf()

# data_point_error_report(C_HybridCaster, df, brand='BingeNBash', cv = 75)
all_brands_data_point_error(C_HybridCaster, brands=['BingeNBash', 'BingeTown'], path='../competitors_data.xlsx',cv=75)