from models.competitors_models import C_HybridCaster
import pandas as pd
from utilities.competitor_util import load_competitors_df
from datetime import timedelta

def get_all_brands(path=r'data/competitors_data.xlsx'):
    brands = pd.read_excel(path, usecols='C')
    brands.columns = ['brand']
    brands = brands['brand'].unique()
    return brands

def forecast_all_brands(path=r'data/competitors_data.xlsx', required_brands=[]):
    main_brands = required_brands
    all_brands = get_all_brands(path)
    other_brands = list(filter(lambda x: x not in main_brands, all_brands))

    # Creating resultant dataset
    columns = ['date']
    if main_brands:
        columns.extend(main_brands)
    columns.append('others')
    result = pd.DataFrame(columns=columns)

    for brand in main_brands:
        print("Brand: ", brand)
        df = load_competitors_df(path, brand)
        forecaster = C_HybridCaster(df, brand)
        y_pred = forecaster.forecast()
        result[brand] = y_pred
        print(f'predictions: {y_pred}\n')

    # Others
    print("Brand: Others")
    df = load_competitors_df(path, other_brands)
    forecaster = C_HybridCaster(df, 'other_brands')
    y_pred = forecaster.forecast()
    result['others'] = y_pred
    print(f'predictions: {y_pred}\n')

    # filling up date column
    result['date'] = [forecaster.today+timedelta(days=i+1) for i in range(6)]
    result.reset_index(drop=True, inplace=True)

    print(result)
    result.to_csv(r'data/competitors_output.csv')

def get_max_slot(path='data/competitors_data.xlsx', save_file=True):
    brands = get_all_brands(path)
    dict = {}
    df = pd.read_excel(path, usecols="A,C:E", parse_dates=['Date of recording'])
    df.columns = ['date', 'brand', 'location', 'slots']
    total = 0
    for brand in brands:
        tempDf = df[df['brand'] == brand]
        tempDf = tempDf.groupby(['date'])['slots'].sum().reset_index()
        dict[brand] = tempDf['slots'].iloc[-1]
        total += dict[brand]
        # print(total)
    dict['all'] = total
    print(dict)
    df = pd.DataFrame(dict, index=[0])
    df.to_csv('data/max_slots.csv')
    return dict

if __name__ == "__main__":
    print(get_max_slot('../competitors_data.xlsx'))
    forecast_all_brands('../competitors_data.xlsx')

