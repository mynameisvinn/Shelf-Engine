import fbprophet
from fbprophet import Prophet

import pandas as pd

from typing import List

def make_predictions(df, stockcode_id:str, period:List[str]):

    df_stock = df[df['StockCode'] == stockcode_id]  # segment dataframe by Stockcode id
    data = {'Date': period}
    
    for country in set(list(df_stock['Country'])):
        df_stock_country = df_stock[df_stock['Country'] == country]

        # ignore countries that have few sales since we want to be conservative
        if df_stock_country.shape[0] < 2:
            data[country] = [0] * len(period)
        else:
            x = df_stock_country[['date', 'Quantity']]
            x.columns = ['ds', 'y']
            
            model = Prophet(yearly_seasonality=True, daily_seasonality=True, weekly_seasonality=True)
            model.fit(x)

            prediction_period = {"ds":period}
            predictions = model.predict(pd.DataFrame(prediction_period))
            data[country] = predictions['yhat'].values
    return pd.DataFrame(data)