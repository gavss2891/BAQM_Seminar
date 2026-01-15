import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import numpy as np

merged_data_path = os.path.join(os.getcwd(), 'data', 'merged_data')
master_train_path = os.path.join(merged_data_path, 'master_train.csv')

df = pd.read_csv(master_train_path, low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.convert_dtypes()

df.rename(columns={'date': 'ds', 'sales_volume_index': 'y'}, inplace=True)
df = df[['ds', 'y']]
df['y'] = np.log(df['y'])

m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
m.fit(df)

future = m.make_future_dataframe(periods=31, freq='D')
forecast = m.predict(future)

fig1 = m.plot(forecast)
plt.title('Forecast with Confidence Intervals')
plt.show()

fig2 = m.plot_components(forecast)
plt.show()