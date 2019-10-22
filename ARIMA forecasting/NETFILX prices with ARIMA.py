import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

import plotly.figure_factory as ff
import plotly
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = 'browser'

warnings.filterwarnings('ignore')

df = pd.read_csv('NFLX.csv')
print(df.head())
print(f'shape of NFLX data: {df.shape}')
print(f'columns of NFLX data: {df.columns}')
#
# df['Close'].plot()
# plt.title('NETFLIX Prices')
#
# dr = df.cumsum()
# dr.plot()
# plt.title('NETFLIX Cumulative Returns')
#
# lag = 2
# lag_plot(df['Close'], lag=lag)
# plt.title('NETFLIX Atuocorrelation plot')

length = 0.8
train_data, test_data = df[0:int(len(df) * length)], df[int(len(df) * length):]
# plt.title('NETFLIX Prices')
# plt.xlabel('Dates')
# plt.ylabel('Prices')
# plt.plot(train_data['Close'], 'blue', label='Training Data')
# plt.plot(test_data['Close'], 'green', label='Testing Data')
# plt.xticks(np.arange(0, 4383, 300), df['Date'][0:4383:300])
# plt.legend()
# plt.show()

def smap_error(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

train_ar = train_data['Close'].values
test_ar = test_data['Close'].values

history = [x for x in train_ar]
predictions = list()

for t in range(len(test_ar)):
    model = ARIMA(history, order=(3, 2, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0][0]
    predictions.append(yhat)
    actual_data = test_ar[t]
    history.append(actual_data)
    print(f'예측값 = {yhat:.6}, 실제값 = {actual_data:.6}, 오차 = {yhat - actual_data:.6}')

error1 = mean_squared_error(test_ar, predictions)
print(f'오차제곱평균(MSE, Mean Squared Error): {error1:.6}')
error2 = smap_error(test_ar, predictions)
print(f'대칭절대백분율오차평균(SMAPE; Symmetric Mean Absolute Percentage Error): {error2:.6}')

plt.plot(train_data['Close'], color='blue', label='Training Data')
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', label='Predicted Price')
plt.plot(test_data.index, test_data['Close'], color='red', label='Actual Price')
plt.title('NETFLIX Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(0, 4383, 300), df['Date'][0:4383:300])
plt.legend()
plt.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=train_data['Date'],
                         y=train_data['Close'],
                         mode='lines',
                         name='훈련 데이터'))

fig.add_trace(go.Scatter(x=test_data['Date'],
                         y=test_data['Close'],
                         mode='lines',
                         name='실제 주가'))

fig.add_trace(go.Scatter(x=test_data['Date'],
                         y=predictions,
                         mode='lines+markers',
                         name='예측한 주가'))

fig.update_layout(title=go.layout.Title(text=f"ARIMA(2, 2, 0) 모델로 예측한 NETFLIX 주가"))
fig.show()