
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

df = pd.read_csv('NFLX.csv').fillna(0)

lag = 5
# lag_plot(df.Close, lag=lag)
# plt.title(f'NETFLIX Autocorrelation plot with lag={lag}')
# plt.show()

length = 0.8
train_data, test_data = df[0:int(len(df) * length)], df[int(len(df) * length):]

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=train_data['Date'],
#                          y=train_data['Close'],
#                          mode='lines',
#                          name='Training Data'))
#
# fig.add_trace(go.Scatter(x=test_data['Date'],
#                          y=test_data['Close'],
#                          mode='lines',
#                          name='Testing Data'))
# fig.update_layout(title=go.layout.Title(text="NETFLIX 주가"))
# fig.show()

def smap_error(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true))))

train_ar = train_data['Close'].values
test_ar = test_data['Close'].values

history = [x for x in train_ar]

predictions = list()
p = 2
d = 2
q = 3
#pdq = (0, 2, 1)
pdq = [(2, 2, 4), (2, 2, 3), (2, 2, 0), (2, 1, 1)]
#, (5, 1, 2)](5, 2, 0), (5, 1, 5)
    # , (4, 2, 5), (4, 2, 4), (4, 2, 3),
    #    (4, 2, 0), (4, 1, 3), (3, 2, 5), (3, 2, 0), (2, 2, 4), (2, 2, 3), (2, 2, 0), (2, 1, 1)]
# for p in range(6):
#     for d in range(1, 3):
#         for q in range(6):
#             model = ARIMA(history, order=(p, d, q))
#             try:
#                 model_fit = model.fit(disp=0)
#                 print(model_fit.summary())
#             except ValueError:
#                 pass

for pdq in pdq:
    for t in range(len(test_ar)):
        model = ARIMA(history, order=pdq)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_ar[t]
        history.append(obs)
    error = mean_squared_error(test_ar, predictions)
    print(f'ARIMA{pdq} 모델 테스트 오차제곱평균: {error:.6}')
    error2 = smap_error(test_ar, predictions)
    print(f'ARIMA{pdq} 모델 대칭절대백분율오차평균: {error2:.6}')
    print()

        # fig = go.Figure()
        #
        # fig.add_trace(go.Scatter(x=train_data['Date'],
        #                          y=train_data['Close'],
        #                          mode='lines',
        #                          name='Training Data'))
        #
        # fig.add_trace(go.Scatter(x=test_data['Date'],
        #                          y=test_data['Close'],
        #                          mode='lines',
        #                          name='실제 주가'))
        #
        # fig.add_trace(go.Scatter(x=test_data['Date'],
        #                          y=predictions,
        #                          mode='lines+markers',
        #                          name='예측한 주가'))
        #
        # fig.update_layout(title=go.layout.Title(text=f"ARIMA{pdq} 모델로 예측한 NETFLIX 주가"))
        # fig.show()