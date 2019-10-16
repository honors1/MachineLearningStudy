from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston

import pandas as pd
import numpy as np
import math
from scipy.stats import norm

import plotly.figure_factory as ff
import plotly
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'


boston = load_boston()
california = fetch_california_housing()

dataset = pd.DataFrame(boston.data, columns=boston.feature_names)

dataset['target'] = boston.target

mean_expected_value = dataset['target'].mean()
print()
print(f'mean_expected_value = {mean_expected_value}')

mean_ev_numpy = np.mean(dataset['target'])
print()
print(f'mean_ev_numpy = {mean_ev_numpy}')

print()
print(f'mean_ev_numpy - mean_expected_value = {mean_ev_numpy - mean_expected_value}')

Squared_errors = pd.Series(mean_expected_value - dataset['target'])**2
print()
print(Squared_errors)

SSE = np.sum(Squared_errors)
print()
print(f'Sum of Squared Errors(SSE) = {SSE}')

hist_data = [Squared_errors]
group_labels = ['Squared Errors']

fig = ff.create_distplot(hist_data=hist_data,
                         group_labels=group_labels,
                         show_curve=False,
                         show_rug=False,
                         bin_size=77)
fig.show()

def standardize(x):
    return (x - np.mean(x)) / np.std(x)

def covariance(var1, var2):
    observations = len(var1)
    return np.sum((var1 - np.mean(var1)) * (var2 - np.mean(var2))) / observations

def correlation(var1, var2, bias=0):
    return covariance(standardize(var1), standardize(var2))

from scipy.stats.stats import pearsonr
print()
print(f"Our correlation estimation: {correlation(dataset['RM'], dataset['target']):.06}")
print(f"Correlation from Scipy pearsonr estimation: {pearsonr(dataset['RM'], dataset['target'])[0]:.06}")

fig = go.Figure(data=go.Scatter(x=dataset['RM'],
                                y=dataset['target'],
                                mode='markers',
                                name="RM vs Median Value($1000's)")
                )

fig.add_trace(go.Scatter(x=[dataset['RM'].min() - 1, dataset['RM'].max() + 1],
                         y=[dataset['target'].mean(), dataset['target'].mean()],
                         mode='lines',
                         line=dict(dash='dash',
                                   color='red')
                         )
              )

fig.add_trace(go.Scatter(x=[dataset['RM'].mean(), dataset['RM'].mean()],
                         y=[dataset['target'].min() - 1, dataset['target'].max() + 1],
                         mode='lines',
                         line=dict(dash='dash',
                                   color='red')
                         )
              )
fig.update_layout(title=go.layout.Title(text="Scatter of Average # of Rooms & Median Value($1000)",
                                        xref="paper",
                                        x=0),
                  xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Average number of rooms per dwelling",
                                                                    # font=dict(family="Courier New, monospace",
                                                                    #           size=18,
                                                                    #           color="#7f7f7f")
                                                                    )
                                        ),
                  yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Median value of owner-occupied homes in $1000's",
                                                                    # font=dict(family="Courier New, monospace",
                                                                    #           size=18,
                                                                    #           color="#7f7f7f")
                                                                    )
                                        )
                  )

fig.show()

import statsmodels.api as sm
import statsmodels.formula.api as smf

y = dataset['target']
X = dataset['RM']
print()
print(f'X = \n{X}')

X = sm.add_constant(X)
print()
print(f'sm.add_constant(X) = \n{sm.add_constant(X)}')

# 선형회귀 계산 초기화
linear_regression = sm.OLS(y, X)

# 회귀계수 b 추정
fitted_model = linear_regression.fit()
print()
print(f'fitted_model = \n{fitted_model.summary()}')

linear_regression_smf = smf.ols(formula='target ~ RM', data=dataset)
fitted_model_smf = linear_regression_smf.fit()
print()
print(f'fitted_model_smf = \n{fitted_model_smf.summary()}')

print()
print(f'fitted_model.params = \n{fitted_model.params}')

betas = np.array(fitted_model.params)
fitted_values = fitted_model.predict(X)

mean_sum_sqaured_errors = np.sum((dataset['target'] - dataset['target'].mean())**2)
reg_sum_squared_errors = np.sum((dataset['target'] - fitted_values)**2)

deviation_squared_errors = (mean_sum_sqaured_errors - reg_sum_squared_errors) / mean_sum_sqaured_errors
print()
print(f'deviation_squared_errors = {deviation_squared_errors:.06}')