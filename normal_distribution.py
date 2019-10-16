import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

import plotly.figure_factory as ff
import plotly
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'


X = (np.arange(9, dtype=np.float) - 3).reshape(-1, 1)
print(f'X = \n{X}')

X = np.vstack([X, [100]])
print()
print(f'np.vstack([X, [100]] = \n{X}')

X = pd.DataFrame(X)
print()
print(f'pd.DataFrame(X) = \n{X}')

print()
print(f'X.describe() = \n{X.describe()}')

# StandardScaler(): 평균이 0이고, 표준편차가 1이 되도록 변환
# RobustScaler(X): 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환
# MinMaxScaler(X): 최대값이 1, 최소값이 0이 되도록 변환
# MaxAbsScaler(X): 0을 기준으로 절대값이 가장 큰 수가 1 또는 -1이 되도록 변환
scaler = StandardScaler()

# 학습용 데이터의 분포 추정: 학습용 데이터를 입력으로 하여 fit 메서드를 실행하면 분포 모수를 객체에 저장
scaler.fit(X)
print()
print(f'scaler.fit(X) = \n{scaler.fit(X)}')
# 학습용 데이터 변환: 학습용 데이터를 입력받아 transform 메서드를 실행하면 학습용 데이터를 변환
X_scaled = scaler.transform(X)
print()
print(f'X_scaled = \n{X_scaled}')

m = np.mean(X_scaled)
print()
print(f'np.mean(X_scaled) = {m}')

s = np.std(X_scaled)
print()
print(f'np.std(X_scaled) = {s}')

X_scaled_pd = pd.DataFrame(X_scaled)
print()
print(f'X_scaled_pd = \n{X_scaled_pd}')

print()
print(f'X_scaled_pd.describe() = \n{X_scaled_pd.describe()}')


# 정규분포 곡선 그리기
x = np.linspace(-6, 6, 1000000)

# 확률밀도함수(PDF) 값을 이용하여 다른 평균과 표준편차의 데이터를 만든다.
x1 = norm.pdf(x, 0, 0.7)
x2 = norm.pdf(x, 0, 1)
x3 = norm.pdf(x, 1, 1.5)
x4 = norm.pdf(x, -2, 0.5)

# graph는 plotly를 이용하여 그린다.
fig = go.Figure()

fig.add_trace(go.Scatter(x=x,
                         y=x1,
                         mode='lines',
                         name='(mean=0, std=0.7)'))
fig.add_trace(go.Scatter(x=x,
                         y=x2,
                         mode='lines',
                         name='(mean=0, std=1)'))
fig.add_trace(go.Scatter(x=x,
                         y=x3,
                         mode='lines',
                         name='(mean=1, std=1.5)'))
fig.add_trace(go.Scatter(x=x,
                         y=x4,
                         mode='lines',
                         name='(mean=-1, std=0.5)'))
fig.show()