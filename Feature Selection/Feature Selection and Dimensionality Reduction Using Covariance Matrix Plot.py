# https://medium.com/towards-artificial-intelligence/feature-selection-and-dimensionality-reduction-using-covariance-matrix-plot-b4c7498abd07
# https://github.com/bot13956/ML_Model_for_Predicting_Ships_Crew_Size/blob/master/Ship_Crew_Size_ML_Model.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

pio.renderers.default = 'browser'

df = pd.read_csv('./cruise_ship_info.csv')

print(df.head())

# 데이터 정보 확인
print(f'\n\n데이터 정보')
print('===========================================================')
print(df.info())

# 데이터 column 정보
print(f'\n\n데이터 column')
print('===========================================================')
print(df.columns)


# 데이터 통계량 확인
print(f'\n\ncruise ship info 데이터 통계량')
print('===========================================================')
print(df.describe())

# 필요한 feature의 columns만 선택
cols = ['Age', 'Tonnage', 'passengers', 'length', 'cabins', 'passenger_density', 'crew']

# 상관관계 그래프로 feature 간의 관계 확인

fig = px.scatter_matrix(data_frame=df,
                        dimensions=cols
                        )
fig.update_layout(title='cruise ship info 데이터의 feature 간의 상관 그래프',
                  dragmode='select',
                  width=900,
                  height=900,
                  hovermode='x')
fig.show()

# 'crew' 변수를 기준으로 'passenger_density' 변수를 제외한 나머지 4개의 변수와
# 상관관계가 있는 것으로 보여, 상관계수를 계산해본다.

stdsc = StandardScaler()
X_std = stdsc.fit_transform(df[cols].iloc[:, range(0, 7)].values)

print()
print(f'표준정규화된 데이터')
print('===================================================')
print(X_std)

# 공분산 행렬로 변환
cov_mat = np.round(np.cov(X_std.T), 4)
print()
print('공분산 행렬')
print('===================================================')
print(cov_mat)

# fig = go.Figure(data=go.Heatmap(z=cov_mat,
#                                 x=cols,
#                                 y=cols))

fig = ff.create_annotated_heatmap(z=cov_mat,
                                  x=cols,
                                  y=cols,
                                  showscale=True)
fig.update_layout(title='cruise ship info 데이터 feature 간의 상관계수를 보여주는 공분산 행렬',
                  dragmode='select',
                  width=900,
                  height=900,
                  hovermode='x')

fig.show()

# 상관계수가 큰 feature만 모은다.
cols_selected = ['Tonnage', 'passengers', 'length', 'cabins', 'crew']

print()
print('상관이 있는 feature 데이터만 보자')
print('==================================================================')
print(df[cols_selected])