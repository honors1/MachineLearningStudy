from sklearn import preprocessing
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = 'browser'

# dataset 가져오기
df = pd.read_csv('./data/california_housing_train.csv', sep=",")
print()
print(df.head())
print(df.info())

# normalize total_bedrooms column
normalized_bedrooms = preprocessing.normalize([np.array(df['total_bedrooms'])])
df['normalized_total_bedrooms'] = normalized_bedrooms[0]
# normalize median_income column
normalized_income = preprocessing.normalize([np.array(df['median_income'])])
df['normalized_median_income'] = normalized_income[0]
print()
print(df.info())


names = df.columns
# StandardScaler(): 평균이 0이고, 표준편차가 1이 되도록 변환
scaler = preprocessing.StandardScaler()

scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)

sub_fig = make_subplots(rows=2,
                        cols=2,
                        subplot_titles=('California Housing Prices - Number of Bedrooms',
                                        'California Housing Prices - Median Income',
                                        'California Housing Prices - Number of Bedrooms (Standardized)',
                                        'California Housing Prices - Median Income (Standardized)'))
left_fig = ff.create_distplot([df['total_bedrooms']],
                              group_labels=['Number of Bedrooms'],
                              curve_type='normal',
                              colors=['blue'])
left_distplot = left_fig['data']

right_fig = ff.create_distplot([df['median_income']],
                               group_labels=['Median Income'],
                               curve_type='normal',
                               colors=['red'])
right_distplot = right_fig['data']

left_standardized = ff.create_distplot([scaled_df['total_bedrooms']],
                                       group_labels=['Number of Bedrooms (Standardized)'],
                                       curve_type='normal',
                                       colors=['darkblue'])

left_standardized_distplot = left_standardized['data']

right_standardized = ff.create_distplot([scaled_df['median_income']],
                                        group_labels=['Median Income (Standardized)'],
                                        curve_type='normal',
                                        colors=['darkred'])

right_standardized_distplot = right_standardized['data']

sub_fig.append_trace(left_distplot[0], 1, 1)
sub_fig.append_trace(left_distplot[1], 1, 1)

sub_fig.append_trace(right_distplot[0], 1, 2)
sub_fig.append_trace(right_distplot[1], 1, 2)

sub_fig.append_trace(left_standardized_distplot[0], 2, 1)
sub_fig.append_trace(left_standardized_distplot[1], 2, 1)

sub_fig.append_trace(right_standardized_distplot[0], 2, 2)
sub_fig.append_trace(right_standardized_distplot[1], 2, 2)

sub_fig.show()
