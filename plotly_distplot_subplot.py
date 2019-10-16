import chart_studio.plotly as py
import plotly.graph_objs as gobj
import plotly
import plotly.figure_factory as ff
import numpy as np
import plotly.io as pio
from plotly.subplots import make_subplots


pio.renderers.default = 'browser'

x = np.random.randn(1000)
hist_data = [x]
group_labels = ['distplot']

fig = ff.create_distplot(hist_data, group_labels)
# fig.show()

my_fig = make_subplots(rows=1,
                       cols=2,
                       subplot_titles=('California Housing Prices - Number of Bedrooms',
                                       'California Housing Prices - Median Income'))

right_trace = gobj.Scatter(
    y=np.random.randn(500),
    mode='markers',
    marker=dict(
        size=16,
        color=np.random.randn(500), #set color equal to a variable
        colorscale='Viridis',
        showscale=False
    )
)

distplot = fig['data']
print(distplot)

for item in distplot:
    print(item['type'])

for item in distplot:
    print(item.pop('xaxis', None))
    print(item.pop('yaxis', None))