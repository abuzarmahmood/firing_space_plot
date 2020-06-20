import datetime
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash


data_len = 10000 # we'll generate data for this many MINUTES

# Generate time values
start_time = datetime.datetime.today()
delta_time = datetime.timedelta(minutes = 1)
datetime_vec = start_time + np.arange(data_len) * delta_time 

# Generate temp and himidity data
temp = 40 * stats.norm.rvs(size=data_len) + 60
humid = 20 * stats.norm.rvs(size=data_len) + 40

# Convert to pandas dataframe
frame = pd.DataFrame({
    'date' : datetime_vec,
    'temp' : temp,
    'humidity' : humid})

# Melt frame to generate data 
cat_frame = pd.melt(frame,
            id_vars = 'date',
            value_vars = ['temp','humidity'],
            var_name = 'category',
            value_name = 'data')

# Create plot
fig = px.line(cat_frame,
            x = 'date',
            y = 'data',
            color = 'category')
fig.show()



# Create plot
#fig = make_subplots(rows=2, cols=1)
#fig.append_trace(go.Scatter(
#        x=datetime_vec, y= temp), row=1, col=1)
#fig.append_trace(go.Scatter(
#        x=datetime_vec, y= humid), row=2, col=1)
#fig.show()
