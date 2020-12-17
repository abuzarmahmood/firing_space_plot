import datetime
from dateutil.relativedelta import *
import numpy as np
import pandas as pd
from scipy import stats
#import plotly.graph_objects as go
import plotly.express as px
#from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


data_len = 3000 # we'll generate data for this many MINUTES

# Generate time values
start_time = datetime.datetime.today()
delta_time = datetime.timedelta(minutes = 30)
datetime_vec = start_time + np.arange(data_len) * delta_time 

# Generate temp and himidity data
temp = 40 * stats.norm.rvs(size=data_len) + 60
humid = 20 * stats.norm.rvs(size=data_len) + 40

# Convert to pandas dataframe
frame = pd.DataFrame({
    'date' : datetime_vec,
    'temp' : temp,
    'humidity' : humid})
frame.set_index('date', inplace = True)

# Melt frame to generate data 
#cat_frame = pd.melt(frame,
#            id_vars = 'date',
#            value_vars = ['temp','humidity'],
#            var_name = 'category',
#            value_name = 'data')

# Create plot
#fig = px.line(cat_frame,
#            x = 'date',
#            y = 'data',
#            color = 'category')
#fig.show()

# Create plot
#fig = make_subplots(rows=2, cols=1)
#fig.append_trace(go.Scatter(
#        x=datetime_vec, y= temp), row=1, col=1)
#fig.append_trace(go.Scatter(
#        x=datetime_vec, y= humid), row=2, col=1)
#fig.show()

########################################
# Create Dash server
########################################

app = dash.Dash(__name__)
server = app.server
#window_names = ['1 week','2 week','1 month','3 months','6 months']
window_names = ['7 D','14 D','30 D','90 D','180 D']
window_periods = [x.split(' ') for x in window_names]
#window_periods = [relativedelta(weeks=-1),
#                    relativedelta(weeks=-2),
#                    relativedelta(months=-1),
#                    relativedelta(months=-3),
#                    relativedelta(months=-6)]

# Type relativedelta is not JSON serializable
# Convert each to how many indices you'd have to go back from the present
#window_prev_times = [pd.to_datetime(cat_frame.date.iloc[-1]) + delta \
#        for delta in window_periods]
#window_prev_inds = [np.argmin(np.abs(cat_frame.date - x)) \
#        for x in window_prev_times]

# Apparently you can't do both at the same time without a callback

# Just the graph
#app.layout = html.Div([
#    dcc.Graph(id = 'sensor_data',
#        figure = {
#            'data' : [
#                dict( x = cat_frame.date[cat_frame.category == i], 
#                    y = cat_frame.data[cat_frame.category == i],
#                    name = i) \
#                            for i in cat_frame.category.unique()]}
#            )])

#selected_window = window_periods[0] 
#end_timepoint = frame.last_valid_index()
#start_timepoint = end_timepoint - pd.Timedelta(int(selected_window[0]),selected_window[1])
#wanted_frame = frame.loc[start_timepoint:end_timepoint] 
#dates = pd.to_datetime(wanted_frame.index.values)
#traces = [dict( x = dates,
#                y = wanted_frame.temp, name = 'temp'),
#        dict (x = dates,
#            y = wanted_frame.humidity, name = 'humidity')]
#app.layout = html.Div([
#    dcc.Graph(id = 'sensor_data',
#        figure = {
#            'data' : traces})])

# Just the menu
#app.layout = html.Div([
#    dcc.Dropdown(
#        options=[{'label': i, 'value': j} \
#            for i,j in zip(window_names,window_prev_inds)],
#                    value='1 week')])
#app.layout = html.Div([
#    dcc.Dropdown(
#        options=[{'label': i, 'value': j} \
#            for i,j in zip(window_names,window_periods)],
#                    value='7 D')])

# Plot with callback
app.layout = html.Div([
    dcc.Graph(id = 'sensor_data'),
    dcc.Dropdown(
        id = 'window_select',
        options=[{'label': i, 'value': j} \
            for i,j in zip(window_names,window_periods)],
                    value='7 D')])

@app.callback(
    Output('sensor_data', 'figure'),
    [Input('window_select', 'value')])
def update_graph(selected_window):
    #return px.scatter(all_teams_df[all_teams_df.group == grpname], 
    #        x='min_mid', y='player', size='shots_freq', color='pl_pps')
    #import plotly.express as px
    #return px.line(cat_frame, x = 'date', y = 'data', color = 'category')
    # Find required range
    end_timepoint = frame.last_valid_index()
    start_timepoint = end_timepoint - pd.Timedelta(int(selected_window[0]),selected_window[1])
    wanted_frame = frame.loc[start_timepoint:end_timepoint] 
    dates = pd.to_datetime(wanted_frame.index.values)
    traces = [dict( x = dates,
                    y = wanted_frame.temp, name = 'temp'),
            dict (x = dates,
                y = wanted_frame.humidity, name = 'humidity')]
    return {'data' : traces}

#app.run_server(debug=False)

if __name__ == '__main__':
    app.run_server(debug=False)
