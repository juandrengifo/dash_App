# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from pyexpat import features
from dash import Dash, html, dcc, State, ctx
import pandas as pd
from plotly import express as px
import requests
import plotly
from dash.dependencies import Input, Output
import dash_daq as daq
from pyproj import Transformer
import plotly.graph_objs as go
import plotly.offline as py
from ipywidgets import interactive, HBox, VBox
import json

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset, Experiment
import dash_bootstrap_components as dbc

import os
#!pip install geopandas
import numpy.ma as ma
import matplotlib.tri as tri
import geopandas as gpd
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon, MultiPolygon
import plotly.offline as py
from plotly.graph_objs import *
import numpy as np           
from scipy.io import netcdf  
from mpl_toolkits.basemap import Basemap



dash_app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app = dash_app.server

mapbox_token = "pk.eyJ1IjoianVhbmRyZW5naWZvIiwiYSI6ImNsM2JjeWNxMjA3bTUzZHM2MDRlMXlzNHoifQ.J5zYv4JEeSzBYtqTHFiotg"
px.set_mapbox_access_token(mapbox_token)


subscription_id = '755b224d-5144-4359-b618-f62b1efb1c57'
resource_group = 'AUAZE-CORP-DEV-EXPLORATIONAI'
workspace_name = 'DevAIMLWorkspace'

workspace = Workspace(subscription_id, resource_group, workspace_name)
experiment = Experiment(workspace, "dummy")

dataset = Dataset.get_by_name(workspace, name='dataset1')
df = dataset.to_pandas_dataframe()
print("done.")
points = list(zip(list(df["longitud(m)"]), list(df["latitude(m)"])))
#print(points)

train_ds, df_sel = pd.DataFrame(), pd.DataFrame()
transformer = Transformer.from_crs(3978, 4326, always_xy=True)
pts = []
for pt in transformer.itransform(points):
    pts.append(pt)

display = False
df["lon"] = [x[0] for x in pts]
df["lat"] = [x[1] for x in pts]
df.drop(['longitud(m)', 'latitude(m)'], axis = 1, inplace = True)




df_train, df_test, df_eval = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
target, features = "group_rocktype", list(df.columns)



fig = px.scatter_mapbox(lat=["50.47358"], lon=["-84.86125"], size_max=20, height=860, width=1300, zoom=5.2, range_color=(0,700))
fig.update_layout(mapbox_style="dark", mapbox_accesstoken=mapbox_token)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(legend=dict( x=0, y=1, bgcolor="rgba(0,0,0, 0)"), font=dict(color="white"))


"""
df_e = pd.read_csv("exp/fer.csv")
points = list(zip([float(x.split(",")[0][1:]) for x in df_e["geometry"]], [float(x.split(",")[1][:-1]) for x in df_e["geometry"]]))
transformer = Transformer.from_crs(3978, 4326, always_xy=True)
pts = []
for pt in transformer.itransform(points):
    pts.append(pt)

display = False
df_e["lat"] = [x[0] for x in pts]
df_e["lon"] = [x[1] for x in pts]
air = df_e["pred_probability"]
lat = df_e["lat"]
lon = df_e["lon"]

lat, lon, air = lat[:1000], lon[:1000], air[:1000]


import plotly.graph_objects as go
fig2 = go.Figure(data =
     go.Contour(x = lon, y = lat, z = air))
"""

"""
fig_e = go.Figure(data =
    go.Contour(
        z=air,
        x=lon, # horizontal axis
        y=lat # vertical axis
    ))

"""


def aux():
    return html.Div(style={"display":"none"})

app.layout = dbc.Container([
    dbc.Row([
        html.Center(html.H1(children='Training pipeline configuration board',
                        className='my-5'))
    ]),

    dbc.Row([
        

        dbc.Col([

            

            html.Center(dcc.Input(id='model_name', placeholder="Name the model", type='text', style={"text-align":"center","width":"70%", "margin-top": "30px"})),
            #html.Button(id='submit-button', type='submit', children='Submit'),
            html.Div(id='output_model_name', style={"display":"none"}),
            dbc.Tooltip(
                [html.P("Type a name for the model.")],
                target="model_name"
            ),

            html.Center(html.Div(dcc.Dropdown(list(workspace.datasets),multi=False,placeholder="Select the dataset",id = "ds_name_"), style={"margin-top":"30px", "width":"70%"}, id="z")),
            #(dcc.Input(id='ds_name_', placeholder="Name of the dataset", type='text', style={"text-align":"center", "width":"20%"})),
            #html.Button(id='submit-button', type='submit', children='Submit'),
            html.Div(id='output_ds_name', style={"display":"none"}),
            dbc.Tooltip(
                [html.P("Select the dataset that is going to be used for training.")],
                target="z"
            ),

            html.Center(dcc.Input(id='ds_version', placeholder="Select the version of the dataset", type='text', style={"text-align":"center","width":"70%", "margin-top": "30px"})),
            #html.Button(id='submit-button', type='submit', children='Submit'),
            html.Div(id='output_ds_version', style={"display":"none"}),
            dbc.Tooltip(
                [html.P("Type the version of the dataset")],
                target="ds_version"
            ),
            

            html.Center(html.Div(dcc.Dropdown(list(df.columns),multi=False,placeholder="Select the prediction target",id = "target"), style={"width":"70%", "margin-top": "30px", "margin-bottom": "10px"}, id="t")),
            html.Div(style={"display":"none"}, id="tar"),
            dbc.Tooltip(
                [html.P("Specify the prediction target.")],
                target="t"
            ),
            
            
            
            
        ]),

        dbc.Col([

            
            #html.Div(id='output_all_feat', style={"display":"none"}),

            html.Center(html.Div(dcc.Dropdown(list(df.columns),multi=True,placeholder="Select the features",id = "features", value=[]), style={"width":"70%", "margin-top": "30px"}, id="f")),
            html.Div(style={"display":"none"}, id="feat"),
            dbc.Tooltip(
                [html.P("Select the features to train the model.")],
                target="f"
            ),

            html.Div(dcc.Checklist(['Include all features'], id="all_feat", inputStyle={"margin-left": "10px"}), style={"margin-left": "85px"}),


            html.Center(html.Div(dcc.Dropdown(["Random", "Spatial", "Manual"],multi=False,placeholder="Select the splitting method",id = "split_type"), style={"width":"70%", "margin-top": "10px"}, id="a")),
            html.Div(style={"display":"none"}, id="output_split_type"),
            dbc.Tooltip(
                [html.P("Random: This method splits the dataset randomly (75% for training and 25% for testing)."),
                html.P("Spatial: This method splits the dataset based on a spatial algorithm."),
                html.P("Manual: This method takes the selected points on the map and use the rest for testing.")],
                target="a"
            ),


            html.Center(html.Div(dcc.Dropdown(["Exploratory", "Common", "Extreme"],multi=False,placeholder="Select the type of training (training time)",id = "time_left"), style={"width":"70%", "margin-top": "30px"}, id="b")),
            html.Div(style={"display":"none"}, id="output_time_left"),
            dbc.Tooltip(
                [html.P("Exploratory: This option searches the best model for 1 minute."),
                html.P("Common: This option searches the best model for 30 minutes."),
                html.P("Manual: This option searches the best model for 60 minutes.")],
                target="b"
            ),

            html.Center(html.Div(html.Button(id='comp_ds', n_clicks=0, children='Train', className='btn-outline-dark'), style={"width":"60%", "margin-top": "35px"})),
            html.Center(html.Div(html.Button(id='config_file', n_clicks=0, children='Train', className='btn-outline-dark'), style={ "display":"none"}))

            
            #daq.ToggleSwitch(
            #id='new_ver',
            #value=False,
            #label='Create new version of the dataset',
            #labelPosition='bottom',
            #className="my-3"
            #),
            #html.Div(id='output_new_ver', style={"display":"none"}),
            #daq.ToggleSwitch(
            #    id='interpretable',
            #    value=False,
            #    label='Compute most relevant features',
            #    labelPosition='bottom',
            #    className="my-3"
            #),
            #html.Div(id='output_interpretable', style={"display":"none"}),

            

            

            

            
        ]),

        

        
        
        

        

        
        

    ]),

    dbc.Row([
        html.Div(html.P("Select the testing points on the map"), id="msg", style={"display":"none"}),
        html.Center(html.Div(dcc.Graph(id='map', figure=fig), style={"margin-top":"30px"})),
        html.Div(id='display-selected-values', style={"display":"none", "margin-up":"30px"}),
        

    ]),

    #dbc.Row([
    #    dcc.Graph(id='map2', figure=fig2),
    #    html.H1("HOLAAA")
    #])

    
    
])

"""
app.layout = html.Div(children=[
    html.H1(children='Points selection'),

    dcc2323<<.Graph(id='map',figure=fig),
    html.Div(id='display-selected-values', style={"display":"none"}),

    html.Div(children='''Select the points that are going to be used on training.'''),
    html.Button(id='training_submit', n_clicks=0, children='Use for training'),
    html.Div(id='train_sbmt', style={"display":"none"}),

    html.Div(children='''Select the points that are going to be used on testing.'''),
    html.Button(id='test_submit', n_clicks=0, children='Use for testing'),
    html.Div(id='test_sbmt', style={"display":"none"}),

    html.Div(children='''Select the points that are going to be used on evaluation.'''),
    html.Button(id='eval_submit', n_clicks=0, children='Use for evaluation'),
    html.Div(id='eval_sbmt', style={"display":"none"})
    
    
])
"""






@app.callback(
    Output('config_file', 'children'),
    Input('comp_ds', 'n_clicks'),
    [State('tar', 'children'),
    State('feat', 'children'),
    State('output_ds_name', 'children'),
    #State('output_ds_version', 'children'),
    State('output_model_name', 'children'),
    State('output_split_type', 'children'),
    State("output_time_left", "children"),
    State("output_ds_version", "children"),
    State("display-selected-values", "children")
    ]
)
def compile_dataset(n_clicks, tar, feat, ds_name, model_name, split_type, output_time_left, ds_version, selection):
    global df, target, features, display, train_ds, df_sel, df_test
    
    config = dict()
    display = True
    if tar!=None and feat!=None and ds_name!=None  and model_name!=None and split_type!=None: #ds_version!=None
        target, features = tar, feat


        df.to_csv("points.csv")
        config["ds_name"] = ds_name
        config["ds_version"] = ds_version
        config["experiment"] = model_name
        config["model_name"] = model_name
        config["create_new_version"] = "True"
        config["target"] = target
        if "Random" == split_type: config["spliting_function"] = "random"
        elif split_type=="Manual": config["spliting_function"] = "manual"
        elif split_type=="Spatial": config["spliting_function"] = "spatial"
        for i in range(len(feat)):
            if feat[i]=="lat":
                feat[i]="latitude(m)"
            elif feat[i]=="lon":
                feat[i]="longitud(m)"
        """
        feat_ = []
        for f in feat:
            if f != target and f!="latitude(m)" and f!="longitud(m)":
                feat_.append(f)
        """
        config["feat_filter"] = {"method": "select", "features":feat}
        if output_time_left == "Exploratory": config["time_left_for_this_task"] = 60
        elif output_time_left == "Common": config["time_left_for_this_task"] = 1800
        elif output_time_left == "Extreme": config["time_left_for_this_task"] = 3600
        config["metric"] = "balanced_accuracy"
        config["interpretable"] = 1
        config['model_specific']= 'lightgbm'

    with open('configs/prepro_train/default.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    if split_type=="Manual" and selection!=[]:
        sel = json.loads(selection)
        idxs = []

        for p in sel["points"]:
            idxs.append((p["lon"], p["lat"]))
        df["coords"] = list(zip(df["lon"], df["lat"]))
        df_sel = df['coords'].isin(idxs)
        df_test=df[df_sel]
        print("TEST DATASET")
        print(df_test)
        df_test.to_csv("data/test.csv")
        df_train = df[~df_sel]
        df_train.to_csv("data/train.csv")
    

    #import subprocess
    if tar!=None and feat!=None and ds_name!=None  and model_name!=None and split_type!=None: #ds_version!=None
        #subprocess.call("prepro_train.py", shell=True)
        import os
        os.system("python prepro_train_tree.py")
    return json.dumps(config, indent=2)

@app.callback(
    Output('output_ds_name', 'children'),
    Input('ds_name_', 'value')
)
def ds_name(value):
    global df
    if value!=None:
        dataset = Dataset.get_by_name(workspace, name=value)
        df = dataset.to_pandas_dataframe()
        points = list(zip(list(df["longitud(m)"]), list(df["latitude(m)"])))
#print(points)
        transformer = Transformer.from_crs(3978, 4326, always_xy=True)
        pts = []
        for pt in transformer.itransform(points):
            pts.append(pt)

        df["lon"] = [x[0] for x in pts]
        df["lat"] = [x[1] for x in pts]
    return value
"""
@app.callback(
    Output('output_ds_version', 'children'),
    Input('ds_version', 'value')
)
def ds_version(value):
    return value
"""

@app.callback(
    Output('tar', 'children'),
    Input('target', 'value')
)
def tarrr(value):
    return value

@app.callback(
    Output('feat', 'children'),
    Input('features', 'value')
)
def tarrr(value):
    return value

@app.callback(
    Output('output_model_name', 'children'),
    Input('model_name', 'value')
)
def model_name(value):
    return value

@app.callback(
    Output('output_time_left', 'children'),
    Input('time_left', 'value')
)
def time_left(value):
    return value

@app.callback(
    Output('output_ds_version', 'children'),
    Input('ds_version', 'value')
)
def ds_version(value):
    return value

@app.callback(
    Output('output_split_type', 'children'),
    Input('split_type', 'value')
)
def split_type(value):
    return value

@app.callback(
    Output('features', 'value'),
    [
        Input('all_feat', 'value'),
        Input('target', 'value')
    ]
)
def all_feat(value, tar):
    global df
    ans = []
    button_id = ctx.triggered_id if not None else 'No clicks yet'

    if button_id=="target" and tar!="": return  ["lat", "lon", tar] 
    if button_id=="all_feat" and value!=None and value!=[] and value!="": return list(df.columns)
    else: return []

    """
    if value!=None: return list(df.columns)
    if tar==None or tar==[]: return []

    return  ["lat", "lon", tar] 
    """
    """
@app.callback(
    Output('output_new_ver', 'children'),
    Input('new_ver', 'value')
)
def new_ver(value):
    return value

@app.callback(
    Output('output_interpretable', 'children'),
    Input('interpretable', 'value')
)
def interpretable(value):
    return value"""

@app.callback(
    Output('display-selected-values', 'children'),
    [Input('map', 'selectedData')])
def select(selectedData):
    # Similarly for data when you select a region
    #print(selectedData)
    return json.dumps(selectedData, indent=2)


@app.callback(
    Output('map', 'figure'),
    Output('msg', 'style'),
    [Input('output_split_type', 'children'),
    Input('tar', 'children')]
)
def update_map(output_range, target):
    global df
    disp = {"display":"none"}
    fig = px.scatter_mapbox(lat=["50.47358"], lon=["-84.86125"], size_max=20, height=860, width=1300, zoom=5.2, range_color=(0,700))
    if target!=None and output_range!=None:
        disp = None
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", size_max=20, height=860, width=1300, zoom=5.2, range_color=(0,700), color=target)
    fig.update_layout(mapbox_style="dark", mapbox_accesstoken=mapbox_token)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(legend=dict( x=0, y=1, bgcolor="rgba(0,0,0, 0)"), font=dict(color="white"))
    """
    fig.add_trace(go.Contour(
        z=air,
        x=lon, # horizontal axis
        y=lat # vertical axis
    ))
    """

    return fig, disp






if __name__ == '__main__':
    app.run_server(debug=True)
