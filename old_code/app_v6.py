"""
This app applies Bootstrap themes to Dash components and Plotly figures by
using the stylesheet, figure templates and theme change component
from the dash-bootstrap-templates library: https://github.com/AnnMarieW/dash-bootstrap-templates

`className="dbc"`:
- Makes the text readable in both light and dark themes.
- Uses the font from the Bootstrap theme's font-family.
- Changes the accent color to the theme's primary color

The figure templates applies Bootstrap themes to Plotly figures.  These figure
templates are included in the theme change component.
"""


import os
import sys

# path_aibedo = '/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/11_07_22/aibedo/'
# sys.path.append(path_aibedo)
# # os.chdir(path_aibedo)

# import xarray as xr
# import numpy as np
# from typing import *
# import wandb
# import torch
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import proplot as pplt
# from aibedo.models import BaseModel
# from aibedo.utilities.wandb_api import reload_checkpoint_from_wandb, get_run_ids_for_hyperparams
# import scipy.stats
# import matplotlib.patches as patches
# import hydra
# from hydra.core.global_hydra import GlobalHydra
# from omegaconf import OmegaConf, DictConfig
# from aibedo.utilities.config_utils import get_config_from_hydra_compose_overrides
# from aibedo.utilities.utils import rsetattr, get_logger, get_local_ckpt_path, rhasattr, rgetattr

#print(sys.path)


from dash import Dash, dcc, html, dash_table, Input, Output, callback, State
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url

import numpy as np
import pandas as pd

import xarray as xr
import torch
from pathlib import Path
import re
import time


#NEW DATA LOADING
DATA_DIR = '/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/aibedo_viz/haruki_notebook_10_27_22/LE_CESM2_data/'
# the data used for prediction must be here, as well as the cmip6 mean/std statistics
# Input data filename (isosph is an order 6 icosahedron, isosph5 of order 5, etc.)
filename_input = "isosph5.CESM2-LE.historical.r11i1p1f1.Input.Exp8.nc"
# Output data filename is inferred from the input filename, do not edit!
# E.g.: "compress.isosph.CESM2.historical.r1i1p1f1.Output.nc"
filename_output = filename_input.replace("Input.Exp8.nc", "Output.nc")

ds_input = xr.open_dataset(f"{DATA_DIR}/{filename_input}")  # Input data
ds_output = xr.open_dataset(f"{DATA_DIR}/{filename_output}") # Ground truth data


# df = px.data.gapminder()

zonal_df = pd.read_pickle("/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/11_07_22/aibedoviz/updated_zonal_df.pkl")
location_df = pd.read_pickle("/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/11_07_22/aibedoviz/location_df.pkl")
prediction_df = pd.read_pickle("/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/11_07_22/aibedoviz/model_prediction_df.pkl")
preturbation_df = pd.read_pickle("/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/11_07_22/aibedoviz/input_preturbation_df.pkl")
#ds = xr.open_dataset('../temp_data/Exp1_CESM2_r1i1p1f1_historical_Output_level4.nc')
input_ds = xr.open_dataset('/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/aws_s3_data/L4/CESM2_level4_input.nc')
output_ds = xr.open_dataset('/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/aws_s3_data/L4/CESM2_level4_output.nc')
timestep = 1000
TOTAL_NUM_POINT = location_df.shape[0]

in_vars = [ 'crelSurf_pre', 'crel_pre', 'cresSurf_pre', 'cres_pre', 'netTOAcs_pre', 'lsMask', 'netSurfcs_pre']
out_vars = ['tas_pre', 'psl_pre', 'pr_pre']

new_in_vars = [ 'crelSurf_nonorm', 'crel_nonorm', 'cresSurf_nonorm', 'cres_nonorm', 'netTOAcs_nonorm', 'lsMask', 'netSurfcs_nonorm']
new_out_vars = ['tas_nonorm', 'psl_nonorm', 'pr_nonorm']


temporal_df = pd.DataFrame()
temporal_df['timestep'] = np.arange(1980,)
for v in [ 'crelSurf_pre', 'crel_pre', 'cresSurf_pre', 'cres_pre', 'netTOAcs_pre', 'netSurfcs_pre']:
    temporal_df[v] = np.mean(input_ds[v].data, axis=1)

#Load the trained SU-net model
unet = torch.load('/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/aws_s3_data/L4/unet_level4_model.pt', map_location=torch.device('cpu'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = unet.module.to(device)

    
# map timestep to mm-yyyy format

START_YEAR = 1850
NUM_YEARS = 165
ts_array = np.arange(1980)
ts_array = ts_array.reshape((165,12))

def ts_2_mmyyyy(ts):
    mm_yyyy = np.where(ts_array == ts)

    yyyy = mm_yyyy[0][0]
    mm = mm_yyyy[1][0]
    year = START_YEAR + yyyy
    month = mm  + 1
    return month, year

def mmyyyy_2_ts(month, year):
    yyyy = year - START_YEAR
    mm = month - 1
    print(ts_array[yyyy][mm])

timestring_list = []
for i in range(1980):
    mm, yyyy = ts_2_mmyyyy(i)
    timestring = f"{mm:02}" + "-" + str(yyyy)
    timestring_list.append(timestring)


varname_dict = {
                    'crelSurf_pre': 'Longwave surface radiative effect', 
                    'crel_pre': 'Longwave TOA radiative effect', 
                    'cresSurf_pre': 'Shortwave surface radiative effect',
                    'cres_pre': 'Shortwave TOA radiative effect',
                    'netTOAcs_pre': 'Net clearsky TOA radiation',
                    'netSurfcs_pre': 'Net clearsky Surface radiation',
                    'lsMask': 'land mask',
                    'pr_pre': 'Precipitation',
                    'psl_pre': 'Sea level pressure',
                    'tas_pre': '2-metre air temperature'
                }

# stylesheet with the .dbc class
dbc_css = (
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"
)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
app.title = "AiBEDO"

header = html.H1(
    "AiBEDO", className="bg-dark text-white p-2 mb-2 text-center"
)
description = html.H5(
    "A hybrid AI framework to capture the effects of cloud properties on global circulation and regional climate patterns", className="bg-secondary text-black p-2 mb-2 text-center"
)

switch_input_preturb_view = daq.BooleanSwitch(
                                #color='k',
                                id="input_preturbation_switch",
                                # label='Preturbation view',
                                # labelPosition='left',
                                # size=50,
                                on=False,
                            )
switch_output_predict_view = daq.BooleanSwitch(
                                    #color='k',
                                    id="out_switch",
                                    # label='GT --- Pred',
                                    # labelPosition='bottom',
                                    # size=50,
                                    on=True,
                                )
switch_whatif = daq.BooleanSwitch(
                    #color='k',
                    id="whatif_analysis_switch",
                    # label='on/off',
                    # labelPosition='top',
                    # size=50,
                    on=False,
                )

dropdown_in_var = dcc.Dropdown(
                        id="in_varname",
                        options=[
                            {"label": str(i), "value": i} for i in [ 'crelSurf_pre', 'crel_pre', 'cresSurf_pre', 'cres_pre', 'netTOAcs_pre', 'lsMask', 'netSurfcs_pre']
                        ],
                        value="crelSurf_pre",
                        clearable=False,
                    )

dropdown_out_var =  dcc.Dropdown(
                        id="out_varname",
                        options=[
                            {"label": str(i), "value": i} for i in ['pr_pre', 'psl_pre', 'tas_pre']
                        ],
                        value="pr_pre",
                        clearable=False,
                    )
dropdown_timestep = dcc.Dropdown(
                        id="time_step",
                        options=[
                            {"label": timestring_list[i], "value": i} for i in range(1980)
                        ],
                        value=416,
                        clearable=False,
                    )
dropdown_projection = dcc.Dropdown(
                        id="input_projection",
                        options=[
                            {"label": str(i), "value": i} for i in ['equirectangular', 'mercator', 'orthographic', 'natural earth', 'kavrayskiy7', 'miller', 'robinson', 'eckert4', 'azimuthal equal area', 'azimuthal equidistant', 'conic equal area', 'conic conformal', 'conic equidistant', 'gnomonic', 'stereographic', 'mollweide', 'hammer', 'transverse mercator', 'albers usa', 'winkel tripel', 'aitoff','sinusoidal']
                        ],
                        value="orthographic",
                        clearable=False,
                    )
# dropdown_zoneselect = dcc.Dropdown(
#                         id="whatif_zone",
#                         options=[
#                             {"label": str(i), "value": i} for i in ['Arctic', 'N_Midlatitues', 'Tropics', 'S_Midlatitudes', 'Antarctic']
#                         ],
#                         value="S_Midlatitudes",
#                         clearable=False,
#                     )
dropdown_zoneselect = dcc.Dropdown(
                        id="whatif_zone",
                        options=[
                            {"label": str(i), "value": i} for i in ['NEP', 'SP', 'NP', 'SEA', 'SEP']
                        ],
                        value="SEP",
                        clearable=False,
                    )
dropdown_preturbvar = dcc.Dropdown(
                        id="preturb_var",
                        options=[
                            {"label": str(i), "value": i} for i in [ 'crelSurf_pre', 'crel_pre', 'cresSurf_pre', 'cres_pre', 'netTOAcs_pre', 'lsMask', 'netSurfcs_pre']
                        ],
                        value="crelSurf_pre",
                        clearable=False,
                    )
buttongroup_whatif = dbc.ButtonGroup(
                        [
                            dbc.Button("Initialize", id="init_preturbation", color="Primary",  size="sm"),
                            dbc.Button("Update", id="update_preturbation", color="warning",  size="sm"),
                            dbc.Button("Reset", id="clear_preturbation", color="success",  size="sm"),
                        ]
                    )
knob_whatif = daq.Knob(
                #label="Perturbation Sigma",
                id="preturb_knob",
                size=70,
                max=5,
                value=0.5,
                #color={"gradient":True,"ranges":{"green":[0,50],"yellow":[50,90],"red":[90,100]}},
                #labelPosition='bottom',
            )






control_panel1 = html.Div(
    [
        html.H5("General Controls"),
        # html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.H6("Timestep"), md=5),
                dbc.Col(dropdown_timestep,md=7),
            ],
            align="center",
            className="g-0 bg-light",
        ),
        # html.Br(),
        dbc.Row(
            [
                dbc.Col(html.H6("Projection"), md=5),
                dbc.Col(dropdown_projection,md=7),
            ],
            align="center",
            className="g-0 bg-light",
        ),
        html.Hr(),
        html.H5("Model Controls"),
        
        
        dbc.Button("Run AiBEDO", id="run_model", color="primary", size="sm", outline=True,className="me-2"),
        dbc.Button("Clear Data", id="clear_prediction", color="secondary", size="sm", outline=True,className="me-2"),
        
        #dbc.Button("Dark", color="dark", size="sm", outline=True),
        
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.H5("What-If Control"), md=8),
                dbc.Col(switch_whatif, md=4),
            ]
        ),
        # html.Hr(),
        
        dbc.Row(
            [
                dbc.Col(html.H6("Select Zone"), md=5),
                dbc.Col(dropdown_zoneselect,md=7),
            ],
            align="center",
            className="g-0 bg-light",
        ),
        dbc.Row(
            [
                dbc.Col(html.H6("Variable"), md=5),
                dbc.Col(dropdown_preturbvar,md=7),
            ],
            align="center",
            className="g-0 bg-light",
        ),
        
        
        html.Br(),
        dbc.Row(dbc.Col(buttongroup_whatif)),
        dbc.Row(
            [
                dbc.Col(dbc.Label(id='current-sigma'), md=3),
                dbc.Col(knob_whatif,md=9),
            ],
            align="center",
            className="g-0",
        ),
        
    ],
    className="mb-2",
)


control_panel2 = html.Div(
    [
        html.H6("Activity Log"),
        html.Hr(),
        dbc.Label(id='test-model-control'),
        dbc.Label(id='test-cleardata'),
        dbc.Label(id='test-preturb-controls-init'),
        dbc.Label(id='test-preturb-controls-update'),
        dbc.Label(id='test-preturb-controls-reset'),
        
    ],
    className="mb-1 p-1",
)



# checklist = html.Div(
#     [
#         dbc.Label("Select Continents"),
#         dbc.Checklist(
#             id="continents",
#             options=[{"label": i, "value": i} for i in continents],
#             value=continents[2:],
#             inline=True,
#         ),
#         dcc.RadioItems(
#                 ['Linear', 'Log'],
#                 'Linear',
#                 id='xaxis-type',
#                 inline=True
#         ),
#     ],
#     className="mb-4",
# )





#controls = dbc.Card([time_control, input_control, model_control, output_control, mcb_control], body=True,)
control1 = dbc.Card([control_panel1], body=True,)
control2 = dbc.Card([control_panel2], body=True,)


c1 = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(html.H6("Variable"), md=2),
                dbc.Col(dropdown_in_var, md=3),
                dbc.Col(html.H6(""), md=3),
                dbc.Col(html.H6("Perturbation"), md=3),
                dbc.Col(switch_input_preturb_view,md=1),
            ],
            align="center",
            className="g-0 bg-light",
        ),
        dbc.Row(dbc.Col(dcc.Graph(id="input_panel"))),
        html.Br(),
    ],
    body=True
    )
c2 = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(html.H6("Variable"), md=2),
                dbc.Col(dropdown_out_var, md=3),
                dbc.Col(html.H6(""), md=3),
                dbc.Col(html.H6("Model Prediction"), md=3),
                dbc.Col(switch_output_predict_view,md=1),
            ],
            align="center",
            className="g-0 bg-light",
        ),
        dbc.Row(dbc.Col(dcc.Graph(id="output_panel"))),
        html.Br(),
    ], 
    body=True,
    )
c3 = dbc.Card([dcc.Graph(id="temporal_plot_panel")], body=True)
c4 = dbc.Card([dcc.Graph(id="input_zone_panel"),html.Br(),html.Br(),], body=True)
c5 = dbc.Card([dcc.Graph(id="error_stat_panel"),html.Br(),html.Br(),], body=True)
# c6 = dbc.Card([dcc.Graph(id="pcp_panel")], body=True)

tabs1 = dbc.Tabs(
    [
        dbc.Tab(c1, label='Input View'),
        dbc.Tab(c4, label="Zonal View"),
    ]
)

tabs2 = dbc.Tabs(
    [
        dbc.Tab(c2, label='Output View'),
        dbc.Tab(c5, label="Error Statitics"),
    ]
)

tabs3 = dbc.Tabs(
    [
        # dbc.Tab(c6, label='Multivariate Analysis'),
        dbc.Tab(c3, label="Temporal Trends"),
    ]
)

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(header)),
        dbc.Row(dbc.Col(description)),
        # dbc.Row(
        #     [
        #         dbc.Col([controls, ThemeChangerAIO(aio_id="theme", radio_props={"value":dbc.themes.MATERIA})], width=2),
        #         dbc.Col([c1], width=5),
        #         dbc.Col([c2], width=5),
        #         dbc.Col([c3], width=10),
        #     ]
        # ),
        dbc.Row(
            [
                dbc.Col([control1], width=2),
                dbc.Col([tabs1], width=5),
                dbc.Col([tabs2], width=5),
                
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col([control2, ThemeChangerAIO(aio_id="theme", radio_props={"value":dbc.themes.MATERIA})], width=2),
                dbc.Col([tabs3], width=10),
            ]
        ),
    ],
    fluid=True,
    className="dbc",
)


@callback(
    Output("input_panel", "figure"),
    Input("input_projection", "value"),
    Input("in_varname", "value"),
    Input("time_step", "value"),
    Input("input_preturbation_switch", "on"),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)
def update_input_panel(proj, varname, ts, preturb_switch, theme):
    if preturb_switch:
        local_df = preturbation_df
    else:
        # local_df = location_df.copy()
        # local_df[varname] = input_ds[varname][ts].data
        ## Create a df from a dataset on demain: important for dash application
        print("update_input_panel",ts)
        resolution = 2562
        timeslice = slice("1900-01-01","1900-01-01" )
        varname = 'netSurfcs_nonorm'

        lon_list = ds_input.lon.data[:resolution]
        lat_list = ds_input.lat.data[:resolution]
        marker_size_array = np.full((lat_list.shape[0]), 1.)
        vardata_array = ds_input.sel(time=timeslice)[varname][0][:resolution].data
        column_name = ['lat', 'lon', 'm_size', varname]

        local_df = pd.DataFrame(data = np.vstack((lat_list, lon_list, marker_size_array, vardata_array)).T, 
                        columns = column_name)
        
    
    fig = px.scatter_geo(local_df, lat="lat", lon="lon",
                     color=varname, # which column to use to set the color of markers
                     #hover_name="val1", # column added to hover information
                     size="m_size",
                     size_max=4, # size of markers
                     projection=proj,
                     color_continuous_scale='Turbo',
                     basemap_visible=True,
                     template=template_from_url(theme))
    
    fig.update_layout(title_text=varname , title_x=0.45)
    

    return fig

@callback(
    Output("input_zone_panel", "figure"),
    Input("input_projection", "value"),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)
def update_zonal_view(proj, theme):
    q1 = 'Zones != "Others"'
    q2 = 'MCB_site != "Others"'
    zonal_dff = zonal_df.query(q2)
    
    fig = px.scatter_geo(zonal_dff, lat="lat", lon="lon",
                     color='MCB_site', # which column to use to set the color of markers
                     #hover_name="val1", # column added to hover information
                     size="m_size",
                     size_max=4, # size of markers
                     projection=proj,
                     color_continuous_scale='viridis',
                     basemap_visible=True,
                     template=template_from_url(theme))
    
    fig.update_layout(legend=dict(
        orientation="v",
    ))

    return fig

@callback(
    Output("output_panel", "figure"),
    Input("input_projection", "value"),
    Input("out_varname", "value"),
    Input("time_step", "value"),
    Input("out_switch", "on"),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)
def update_output_panel(proj, varname, ts, show_pred, theme):
    
    if show_pred:
        local_df = prediction_df
    else:
        local_df = location_df.copy()
        local_df[varname] = output_ds[varname][ts].data
    
    #print(show_pred)
    
    fig2 = px.scatter_geo(local_df, lat="lat", lon="lon",
                     color=varname, # which column to use to set the color of markers
                     #hover_name="val1", # column added to hover information
                     size="m_size",
                     size_max=5, # size of markers
                     projection=proj,
                     color_continuous_scale='RdBu',
                     basemap_visible=True,
                     template=template_from_url(theme))
    
    # fig2.update_layout(legend=dict(
    #     orientation="v",
    # ))
    fig2.update_layout(title_text=varname_dict[varname], title_x=0.45)
    return fig2

@callback(
    Output('test-model-control', 'children'),
    Input('run_model', 'n_clicks'),
    State('time_step', 'value'),
    State('whatif_analysis_switch', 'on')
)
def update_model_control(n_click, ts, whatif):
    if n_click:
        if whatif:
            #if this is with whatif turned on: used the preturbed df to generate predictions
            model_input = preturbation_df[in_vars].to_numpy()
        else:
            model_input = []
            for var in in_vars:
                model_input.append(input_ds[var][ts].data)
            model_input = np.array(model_input).T
            
        model_input = np.expand_dims(model_input, axis=0)
            
        preds = unet(torch.Tensor(model_input)).detach().numpy()
        preds = np.squeeze(preds, axis=0).T
        
        for i,v in enumerate(out_vars):
            prediction_df[v] = preds[i]
            
            
        
        # prediction_df['pr_pre'] = np.full((TOTAL_NUM_POINT,), 10.0)
        # prediction_df['psl_pre'] = np.full((TOTAL_NUM_POINT,), 12.0)
        # prediction_df['tas_pre'] = np.full((TOTAL_NUM_POINT,), 13.2)
        return 'Model ran at time ' + str(ts)
    else:
        return 'Model has not been run yet! '
    
@callback(
    Output('test-cleardata', 'children'),
    Input('clear_prediction', 'n_clicks'),
    State('time_step', 'value'),
)
def cleardata_button_callback(n, ts):
    if n:
        # print("button clicked")
        prediction_df['pr_pre'] = np.zeros((TOTAL_NUM_POINT,))
        prediction_df['psl_pre'] = np.zeros((TOTAL_NUM_POINT,))
        prediction_df['tas_pre'] = np.zeros((TOTAL_NUM_POINT,))
        return 'Data cleared at time ' + str(ts)
    else:
        return 'Data has not been cleared yet! '





# @callback(
#     Output("temporal_plot_panel", "figure"),
#     Input("time_step", "value"),
#     Input(ThemeChangerAIO.ids.radio("theme"), "value"),
# )
# def update_temporal_panel(ts, theme):
#     #calculate range for range slider
#     half_rw = 50
#     left = ts - half_rw if (ts - half_rw) > 0 else 0  
#     right = ts + half_rw if (ts + half_rw) < 1980 else 1979
    
    
#     fig = px.line(temporal_df, x="timestep", y=temporal_df.columns,
#               template=template_from_url(theme))
#     #fig.add_vline(x=ts, line_width=3, line_dash="dash", line_color="green")
#     fig.update_layout(title_text='Avg input variables across time', title_x=0.45)
#     fig.update_xaxes(autorange=False, range=[left,right], rangeslider_visible=True, rangeslider_thickness=0.1, rangeslider_bgcolor="#ccffff", rangeslider_yaxis=dict())
#     return fig


# @callback(
#     Output("pcp_panel", "figure"),
#     Input("time_step", "value"),
#     Input(ThemeChangerAIO.ids.radio("theme"), "value"),
# )
# def update_pcp_panel(ts, theme):
    
#     pcp_df = zonal_df.copy()
#     for v in out_vars:
#         pcp_df[v] = output_ds[v[:-4]][ts].data
        
    
#     cat_vals = ['Antarctic', 'S_Midlatitudes', 'Tropics', 'N_Midlatitudes', 'Arctic']
#     value2dummy = dict(zip(cat_vals, range(len(cat_vals))))
#     pcp_df['Zones'] = [value2dummy[v] for v in pcp_df['Zones']]

#     col_dict = dict(
#         label='Zones',
#         tickvals=list(value2dummy.values()),
#         ticktext=list(value2dummy.keys()),
#         values=pcp_df['Zones'],
#     )
#     col_to_show = out_vars.copy()
#     col_list = []
#     col_list.append(col_dict)
#     for col in col_to_show:
#         col_dict = dict(
#             range=(pcp_df[col].min(), pcp_df[col].max()),
#             label=varname_dict[col],
#             values=pcp_df[col],
#         )
#         col_list.append(col_dict)
    
#     fig = go.Figure(data=go.Parcoords(
#                             line = dict(color = pcp_df['Zones'],
#                                         colorscale = [[0, '#9933ff'], [0.25, '#ff0066'], [0.5,'#ccffcc'], [0.75,'#66ccff'], [1, '#ffcc66']],
#                                         showscale = False),
#                             dimensions=col_list,
#                 )
#         )
#     fig.update_layout(template=template_from_url(theme), margin=dict(l=100, r=50, b=50) )
    
#     return fig
 
@callback(
    Output('current-sigma', 'children'),
    Input('preturb_knob', 'value'),
)
def update_knob_val(knob_val):
    return 'Scale : ' + str(knob_val)
    
@callback(
    Output('test-preturb-controls-init', 'children'),
    Input('init_preturbation', 'n_clicks'),
    State('whatif_analysis_switch', 'on'),
    State('time_step', 'value'),
)
def update_whatif_init(n_click, whatif, ts):
    if n_click and whatif:
        for var in in_vars:
            preturbation_df[var] = input_ds[var][ts].data
        
        return 'Whatif init at time ' + str(ts)
    else:
        return 'whatif not init!'
    
@callback(
    Output('test-preturb-controls-reset', 'children'),
    Input('clear_preturbation', 'n_clicks'),
    State('whatif_analysis_switch', 'on'),
    State('time_step', 'value'),
)
def update_whatif_reset(n_click, whatif, ts):
    if n_click and whatif:
        for var in in_vars:
            preturbation_df[var] = np.zeros((TOTAL_NUM_POINT,))
        
        return 'Whatif cleared at time ' + str(ts)
    else:
        return 'whatif not cleared!'

@callback(
    Output('test-preturb-controls-update', 'children'),
    Input('update_preturbation', 'n_clicks'),
    State('whatif_analysis_switch', 'on'),
    State('whatif_zone', 'value'),
    State('preturb_var', 'value'),
    State('preturb_knob', 'value'),
    State('time_step', 'value'),
)
def update_whatif_update(n_click, whatif, whatif_zone, varname, knob_val, ts):
    if n_click and whatif:
        #update the selected variable according to the knob/sigma value for the selected zone
        # idx_list = zonal_df.index[zonal_df['Zones'] == whatif_zone].tolist()
        idx_list = zonal_df.index[zonal_df['MCB_site'] == whatif_zone].tolist()
        current_array = preturbation_df[varname].to_numpy()
        current_mean = preturbation_df[varname].mean()
        current_std = preturbation_df[varname].std()
        current_array[idx_list] += (knob_val*current_std)
        preturbation_df[varname] = current_array
        # for var in in_vars:
        #     preturbation_df[var] = np.zeros((TOTAL_NUM_POINT,))
        
        return 'whatif updated at time ' + str(ts)
    else:
        return 'whatif not updated!'

if __name__ == "__main__":
    app.run_server(debug=True)