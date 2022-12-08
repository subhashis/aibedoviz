import xarray as xr
import numpy as np

import os
import sys

path_aibedo = '/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/11_07_22/aibedo/'
sys.path.append(path_aibedo)

import torch
from typing import *
from aibedoviz_utils import *


from aibedo.models import BaseModel


##dash libraries
import pandas as pd
from datetime import date
from dash import Dash, dcc, html, dash_table, Input, Output, callback, State
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url

from sklearn.decomposition import PCA

#############################################################################
DATA_DIR = '/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/aibedo_viz/haruki_notebook_10_27_22/LE_CESM2_data/'
# the data used for prediction must be here, as well as the cmip6 mean/std statistics
# Input data filename (isosph is an order 6 icosahedron, isosph5 of order 5, etc.)
filename_input = "isosph5.CESM2-LE.historical.r11i1p1f1.Input.Exp8.nc"
# Output data filename is inferred from the input filename, do not edit!
# E.g.: "compress.isosph.CESM2.historical.r1i1p1f1.Output.nc"
filename_output = filename_input.replace("Input.Exp8.nc", "Output.nc")

ds_input = xr.open_dataset(f"{DATA_DIR}/{filename_input}")  # Input data
ds_output = xr.open_dataset(f"{DATA_DIR}/{filename_output}") # Ground truth data
# Get the appropriate device (GPU or CPU) to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
overrides = [f'datamodule.data_dir={DATA_DIR}', f"++model.use_auxiliary_vars=False"]

output_ds_dict = {}
output_ds_dict['CLEAN'] = clean_output_dataset(ds_output.sel(time=slice("1900-01-01","1900-01-01")))
input_ds_dict = {}

in_vars = [ 'crelSurf_nonorm', 'crel_nonorm', 'cresSurf_nonorm', 'cres_nonorm', 'netTOAcs_nonorm', 'lsMask', 'netSurfcs_nonorm']
out_vars = ['tas_nonorm', 'pr_nonorm', 'ps_nonorm']

mcb_regions = {'SEP':{'lats':[-30,0],'lons':[-110,-70]},
           'NEP':{'lats':[0,30],'lons':[-150,-110]},
           'SEA':{'lats':[-30,0],'lons':[-25,15]},
          }

varname_dict = {
                    'crelSurf_nonorm': 'Longwave surface radiative effect', 
                    'crel_nonorm': 'Longwave TOA radiative effect', 
                    'cresSurf_nonorm': 'Shortwave surface radiative effect',
                    'cres_nonorm': 'Shortwave TOA radiative effect',
                    'netTOAcs_nonorm': 'Net clearsky TOA radiation',
                    'netSurfcs_nonorm': 'Net clearsky Surface radiation',
                    'lsMask': 'land mask',
                    'pr_nonorm': 'Precipitation',
                    'ps_nonorm': 'Sea level pressure',
                    'tas_nonorm': '2-metre air temperature'
                }

tp_region_defs = {
              'Sahel':{'lat':[10,20],'lon':[-15,35],'variable':['pr_nonorm'], 'trend':["increase"]},
              'Atlantic Subpolar Gyre':{'lat':[45,60],'lon':[-50,-20],'variable':['tas_nonorm'], 'trend':["increase"]},
              'Eurasia Boreal':{'lat':[60,80],'lon':[65,170],'variable':['tas_nonorm'], 'trend':["increase"]},
              'America Boreal':{'lat':[60,75],'lon':[-160,-60],'variable':['tas_nonorm'], 'trend':["increase"]},
              'Amazon':{'lat':[-10,10],'lon':[-65,-45],'variable':['pr_nonorm'], 'trend':["decrease"]},
              'Coral Sea':{'lat':[-25,-10],'lon':[145,165],'variable':['tas_nonorm'], 'trend':["increase"]},
              'Barents Sea Ice':{'lat':[70,90],'lon':[10,60],'variable':['tas_nonorm'], 'trend':["increase"]},
            }

#create the PCA dict for OOD testing
pca_dict = {}
for v in [ 'crelSurf_nonorm', 'crel_nonorm', 'cresSurf_nonorm', 'cres_nonorm', 'netTOAcs_nonorm', 'netSurfcs_nonorm']:
    mv_data = ds_input[v][:,:642].data
    pca = PCA()
    pca.fit(mv_data)
    pca_dict[v] = pca

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
    dcc.Markdown("A hybrid AI framework to capture the effects of cloud properties on global circulation and regional climate patterns"), className="text-black p-2 mb-2 text-center"
)


model = torch.load('/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/11_07_22/aibedoviz/fullmodel/MLP_aibedo.pt')
model.eval()

print(model)

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

## INDIVIDUAL INPUT CONTROL DESCRIPTIONS
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


select_output_predict_view  = dbc.RadioItems(
                                    id="out_radio",
                                    className="btn-group",
                                    inputClassName="btn-check",
                                    labelClassName="btn btn-outline-info",
                                    labelCheckedClassName="active",
                                    options=[
                                        {"label": "Current", "value": 'current'},
                                        {"label": "Before MCB", "value": 'before_mcb'},
                                        {"label": "After MCB", "value": 'after_mcb'},
                                        {"label": "Diff", "value": 'diff'},
                                    ],
                                    value='current',
                                )

switch_mcb = daq.BooleanSwitch(
                    #color='k',
                    id="mcb_switch_id",
                    # label='on/off',
                    # labelPosition='top',
                    # size=50,
                    on=False,
                )
dropdown_in_var = dcc.Dropdown(
                        id="in_varname",
                        options=[
                            {"label": str(i), "value": i} for i in [ 'crelSurf_nonorm', 'crel_nonorm', 'cresSurf_nonorm', 'cres_nonorm', 'netTOAcs_nonorm', 'lsMask', 'netSurfcs_nonorm']
                        ],
                        value="crelSurf_nonorm",
                        clearable=False,
                    )

dropdown_out_var =  dcc.Dropdown(
                        id="out_varname",
                        options=[
                            {"label": str(i), "value": i} for i in ['pr_nonorm', 'ps_nonorm', 'tas_nonorm']
                        ],
                        value="tas_nonorm",
                        clearable=False,
                    )
dropdown_tp_reg =  dcc.Dropdown(
                        id="tp_reg",
                        options=[
                            {"label": str(i), "value": i} for i in ['Sahel', 'Atlantic Subpolar Gyre', 'Eurasia Boreal', 'America Boreal', 'Amazon', 'Coral Sea', 'Barents Sea Ice' ]
                        ],
                        value="Coral Sea",
                        clearable=False,
                    )
dropdown_timestep = dcc.Dropdown(
                        id="time_step",
                        options=[
                            {"label": timestring_list[i], "value": i} for i in range(1980)
                        ],
                        value=600,
                        clearable=False,
                    )

dropdown_starttime = dcc.Dropdown(
                        id="start_time",
                        options=[
                            {"label": timestring_list[i], "value": i} for i in range(1980)
                        ],
                        value=600,
                        clearable=False,
                    )
dropdown_endtime = dcc.Dropdown(
                        id="end_time",
                        options=[
                            {"label": timestring_list[i], "value": i} for i in range(1980)
                        ],
                        value=600,
                        clearable=False,
                    )

dcc_timesteppicker = dcc.DatePickerSingle(
                id='single_timestep_picker',
                min_date_allowed=date(1850, 2, 1),
                max_date_allowed=date(2015, 1, 30),
                initial_visible_month=date(1900, 1, 1),
                date=date(1900, 1, 1)
            )

dropdown_projection = dcc.Dropdown(
                        id="input_projection",
                        options=[
                            {"label": str(i), "value": i} for i in ['equirectangular', 'mercator', 'orthographic', 'natural earth', 'kavrayskiy7', 'miller', 'robinson', 'eckert4', 'azimuthal equal area', 'azimuthal equidistant', 'conic equal area', 'conic conformal', 'conic equidistant', 'gnomonic', 'stereographic', 'mollweide', 'hammer', 'transverse mercator', 'albers usa', 'winkel tripel', 'aitoff','sinusoidal']
                        ],
                        value="natural earth",
                        clearable=False,
                    )
dropdown_zoneselect = dcc.Dropdown(
                        id="mcb_zone",
                        options=[
                            {"label": str(i), "value": i} for i in ['NEP', 'SEA', 'SEP']
                        ],
                        value="SEP",
                        clearable=False,
                    )
dropdown_preturbvar = dcc.Dropdown(
                        id="perturb_var",
                        options=[{"label": str(i), "value": i} for i in [ 'crelSurf', 'crel', 'cresSurf', 'cres', 'netTOAcs', 'netSurfcs']],
                        value=None,
                        clearable=False,
                        multi=True,
                    )
buttongroup_whatif = dbc.ButtonGroup(
                        [
                            dbc.Button("Initialize", id="init_preturbation", color="Primary",  size="sm"),
                            dbc.Button("Update", id="update_preturbation", color="warning",  size="sm"),
                            dbc.Button("Reset", id="clear_preturbation", color="success",  size="sm"),
                        ]
                    )
button_mcb_initialize = dbc.Button("Initialize", id="init_mcb", color="primary",  size="sm",  outline=True,className="me-2")
button_mcb_update = dbc.Button("Update", id="update_mcb", color="primary",  size="sm",  outline=True,className="me-2")

button_refresh_tp = dbc.Button("TP Risk", id="refresh_tp", color="info",  size="sm",  outline=True,className="me-2")

knob_whatif = daq.Knob(
                #label="Perturbation Sigma",
                id="preturb_knob",
                size=70,
                max=5,
                value=0.5,
                #color={"gradient":True,"ranges":{"green":[0,50],"yellow":[50,90],"red":[90,100]}},
                #labelPosition='bottom',
            )


slider_crelSurf = dcc.Slider(-50, 50, 0.5, value=0, id='slider_crelSurf', 
                      marks={
                                -50: {'label': '-50'},
                                0: {'label': 'crelSurf','style': {'color': '#000000'}},
                                50: {'label': '50'},
                            }, 
                      included=False,
                      tooltip={"placement": "top", "always_visible": False}
                      )

slider_crel = dcc.Slider(-50, 50, 0.5, value=0,  id='slider_crel',
                      marks={
                                -50: {'label': '-50'},
                                0: {'label': 'crel','style': {'color': '#000000'}},
                                50: {'label': '50'},
                            }, 
                      included=False,
                      tooltip={"placement": "top", "always_visible": False}
                      )

slider_cresSurf = dcc.Slider(-20, 20, 0.5, value=0, id='slider_cresSurf',
                      marks={
                                -20: {'label': '-20'},
                                0: {'label': 'cresSurf','style': {'color': '#000000'}},
                                20: {'label': '20'},
                            }, 
                      included=False,
                      tooltip={"placement": "top", "always_visible": False}
                      )

slider_cres = dcc.Slider(-20, 20, 0.5, value=0, id='slider_cres', 
                      marks={
                                -20: {'label': '-20'},
                                0: {'label': 'cres','style': {'color': '#000000'}},
                                20: {'label': '20'},
                            }, 
                      included=False,
                      tooltip={"placement": "top", "always_visible": False}
                      )
slider_netTOAcs = dcc.Slider(-10, 10, 0.5, value=0, id='slider_netTOAcs', 
                      marks={
                                -10: {'label': '-10'},
                                0: {'label': 'netTOAcs','style': {'color': '#000000'}},
                                10: {'label': '10'},
                            }, 
                      included=False,
                      tooltip={"placement": "top", "always_visible": False}
                      )
slider_netSurfcs = dcc.Slider(-10, 10, 0.5, value=0, id='slider_netSurfcs', 
                      marks={
                                -10: {'label': '-10'},
                                0: {'label': 'netSurfcs','style': {'color': '#000000'}},
                                10: {'label': '10'},
                            }, 
                      included=False,
                      tooltip={"placement": "top", "always_visible": False}
                      )
## table elements:

table_mcb = dash_table.DataTable(
                id='adding-rows-table',
                columns=[
                    {"name": ["","MCB ID"], "id": "mcb_id" },
                    {"name": ["MCB Duration (mm-yyyy)", "Start Time"], "id": "start_time"},
                    {"name": ["MCB Duration (mm-yyyy)", "End Time"], "id": "end_time" },
                    {"name": ["MCB Site (Bounding Box)", "Latitudes"], "id": "lats"},
                    {"name": ["MCB Site (Bounding Box)", "Longitude"], "id": "lons" },
                    {"name": ["Perturbations", "crelSurf"], "id": "crelSurf", "renamable": True},
                    {"name": ["Perturbations", "crel"], "id": "crel", "renamable": True},
                    {"name": ["Perturbations", "cresSurf"], "id": "cresSurf", "renamable": True},
                    {"name": ["Perturbations", "cres"], "id": "cres", "renamable": True},
                    {"name": ["Perturbations", "netTOAcs"], "id": "netTOAcs", "renamable": True},
                    {"name": ["Perturbations", "netSurfcs"], "id": "netSurfcs", "renamable": True},
                    
                    {"name": ["", "Additional Comments"], "id": "comment", "clearable": True, "renamable": True, "deletable": True },
                    
                #     {
                #     'name': 'Column {}'.format(i),
                #     'id': 'column-{}'.format(i),
                #     'deletable': True,
                #     'renamable': True
                # } for i in range(1, 5)
                         ],
                data=[],
                editable=True,
                row_deletable=True,
                export_format='xlsx',
                export_headers='display',
                merge_duplicate_headers=True,
                style_table={'overflowX': 'auto'},
                
            )
table_button = html.Button('Add Row', id='editing-rows-button', n_clicks=0)
button_table = dbc.Button("Save MCB settings", id="save_mcb_record", color="info",  size="sm",  outline=True,className="me-2")


##CONTROL PANELS
control_panel1 = html.Div(
    [
        html.H5(dcc.Markdown("**General Controls**")),
        # html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.H6("Timestep"), md=5),
                dbc.Col(dropdown_timestep,md=7),
            ],
            align="center",
            className="g-0",
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(html.H6("Select time"), md=5),
        #         dbc.Col(dcc_timesteppicker,md=7),
        #     ],
        #     align="center",
        #     className="g-0 bg-light",
        # ),
        # html.Br(),
        dbc.Row(
            [
                dbc.Col(html.H6("Projection"), md=5),
                dbc.Col(dropdown_projection,md=7),
            ],
            align="center",
            className="g-0",
        ),
        html.Hr(),
        html.H5(dcc.Markdown("**Model Controls**")),
        
        
        dbc.Button("Run AiBEDO", id="run_model", color="primary", size="sm", outline=True,className="me-2"),
        dbc.Button("Clear Data", id="clear_prediction", color="secondary", size="sm", outline=True,className="me-2"),
        
        #dbc.Button("Dark", color="dark", size="sm", outline=True),
        
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.H5(dcc.Markdown("**MCB Control**"))),
                dbc.Col(switch_mcb, md=4),
            ]
        ),
        # html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.H6("Start Time"), md=6),
                dbc.Col(html.H6("End Time"), md=6),
                
            ],
            align="center",
            className="g-0",
        ),
        dbc.Row(
            [
                dbc.Col(dropdown_starttime,md=6),
                dbc.Col(dropdown_endtime,md=6),
            ],
            align="center",
            className="g-0",
        ),
        html.Br(),
        dbc.Row(
            [
                 dbc.Col(button_mcb_initialize, md=8),
            ],
            align="center",
            className="g-0",
        ),
        # dbc.Row(dbc.Col(button_mcb_initialize)),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.H6("Select Zone"), md=5),
                dbc.Col(dropdown_zoneselect,md=7),
            ],
            align="center",
            className="g-0",
        ),
        dbc.Row(
            html.H6("Select Variables"),
            align="center",
            className="g-0",
        ),
        dbc.Row(
            dropdown_preturbvar,
            # [
            #     # dbc.Col(html.H6("Variable"), md=5),
            #     dbc.Col(dropdown_preturbvar,md=7),
            # ],
            align="center",
            className="g-0",
        ),
        html.Br(),
        
        dbc.Row(slider_crelSurf, align="center", className="g-0 "),
        dbc.Row(slider_crel, align="center", className="g-0 "),
        dbc.Row(slider_cresSurf, align="center", className="g-0 "),
        dbc.Row(slider_cres, align="center", className="g-0 "),
        dbc.Row(slider_netTOAcs, align="center", className="g-0 "),
        dbc.Row(slider_netSurfcs, align="center", className="g-0 "),
        
        dbc.Row(
            [
                 dbc.Col(button_mcb_update, md=3),
                 dbc.Col(button_refresh_tp, md=3),
                 dbc.Col(button_table, md=6),
                 
            ],
            align="center",
            className="g-0",
        ),
        html.Br(),
        # dbc.Row(dbc.Col(buttongroup_whatif)),
        # dbc.Row(
        #     [
        #         dbc.Col(dbc.Label(id='current-sigma'), md=3),
        #         dbc.Col(knob_whatif,md=9),
        #     ],
        #     align="center",
        #     className="g-0",
        # ),
        
    ],
    className="mb-2",
)


control_panel2 = html.Div(
    [
        html.H6("Activity Log"),
        html.Hr(),
        dbc.Label(id='test-model-control'),
        html.Br(),
        dbc.Label(id='test-mcb-init'),
        html.Br(),
        dbc.Label(id='test-mcb-update'),
        html.Br(),
        dbc.Label(id='test-cleardata'),
        dbc.Label(id='test-preturb-controls-init'),
        dbc.Label(id='test-preturb-controls-update'),
        dbc.Label(id='test-preturb-controls-reset'),
        
    ],
    className="mb-1 p-1",
)


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
            className="g-0",
        ),
        dbc.Row(dbc.Col(dcc.Graph(id="input_panel"))),
        # html.Br(),
    ],
    body=True
    )
c2 = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(html.H6("Variable"), md=2),
                dbc.Col(dropdown_out_var, md=2),
                dbc.Col(html.H6(""), md=1),
                dbc.Col(select_output_predict_view, md=7, align="center",className="g-0"),
                # dbc.Col(html.H6("Model Prediction"), md=3),
                # dbc.Col(switch_output_predict_view,md=1),
            ],
            align="center",
            className="g-0",
        ),
        dbc.Row(dbc.Col(dcc.Graph(id="output_panel"))),
        # html.Br(),
    ], 
    body=True,
    )
c3 = dbc.Card([dcc.Graph(id="distribution_panel", config={'displayModeBar': False})], body=True)
c_table = dbc.Card(html.Div([table_mcb], className="g-0"),body=True)
c4 = dbc.Card([dcc.Graph(id="input_zone_panel"),html.Br(),html.Br(),], body=True)
c5 = dbc.Card([dcc.Graph(id="error_stat_panel"),html.Br(),html.Br(),], body=True)
c6 = dbc.Card(
        [
            dbc.Row(
                [
                dbc.Col(dcc.Graph(id="tipping_points", config={'displayModeBar': False}), md=7),
                dbc.Col(html.H6(""), md=1),
                dbc.Col(
                    [
                        dbc.Row(html.Div([dropdown_tp_reg] , className="g-0")),
                        dbc.Row(dcc.Graph(id="tp_chart")),
                    ],
                    width=4
                ),
                ],
                align="center",
                className="g-0",
            ),
        ],
        body=True
    )

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
        dbc.Tab(c3, label="Distribution View"),
    ]
)
tabs4 = dbc.Tabs(
    [
        dbc.Tab(c6, label='Tipping Point Risk'),
        dbc.Tab(c_table, label="MCB Records"),
    ]
)

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(header)),
        dbc.Row(dbc.Col(description)),
        
        dbc.Row(
            [
                dbc.Col([control1], width=2),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col([tabs1], width=6),
                                dbc.Col([tabs2], width=6),
                            ]
                            ),
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col([tabs3], width=4),
                                dbc.Col([tabs4], width=8),
                            ]
                            ),
                    ],
                    width=10
                ),
                # dbc.Col([tabs1], width=5),
                # dbc.Col([tabs2], width=5),
                
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col([control2, ThemeChangerAIO(aio_id="theme", radio_props={"value":dbc.themes.YETI})], width=12),
                # dbc.Col([tabs3], width=10),
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
    # Input('single_timestep_picker', 'date'),
    Input("input_preturbation_switch", "on"),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)
def update_input_panel(proj, varname, ts, preturb_switch, theme):
    
    resolution = 2562
    
    if preturb_switch and 'MCB' in input_ds_dict.keys():
        # one corner-case not take case of (when preturb_switch is true but MCB is not there in the input_ds_dict)
        vardata_array = np.mean(input_ds_dict['MCB'][varname],axis=0).data[:resolution]
    else:
        mm, yyyy = ts_2_mmyyyy(ts)
        ts_string = str(yyyy) + "-" + f"{mm:02}" + "-01"
        timeslice = slice(ts_string,ts_string)
        vardata_array = ds_input.sel(time=timeslice)[varname][0][:resolution].data
        if preturb_switch:
            # temp solution the corner case
            print("[WARNING]: The MCB input dataset in not initalized!")
            
    
    
    #print(timeslice)
    #varname = 'netSurfcs_nonorm'

    lon_list = ds_input.lon.data[:resolution]
    lat_list = ds_input.lat.data[:resolution]
    marker_size_array = np.full((lat_list.shape[0]), 1.)
    column_name = ['lat', 'lon', 'm_size', varname]

    local_df = pd.DataFrame(data = np.vstack((lat_list, lon_list, marker_size_array, vardata_array)).T, 
                    columns = column_name)
        
    
    fig = px.scatter_geo(local_df, lat="lat", lon="lon",
                     color=varname, # which column to use to set the color of markers
                     #hover_name="val1", # column added to hover information
                     size="m_size",
                     size_max=5, # size of markers
                     projection=proj,
                     color_continuous_scale='Turbo',
                     basemap_visible=True,
                     template=template_from_url(theme))
    
    fig.update_layout(title_text=varname_dict[varname] , title_x=0.45, )
    fig.update_layout(coloraxis_colorbar=dict(
        title=None,
        len=0.5,
        # xanchor="right", #x=1,
        # yanchor='top', #y=0.1,
        thickness=15,
        
    ))
    

    return fig

@callback(
    Output("distribution_panel", "figure"),
    Input("in_varname", "value"),
    State('time_step', 'value'),
    Input("input_preturbation_switch", "on"),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)
def update_distribution_view(varname, ts, perturb_view, theme):
    
    mv_data = ds_input[varname][:,:642].data
    pca = pca_dict[varname]
    transformed_data = pca.transform(mv_data)
    
    ood_test_data = ds_input[varname][ts:ts+1, :642].data
    ood_color = 'rgba(255, 170, 0,0.8)'
    if perturb_view and 'MCB' in input_ds_dict.keys():
        ood_test_data = input_ds_dict['MCB'][varname][:, :642].data
        ood_color = 'rgba(255,0,0,0.8)'
        
    ood_transformed = pca.transform(ood_test_data)
    ood_ll = pca.score_samples(ood_test_data)
    ood_labels = ["ll:" + "{:.2f}".format(s) for s in ood_ll]
        
    
    # perturb_data = perturbed_SEP[vname][:,:642].data

    
    # transformed_data_perturbed = pca.transform(perturb_data)

    # sample_score = pca.score_samples(perturb_data)
    # sample_labels = ["ll:" + "{:.2f}".format(s) for s in sample_score]
    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
            x = transformed_data[:,0],
            y = transformed_data[:,1],
            colorscale = 'Blues',
            xaxis = 'x',
            yaxis = 'y',
        ))
    fig.add_trace(go.Scatter(
            x = transformed_data[:,0],
            y = transformed_data[:,1],
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = dict(
                color = 'rgba(0,0,0,0.1)',
                size = 2
            ),
            name="original"
        ))
    fig.add_trace(go.Scatter(
            x = ood_transformed[:,0],
            y = ood_transformed[:,1],
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = dict(
                color = ood_color,
                size = 8
            ),
            text=ood_labels,
            name="perturbed"
        ))


    fig.update_layout(
        title_text="Out-of-distribution test for " + varname_dict[varname], 
        # title_x=0.45,
        hovermode = 'closest',
        showlegend = False,
        margin=dict(l=0, r=0, b=0, t=40, pad=0),
        template=template_from_url(theme)
    )

    return fig


@callback(
    Output("output_panel", "figure"),
    Input("input_projection", "value"),
    Input("out_varname", "value"),
    Input("time_step", "value"),
    # Input("out_switch", "on"),
    Input("out_radio", "value"),
    State('mcb_switch_id', 'on'),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)
def update_output_panel(proj, varname, ts, radio_select, mcb, theme):
    
    # if show_pred:
    #     local_df = prediction_df
    # else:
    #     local_df = location_df.copy()
    #     local_df[varname] = output_ds[varname][ts].data
    
    print("radio value =", radio_select)
    resolution = 2562
    print("out data dict keys:", output_ds_dict.keys())
    custom_range = None
    
    local_ds = output_ds_dict['CLEAN']
    vardata_array = local_ds[varname][0][:resolution].data
    if radio_select == 'current':
        if 'BASE_PREDICTION' in output_ds_dict.keys():
            local_ds = output_ds_dict['BASE_PREDICTION']
            vardata_array = local_ds[varname][0][:resolution].data
    elif radio_select == 'before_mcb' and mcb:
        if 'BASE_MCB_PREDICTION' in output_ds_dict.keys():
            local_ds = output_ds_dict['BASE_MCB_PREDICTION']
            vardata_array = np.mean(output_ds_dict['BASE_MCB_PREDICTION'][varname],axis=0).data[:resolution]
    elif radio_select == 'after_mcb' and mcb:
        if 'PERTURB_MCB_PREDICTION' in output_ds_dict.keys():
            local_ds = output_ds_dict['PERTURB_MCB_PREDICTION']
            vardata_array = np.mean(output_ds_dict['PERTURB_MCB_PREDICTION'][varname],axis=0).data[:resolution]
    elif radio_select == 'diff' and mcb:
        if 'BASE_MCB_PREDICTION' in output_ds_dict.keys() and 'PERTURB_MCB_PREDICTION' in output_ds_dict.keys():
            local_ds = output_ds_dict['PERTURB_MCB_PREDICTION']
            vardata_array = np.mean(output_ds_dict['PERTURB_MCB_PREDICTION'][varname],axis=0).data[:resolution] - \
                np.mean(output_ds_dict['BASE_MCB_PREDICTION'][varname],axis=0).data[:resolution]
            custom_range = [-0.1,0.1]
    
    
    
    lon_list = local_ds.lon.data[:resolution]
    lat_list = local_ds.lat.data[:resolution]
    marker_size_array = np.full((lat_list.shape[0]), 1.)
    
    column_name = ['lat', 'lon', 'm_size', varname]

    local_df = pd.DataFrame(data = np.vstack((lat_list, lon_list, marker_size_array, vardata_array)).T, 
                    columns = column_name)
    
    fig2 = px.scatter_geo(local_df, lat="lat", lon="lon",
                     color=varname, # which column to use to set the color of markers
                     #hover_name="val1", # column added to hover information
                     size="m_size",
                     size_max=5, # size of markers
                     projection=proj,
                     color_continuous_scale='RdBu_r',
                     basemap_visible=True,
                     range_color=custom_range,
                     template=template_from_url(theme))
    
    # fig2.update_layout(legend=dict(
    #     orientation="v",
    # ))
    fig2.update_layout(title_text=varname_dict[varname], title_x=0.45)
    fig2.update_layout(coloraxis_colorbar=dict(
        title=None,
        len=0.5,
        # xanchor="right", #x=1,
        # yanchor='top', #y=0.1,
        thickness=15,
        
    ))
    
    return fig2

@callback(
    Output('test-model-control', 'children'),
    Input('run_model', 'n_clicks'),
    State('time_step', 'value'),
    State('start_time', 'value'),
    State('end_time', 'value'),
    State('mcb_switch_id', 'on')
)
def update_model_control(n_click, ts, start_ts, end_ts, mcb):
    if n_click and not mcb:
        
        ## run the aibedo model with the selected time slice
        mm, yyyy = ts_2_mmyyyy(ts)
        ts_string = str(yyyy) + "-" + f"{mm:02}" + "-01"
        timeslice = slice(ts_string, ts_string)
        
        output_ds_dict['BASE_PREDICTION'] = run_aibedomodel(model, ds_input.sel(time=timeslice),ds_output.sel(time=timeslice))

        print("##########\n\n\n")
        # test_print("test print!")
        
        
        return 'model update active ' + str(n_click)
        
        
    if n_click and mcb:
        
        ## run aibedo when NCB is selected
        mm, yyyy = ts_2_mmyyyy(start_ts)
        start_ts_string = str(yyyy) + "-" + f"{mm:02}" + "-01"
        mm, yyyy = ts_2_mmyyyy(end_ts)
        end_ts_string = str(yyyy) + "-" + f"{mm:02}" + "-01"
        
        timeslice = slice(start_ts_string, end_ts_string)
        
        output_ds_dict['BASE_MCB_PREDICTION'] = run_aibedomodel(model, ds_input.sel(time=timeslice),ds_output.sel(time=timeslice))

        output_ds_dict['PERTURB_MCB_PREDICTION'] = run_aibedomodel(model, input_ds_dict['MCB'], ds_output.sel(time=timeslice))

        print("##########\n\n\n")
        
        
        return 'model update active ' + str(n_click)
    else:
        return 'Model has not been run yet! '

@callback(
    Output('test-mcb-init', 'children'),
    Input('init_mcb', 'n_clicks'),
    State('start_time', 'value'),
    State('end_time', 'value'),
    State('mcb_switch_id', 'on')
)
def update_mcb_initialize(n_click, start_ts, end_ts, mcb):
    if n_click and mcb:
        
        mm, yyyy = ts_2_mmyyyy(start_ts)
        start_ts_string = str(yyyy) + "-" + f"{mm:02}" + "-01"
        mm, yyyy = ts_2_mmyyyy(end_ts)
        end_ts_string = str(yyyy) + "-" + f"{mm:02}" + "-01"
        
        timeslice = slice(start_ts_string, end_ts_string)
        
        input_ds_dict['MCB'] = ds_input.sel(time=timeslice)
        
        return 'MCB timeline initiated: ' + str(n_click) +  " : starttime=" + start_ts_string + " : endtime=" + end_ts_string
        
        
    else:
        return 'MCB timeline not set!'

@callback(
    Output('test-mcb-update', 'children'),
    Input('update_mcb', 'n_clicks'),
    State('start_time', 'value'),
    State('end_time', 'value'),
    State('mcb_switch_id', 'on'),
    State('mcb_zone', 'value'),
    State('perturb_var', 'value'),
    State('slider_crelSurf', 'value'),
    State('slider_crel', 'value'),
    State('slider_cresSurf', 'value'),
    State('slider_cres', 'value'),
    State('slider_netTOAcs', 'value'),
    State('slider_netSurfcs', 'value')
)
def update_mcb_initialize(n_click, start_ts, end_ts, mcb, mcb_zone, perturb_var, crelSurf_val, crel_val, cresSurf_val, cres_val, netTOAcs_val, netSurfcs_val):
    if n_click and mcb:
        
        mm, yyyy = ts_2_mmyyyy(start_ts)
        start_ts_string = str(yyyy) + "-" + f"{mm:02}" + "-01"
        mm, yyyy = ts_2_mmyyyy(end_ts)
        end_ts_string = str(yyyy) + "-" + f"{mm:02}" + "-01"
        
        # timeslice = slice(start_ts_string, end_ts_string)
        
        # input_ds_dict['MCB'] = ds_input.sel(time=timeslice)
        
        perturb_var_dict = {
                    'crelSurf_nonorm': crelSurf_val, 
                    'crel_nonorm': crel_val, 
                    'cresSurf_nonorm': cresSurf_val,
                    'cres_nonorm': cres_val,
                    'netTOAcs_nonorm': netTOAcs_val,
                    'netSurfcs_nonorm': netSurfcs_val,
                }
        
        perturbations = {}
        for v in perturb_var:
            perturbations[v + '_nonorm'] = perturb_var_dict[v + '_nonorm']
            
        print(perturbations)
        
        print(mcb_regions[mcb_zone]['lats'])
        print(mcb_regions[mcb_zone]['lons'])
        
        
        
        generate_perturbed_data(input_ds_dict['MCB'], perturbations, in_vars, mcb_regions[mcb_zone]['lons'], mcb_regions[mcb_zone]['lats'])
        
        #sanity check of perturbation
        for v in in_vars:
            original_data = ds_input.sel(time=slice(start_ts_string, end_ts_string))[v][0].data
            modified_data =input_ds_dict['MCB'][v][0].data
            print(v + " diff=", np.sum(modified_data - original_data))
        
        return 'MCB updated: ' + str(n_click) +  " : zone=" + mcb_zone
        
        
    else:
        return 'MCB not updated!'

@callback(
    Output("tipping_points", "figure"),
    Input("refresh_tp", "n_clicks"),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)
def update_input_panel(n_click, theme):
    
    reg_name_list = []
    mean_lat_list = []
    mean_lon_list = []
    size_list = []
    for reg in tp_region_defs.keys():
        reg_name_list.append(reg)
        mean_lat_list.append(np.mean(tp_region_defs[reg]['lat']))
        mean_lon_list.append(np.mean(tp_region_defs[reg]['lon']))
        lat_diff = np.abs(tp_region_defs[reg]['lat'][0]- tp_region_defs[reg]['lat'][1])
        lon_diff = np.abs(tp_region_defs[reg]['lon'][0]- tp_region_defs[reg]['lon'][1])
        size_list.append((lat_diff+lon_diff)/2)
        
    
    
    if 'BASE_MCB_PREDICTION' in output_ds_dict.keys() and 'PERTURB_MCB_PREDICTION' in output_ds_dict.keys():
        percent_change_dict = {}
        plotcolor = ["lawngreen", "lawngreen", "lawngreen", "lawngreen", "lawngreen", "lawngreen", "lawngreen",]
        for i,reg in enumerate(tp_region_defs.keys()):
            mcb_change_percent = {}
            before_mcb = get_regional_data(output_ds_dict['BASE_MCB_PREDICTION'], tp_region_defs[reg]['lon'], tp_region_defs[reg]['lat'])
            after_mcb = get_regional_data(output_ds_dict['PERTURB_MCB_PREDICTION'], tp_region_defs[reg]['lon'], tp_region_defs[reg]['lat'])
            for v in out_vars:
                mcb_change_percent[v] = ((after_mcb[v] - before_mcb[v])/before_mcb[v]) * 100.0
            percent_change_dict[reg] = mcb_change_percent
        
        
        for i,reg in enumerate(tp_region_defs.keys()):
            vname = tp_region_defs[reg]['variable'][0]
            change_val = percent_change_dict[reg][vname]
            if tp_region_defs[reg]['trend'][0] == 'increase' and change_val > 0:
                plotcolor[i] = "crimson"
                reg_name_list[i] += ": [MCB Risk] Increase in " + varname_dict[vname]
            elif tp_region_defs[reg]['trend'][0] == 'decrease' and change_val < 0:
                plotcolor[i] = "crimson"
                reg_name_list[i] += ": [MCB Risk] Decrease in " + varname_dict[vname]
            else:
                reg_name_list[i] += ": [MCB Risk] None"
        
        
        print("diff can be calculated!")
    else:
        percent_change_dict = {}
        for reg in tp_region_defs.keys():
            mcb_change_percent = {}
            for v in out_vars:
                mcb_change_percent[v] = 0.0
            percent_change_dict[reg] = mcb_change_percent
        plotcolor = ["royalblue", "royalblue", "royalblue", "royalblue", "royalblue", "royalblue", "royalblue",]
        # plotcolor = "crimson" #"lightgreen" #"royalblue"
        print("no compute available!")
    
    print(percent_change_dict)    
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattergeo(
        lon = mean_lon_list,
        lat = mean_lat_list,
        text = reg_name_list,
        marker = dict(
            size = size_list,
            color = plotcolor,
            line_color='rgb(40,40,40)',
            line_width=2,
            sizemode = 'diameter',
            symbol='circle-dot'
        ),
        name = '{0} - {1}'.format('hello','world')))
    
    fig.update_layout(
        title_text = 'Risk of Regional Tipping Points after MCB',
        title_x=0.5,
        showlegend = False,
        margin=dict(l=0, r=0, b=0, t=40, pad=0),
        geo = dict(
            landcolor = 'rgb(242, 242, 242)',
            framecolor = 'white'
        ),
        template=template_from_url(theme),
    )
    
    

    return fig

@callback(
    Output("tp_chart", "figure"),
    Input("refresh_tp", "n_clicks"),
    Input("tp_reg", "value"),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)
def update_tp_chart(n_clicks, tp_reg, theme):
    
    
    if 'BASE_MCB_PREDICTION' in output_ds_dict.keys() and 'PERTURB_MCB_PREDICTION' in output_ds_dict.keys():
        mcb_change_percent = {}
        before_mcb = get_regional_data(output_ds_dict['BASE_MCB_PREDICTION'], tp_region_defs[tp_reg]['lon'], tp_region_defs[tp_reg]['lat'])
        after_mcb = get_regional_data(output_ds_dict['PERTURB_MCB_PREDICTION'], tp_region_defs[tp_reg]['lon'], tp_region_defs[tp_reg]['lat'])
        for v in out_vars:
            mcb_change_percent[v] = ((after_mcb[v] - before_mcb[v])/before_mcb[v]) * 100.0
        
    else:
        mcb_change_percent = {}
        for v in out_vars:
            mcb_change_percent[v] = 0.0
    
    
    x = [v[:-7] for v in list(mcb_change_percent.keys())]
    y = [mcb_change_percent[v] for v in list(mcb_change_percent.keys())]

    # Use textposition='auto' for direct text
    fig = go.Figure(data=[go.Bar(
                x=x, y=y,
                text=y,
                width=[0.6, 0.6, 0.6]
            )])

    fig.update_traces(texttemplate='%{text:.2s}' + '%', textposition='outside')
    fig.update_yaxes(range=(-100,100), title="Percentage change after MCB")
    fig.update_layout(coloraxis_colorbar=dict(
        title=None,
        len=0.5,
        # xanchor="right", #x=1,
        # yanchor='top', #y=0.1,
        # thickness=15,
        
    ))
    

    return fig


@app.callback(
    Output('adding-rows-table', 'data'),
    Input('save_mcb_record', 'n_clicks'),
    State('adding-rows-table', 'data'),
    State('adding-rows-table', 'columns'),
    State('start_time', 'value'),
    State('end_time', 'value'),
    State('mcb_zone', 'value'),
    State('perturb_var', 'value'),
    State('slider_crelSurf', 'value'),
    State('slider_crel', 'value'),
    State('slider_cresSurf', 'value'),
    State('slider_cres', 'value'),
    State('slider_netTOAcs', 'value'),
    State('slider_netSurfcs', 'value'))
def add_row(n_clicks, rows, columns, start_ts, end_ts, mcb_zone, perturb_var, crelSurf_val, crel_val, cresSurf_val, cres_val, netTOAcs_val, netSurfcs_val):
  
    if n_clicks:
        
        mm, yyyy = ts_2_mmyyyy(start_ts)
        start_ts_string = f"{mm:02}" +  "-" + str(yyyy)
        mm, yyyy = ts_2_mmyyyy(end_ts)
        end_ts_string = f"{mm:02}" +  "-" + str(yyyy)
        
        temp_row = {
                "mcb_id":str(n_clicks),
                "start_time": start_ts_string,
                "end_time": end_ts_string,
                "lats": "{:.2f}".format(mcb_regions[mcb_zone]['lats'][0]) + ", " + "{:.2f}".format(mcb_regions[mcb_zone]['lats'][1]),
                "lons": "{:.2f}".format(mcb_regions[mcb_zone]['lons'][0]) + ", " + "{:.2f}".format(mcb_regions[mcb_zone]['lons'][1]),
                "crelSurf": "{:.2f}".format(crelSurf_val),
                "crel": "{:.2f}".format(crel_val),
                "cresSurf": "{:.2f}".format(cresSurf_val),
                "cres": "{:.2f}".format(cres_val),
                "netTOAcs": "{:.2f}".format(netTOAcs_val),
                "netSurfcs": "{:.2f}".format(netSurfcs_val),
                "comment": ""
            }
        rows.append(temp_row)
    return rows
    


if __name__ == "__main__":
    app.run_server(debug=True)

