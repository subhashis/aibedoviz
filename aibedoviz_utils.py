import xarray as xr
import numpy as np

import os
import sys

import pandas as pd
from datetime import date
from sklearn.decomposition import PCA

path_aibedo = '/Users/shazarika/ProjectSpace/currentProjects/AiBEDO/codebase/11_07_22/aibedo/'
sys.path.append(path_aibedo)

import torch
from typing import *


from aibedo.models import BaseModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def concat_variables_into_channel_dim(data: xr.Dataset, variables: List[str]) -> np.ndarray:
    """Concatenate xarray variables into numpy channel dimension (last)."""
    assert len(data[variables[0]].shape) == 2, "Each input data variable must have two dimensions"
    data_ml = np.concatenate(
        [np.expand_dims(data[var].values, axis=-1) for var in variables],
        axis=-1  # last axis
    )
    return data_ml.astype(np.float32)

def get_month_of_output_data(output_xarray: xr.Dataset) -> np.ndarray:
    """ Get month of the snapshot (0-11)  """
    n_gridcells = len(output_xarray['ncells'])
    # .item() is required here as only one timestep is used, the subtraction with -1 because we want 0-indexed months
    month_of_snapshot = np.array(output_xarray['time.month'], dtype=np.float32) - 1
    # now repeat the month for each grid cell/pixel
    dataset_month = np.repeat(month_of_snapshot, n_gridcells)
    return dataset_month.reshape([month_of_snapshot.shape[0], n_gridcells, 1])  # Add a dummy channel/feature dimension

def get_pytorch_model_data(input_xarray: xr.Dataset, output_xarray: xr.Dataset, input_vars: List[str]) -> torch.Tensor:
    """Get the tensor input data for the ML model."""
    # Concatenate all variables into the channel/feature dimension (last) of the input tensor
    data_input = concat_variables_into_channel_dim(input_xarray, input_vars)
    # Get the month of the snapshot (0-11), which is needed to denormalize the model predictions into their original scale
    data_month = get_month_of_output_data(output_xarray)
    # For convenience, we concatenate the month information to the input data, but it is *not* used by the model!
    data_input = np.concatenate([data_input, data_month], axis=-1)
    # Convert to torch tensor and move to CPU/GPU
    data_input = torch.from_numpy(data_input).float().to(device)
    return data_input

def predict_with_aibedo_model(aibedo_model: BaseModel, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Predict with the AiBEDO model.
    Returns:
        A dictionary of output-variable -> prediction-tensor key->value pairs for each variable {var}.
        Keys with name {var} (e.g. 'pr') are in denormalized scale. Keys with name {var}_pre or {var}_nonorm are raw predictions of the ML model.
        To only get the raw predictions, please use aibedo_model.raw_predict(input_tensor)
    """
    aibedo_model.eval()
    with torch.no_grad():  # No need to track the gradients during inference
        prediction = aibedo_model.predict(input_tensor, return_normalized_outputs=True)  # if true, also return {var}_nonorm (or {var}_pre)
    return prediction

def prediction_to_dataset(inDS,preddict,in_vars):
    ds = {var:(['time','ncells'],inDS[var].data) for i,var in enumerate(in_vars[:1])} 
    for var in preddict:
        ds[var] = (['time','ncells'],preddict[var])
    ds_prediction = xr.Dataset(data_vars = ds,
                    coords = {"time":(['time'],inDS.time.values),
                            "lat":(['ncells'],inDS.lat.values),
                            "lon":(["ncells"],inDS.lon.values),},)
    return ds_prediction

def clean_output_dataset(inDS):
    ds = {var:(['time','ncells'],np.zeros_like(inDS[var].data)) for i,var in enumerate(['tas_nonorm', 'pr_nonorm', 'ps_nonorm'])} 
    
    ds_final = xr.Dataset(data_vars = ds,
                    coords = {"time":(['time'],inDS.time.values),
                            "lat":(['ncells'],inDS.lat.values),
                            "lon":(["ncells"],inDS.lon.values),},)
    return ds_final


def run_perturbation(model, ds_input, ds_output, perturbations, invariables, lons = [0,40], lats = [0,30]):
    lat0,lat1 = lats
    lon0,lon1 = lons
    ### Perturb radiation fields
    data_all = []
    for var in invariables:
        if var in perturbations:
            where = np.where((ds_input.lat > lat0) & (ds_input.lat < lat1) & 
                             (ds_input.lon > lon0) & (ds_input.lon < lon1))
            ds_input['{0}'.format(var)][:,where[0]] += perturbations[var]
    
    input_ml = get_pytorch_model_data(ds_input, ds_output, input_vars=model.main_input_vars)
    predictions_ml = predict_with_aibedo_model(model, input_ml)    
    
    ds_prediction = prediction_to_dataset(ds_input,predictions_ml,
                               [var for var in ds_input if 'nonorm' in var])

    return ds_prediction

def generate_perturbed_data(in_data, perturbations, invariables, lons, lats):
    lat0,lat1 = lats
    lon0,lon1 = lons
    ### Perturb radiation fields
    print("invar:", invariables)
    print("perturb:", perturbations)
    
    for var in invariables:
        print("#", var)
        if var in perturbations:
            print("-", var)
            where = np.where((in_data.lat > lat0) & (in_data.lat < lat1) & 
                             (in_data.lon > lon0) & (in_data.lon < lon1))
            in_data['{0}'.format(var)][:,where[0]] += perturbations[var]
            
    


def run_aibedomodel(model, ds_in, ds_out): 
    input_ml = get_pytorch_model_data(ds_in, ds_out, input_vars=model.main_input_vars)
    predictions_ml = predict_with_aibedo_model(model, input_ml)    
    
    ds_prediction = prediction_to_dataset(ds_in,predictions_ml,
                               [var for var in ds_in if 'nonorm' in var])

    return ds_prediction

def reg_avg(ds,var,lats = [0,30],lons = [-150,-110]):
    lat0,lat1 = lats
    lon0,lon1 = lons

    avg = ds[var].where((ds.lat > lat0) & (ds.lat < lat1) & 
                                 (ds.lon > lon0) & (ds.lon < lon1)).mean(('ncells'))
    return avg

def get_regional_data(local_ds, lons, lats):
    lat0,lat1 = lats
    lon0,lon1 = lons
        
    where = np.where((local_ds.lat > lat0) & (local_ds.lat < lat1) & 
                        (local_ds.lon > lon0) & (local_ds.lon < lon1))
    
    local_mean_dict = {
        'pr_nonorm': np.mean(local_ds['pr_nonorm'][:,where[0]].data),
        'ps_nonorm': np.mean(local_ds['ps_nonorm'][:,where[0]].data),
        'tas_nonorm': np.mean(local_ds['tas_nonorm'][:,where[0]].data),
    }
    
    return local_mean_dict


def test_print(msg):
    print(msg)
    
