
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import sys
from global_land_mask import globe
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd

import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams["legend.handlelength"] = 1.0
mpl.rcParams["legend.frameon"] = True

# fixed parameters 
dt = 1800 # time step unit: s
cpair = 1.00464e3 #J/(kg K)
rho_w = 1000 # kg/m3
gravit = 9.8 # m/s2

# # define functions

# ## Func get_color
# -----------------------------------------------------------------------
def get_color(colormap,color_nums):
    '''
    get color based on user-defined colormap and number or colors needed.
    '''
    palette = plt.get_cmap(colormap)
    colors = []
    for ic in range(color_nums):
        color = palette(ic)
        colors.append(color)
    return colors


# ## Func area_averager
# -----------------------------------------------------------------------
def area_averager(data_plot_xr):
    '''
    calculate weighted area mean
    input data is xarray DataArray
    '''
    weights = np.cos(np.deg2rad(data_plot_xr.lat))
    weights.name = "weights"
    # available in xarray version 0.15 and later
    data_weighted = data_plot_xr.weighted(weights)

    weighted_mean = data_weighted.mean(("lat", "lon"))

    return weighted_mean


# ## Func wgt_p_tp
# -----------------------------------------------------------------------
def wgt_p_tp(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
    '''
    if levs[-1] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[1:] - levs[:-1]

    data_mid = (data[:-1].values+data[1:].values)/2.
    # print('data_mid',data_mid.shape)

    ## move the level to the last axis
    data_mid_trans = data_mid
    # print('data_mid_trans',data_mid_trans.shape)

    data_wgt = np.nansum(data_mid_trans*dp,axis=-1)/gravit # kg/m2
    # print('data_wgt',data_wgt.shape)
    
    # covert to xarray 
    # data_integral = xr.DataArray(data_wgt, coords=data[:,0].coords,dims=data[:,0].dims)

    data_integral = data_wgt
    return data_integral


# ## Func wgt_p_1d
# -----------------------------------------------------------------------
def wgt_p_1d(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
            levs -- must be bottom to top 
    '''
    if levs[0] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[:-1] - levs[1:]

    data_mid = (data[:-1].values+data[1:].values)/2.
    # print('data_mid',data_mid.shape)

    ## move the level to the last axis
    data_mid_trans = data_mid
    # print('data_mid_trans',data_mid_trans.shape)

    data_wgt = np.nansum(data_mid_trans*dp,axis=-1)/gravit # kg/m2
    # print('data_wgt',data_wgt.shape)
    
    ## covert to xarray 
    data_integral = xr.DataArray(data_wgt, coords=data[0].coords,dims=data[0].dims)

    return data_integral


# ## Func wgt_p_TimeLev
# -----------------------------------------------------------------------
def wgt_p_TimeLev(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
            levs -- must be bottom to top 
    '''
    if levs[0] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[:-1] - levs[1:]

    data_mid = (data[:,:-1].values+data[:,1:].values)/2.
    # print('data_mid',data_mid.shape)

    ## move the level to the last axis
    data_mid_trans = data_mid
    # print('data_mid_trans',data_mid_trans.shape)

    data_wgt = np.nansum(data_mid_trans*dp,axis=-1)/gravit # kg/m2
    # print('data_wgt',data_wgt.shape)
    
    ## covert to xarray 
    data_integral = xr.DataArray(data_wgt, coords=data[:,0].coords,dims=data[:,0].dims)

    return data_integral


# ## Func wgt_p_TimeLevLatLon
# -----------------------------------------------------------------------
def wgt_p_TimeLevLatLon(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
            levs -- must be bottom to top 
    '''
    if levs[0] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[:-1].values - levs[1:].values

    data_mid = (data[:,:-1,:].values+data[:,1:,:].values)/2.
    # print('data_mid',data_mid.shape)

    ## move the level to the last axis
    data_mid_trans = np.moveaxis(data_mid,1,-1)
    # print('data_mid_trans',data_mid_trans.shape)

    data_wgt = np.nansum(data_mid_trans*dp,axis=-1)/gravit # kg/m2
    # print('data_wgt',data_wgt.shape)
    
    ## covert to xarray 
    data_integral = xr.DataArray(data_wgt, coords=data[:,0,:].coords,dims=data[:,0,:].dims)

    return data_integral


# ## Func wgt_p_LevLatLon
# -----------------------------------------------------------------------
def wgt_p_LevLatLon(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
            levs -- must be bottom to top 
    '''
    if levs[0] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[:-1].values - levs[1:].values

    data_mid = (data[:-1,:].values+data[1:,:].values)/2.
    # print('data_mid',data_mid.shape)

    ## move the level to the last axis
    data_mid_trans = np.moveaxis(data_mid,0,-1)
    # print('data_mid_trans',data_mid_trans.shape)

    data_wgt = np.nansum(data_mid_trans*dp,axis=-1)/gravit # kg/m2
    # print('data_wgt',data_wgt.shape)
    
    ## covert to xarray 
    data_integral = xr.DataArray(data_wgt, coords=data[0,:].coords,dims=data[0,:].dims)

    return data_integral


# ## Func regime-partitioning
# -----------------------------------------------------------------------
def regime_partitioning(reg,omega700_pi_in,omega700_ab_in,omega700_avg_in,data):  
    '''
    Statc regime partitioning 
    ''' 

    # use the same lon as omega700 for data and EIS
    data['lon'] = omega700_pi_in.lon 
        
    fillvalue = np.nan
    # ============== get land mask ========================
    
    lons = data.coords['lon'].data
    lats = data.coords['lat'].data
    lons_here = np.where(lons>180,lons-360,lons)
    lon_grid,lat_grid = np.meshgrid(lons_here,lats)
    globe_land_mask = globe.is_land(lat_grid,lon_grid)

    if len(data.shape) == 4: # (time,lev,lat,lon)
        tmp = np.tile(globe_land_mask,(data.shape[0],data.shape[1],1,1,))
        omega700_pi = xr.DataArray(np.tile(omega700_pi_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
        omega700_ab = xr.DataArray(np.tile(omega700_ab_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
        omega700_avg = xr.DataArray(np.tile(omega700_avg_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
    elif len(data.shape) == 3: # (lev,lat,lon)
        tmp = np.tile(globe_land_mask,(data.shape[0],1,1,))
        omega700_pi = xr.DataArray(np.tile(omega700_pi_in,(data.shape[0],1,1)), coords=data.coords)
        omega700_ab = xr.DataArray(np.tile(omega700_ab_in,(data.shape[0],1,1)), coords=data.coords)
        omega700_avg = xr.DataArray(np.tile(omega700_avg_in,(data.shape[0],1,1)), coords=data.coords)
    elif len(data.shape) == 2: 
        tmp = globe_land_mask
        omega700_pi = omega700_pi_in
        omega700_ab = omega700_ab_in
        omega700_avg = omega700_avg_in
        
    globe_land_mask = xr.DataArray(tmp,coords=data.coords)
    # print('globe_land_mask.shape=',globe_land_mask.shape,'tmp.shape=',tmp.shape)
    
    avg_flag = xr.zeros_like(data)
    
    if reg == 'TropMarineLow':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = xr.where((omega700_pi>0),data1,fillvalue)
        data2_ab = xr.where((omega700_ab>0),data1,fillvalue)
        data2_avg = xr.where((omega700_avg>0),data1,fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 1)
    if reg == 'TropAscent':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = xr.where((omega700_pi<0),data1,fillvalue)                
        data2_ab = xr.where((omega700_ab<0),data1,fillvalue)       
        data2_avg = xr.where((omega700_avg<0),data1,fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 2)  
    if reg == 'MidLat':
        data1 = data.where((((data.lat>=-60)&(data.lat<-30))|((data.lat<60)&(data.lat>=30))),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'HiLat':
        data1 = data.where((((data.lat<-60))|((data.lat>=60))),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'TropLand':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 
    if reg == 'Global':
        data1 = data
        data2_pi = data1 
        data2_ab = data1
        data2_avg = data1
        avg_flag = 1.0 

    # ================ fractional area ============================
    data2m_pi = xr.where(np.isnan(data2_pi),0.0,1.0)
    data2m_ab = xr.where(np.isnan(data2_ab),0.0,1.0)
    data2m_avg = xr.where(np.isnan(data2_avg),0.0,1.0)
    
    return data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,avg_flag


# ## Func StretechOutNormalize
# -----------------------------------------------------------------------
# define a colormap
# cmap0 = LinearSegmentedColormap.from_list('', ['white', *plt.cm.Blues(np.arange(255))])

class StretchOutNormalize(plt.Normalize):
    def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
        self.low = low
        self.up = up
        plt.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# ## Func save_big_dataset
# -----------------------------------------------------------------------
def save_big_dataset(dic_mod,outfile):
    '''
    create a big dataset based on all variables in a dictionary and save to netcdf file.
    '''
    datalist = []
    for svar in dic_mod.keys():
        data = xr.DataArray(dic_mod[svar],name=svar)
        datalist.append(data)

    data_big = xr.merge(datalist,compat='override')

    #data_big.to_netcdf(outfile,encoding={'time':{'dtype': 'i4'},'bin':{'dtype':'i4'}})
    data_big.to_netcdf(outfile)
