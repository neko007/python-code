import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = 'data/precip.2018.nc'
dset = xr.open_dataset(filename)
dset_lat = dset.lat
lat = dset_lat.data
lat_longname = dset_lat.long_name

dset_precip = dset.precip
precip = dset_precip.data

coords = dset_precip.coords
dims = dset_precip.dims
lat = coords['lat'].data

#%% 
### 构造数据
value1 = np.random.randn(30, 72, 144)
value2 = np.random.randn(30, 72, 144)
lat = np.linspace(-90, 90, 72)
lon = np.linspace(0, 360, 144)
time = pd.date_range('2019-01-01', periods=30)

A = xr.DataArray(value1, coords=[time, lat, lon], dims=['time', 'lat', 'lon'])
A.name = 'random_value_1'
A.attrs['unit'] = '1'

B = xr.DataArray(value2, coords=[time, lat, lon], dims=['time', 'lat', 'lon'])
B.name = 'random_value_2'
B.attrs['unit'] = '2'

ds = xr.Dataset({'random_value_1': A, 'random_value_2': B})
ds.attrs['title'] = 'example'

### 坐标索引
a = A[dict(time=10, lat=30, lon=120)]
# 整数索引
a_time_lat_lon = A.isel(time=slice(3, 6), lat=slice(30, 40), lon=slice(40, 50))
print(a_time_lat_lon)
# 标签索引
a1 = A.sel(time='2019-01-10', lat=30, lon=30, method='nearest') # 最近邻匹配 method = 'nearest', 'pad', 'backfill'
print(a1)
# 对整个dataset进行标签索引
ds1 = ds.sel(time='2019-01-10', lat=30, lon=30, method='nearest')
print(ds1)
#%%
### mask
# method 1
A_mask = A.where(A>0)
print(A_mask)

# method 2
mask = (
    (ds.coords['lat'] > 20)& 
    (ds.coords['lat'] < 60)& 
    (ds.coords['lon'] > 220)& 
    (ds.coords['lon'] < 260)
)

A_mask = xr.where(mask, 0, A)
print(A_mask)
#%%
# 删除和添加变量
ds = ds.drop_vars('random_value_1')
ds = ds.assign(random_value_1_mask = A_mask)

#%% 一些计算
A_mask1 = A_mask.mean(dim='time', keep_attrs=True)

g = xr.DataArray(np.array([1, 2, np.nan]), dims=['x'])
print(g.min())

#%%
import calendar as cl

t = ds.coords['time'].to_index()











