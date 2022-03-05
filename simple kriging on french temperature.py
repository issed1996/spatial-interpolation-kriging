# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:55:40 2022

@author: Ed-daki Issam
"""

import os
import requests
import pandas

fname_stations = "file_stations.csv"
url_stations = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/postesSynop.csv"

if os.path.exists(fname_stations) is False:
    with open(fname_stations, "wb") as fid:
        file_stations = requests.get(url_stations)
        fid.write(file_stations.content)

fpd_st = pandas.read_csv("file_stations.csv", delimiter=";")
fpd_st = fpd_st[0:40]  # Mainland only

print("How the station file looks like")
fpd_st

#pip install folium

import folium

mapf = folium.Map(location=(47, 2), zoom_start=5)
for _, station in fpd_st.iterrows():
    marker = folium.Marker(location=(station.Latitude, station.Longitude),
                                     weight=1, color="Green",
                                     tooltip=f"{station.Nom.title()} | {station.ID}")
    marker.add_to(mapf)

bound_map = ((42.3, -5), (51.1, 8.3))
rect = folium.Rectangle(bound_map, color="Gray",
                        tooltip="Bounds for the temperature map to come")
rect.add_to(mapf)

mapf




import gzip

year, month = "2020", "01"

url_temp = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/synop.{}{}.csv.gz".format(year, month)
fname_temp_end = url_temp.split("/")[-1].replace("csv.gz", "csv")  # calling gzip is not necessary (why...)
fname_temp = "file_{}".format(fname_temp_end)

if os.path.exists(fname_temp) is False:
    zipped_content = requests.get(url_temp).content
    with open(fname_temp, "wb") as fid:
            fid.write(gzip.decompress(zipped_content))


fpd_tp = pandas.read_csv(fname_temp, delimiter=";")

# Preprocessing: drop useless data, cast to numeric datatypes
fpd_tp = fpd_tp[["numer_sta", "date", "t"]]  # keep useful columns
fpd_tp = fpd_tp[fpd_tp["numer_sta"].isin(fpd_st.ID)]  # keep metropolitan stations
fpd_tp = fpd_tp.replace({"mq": "nan"})  # replace mq by nan
fpd_tp = fpd_tp.astype({"t": float})  # convert temperatures to float

print("Datatypes of the dataframe")
print(fpd_tp.dtypes, sep="")
print("How the dataframe looks like")

fpd_tp












import matplotlib.pyplot as plt
import numpy as np

const_k2c = 273.15  # to convert Kelvin to Celsius
name_tls = "TOULOUSE-BLAGNAC"
id_tls = int(fpd_st.ID[fpd_st.Nom == name_tls])
fpd_tls = fpd_tp[fpd_tp.numer_sta == id_tls]  # data toulouse
time_formatted = pandas.to_datetime(fpd_tls.date, format="%Y%m%d%H%M%S")

plt.figure(figsize=(12, 5))
plt.plot(time_formatted, fpd_tls.t - const_k2c, "-o",
         color="RoyalBlue", linewidth=2, markersize=4)
plt.title(f"Temperature at {name_tls.title()} [°C]")
plt.margins(x=0)
plt.grid()




year, month, day, hour = year, month, "01", "18" # 2020-01-01 @03:00 am

date = int("".join((year, month, day, hour, "00", "00"))) #AAAAMMDDHHMMSS  
print(date)
temp = fpd_tp[fpd_tp.date == date]
#temp.fillna(281.0,inplace=True)
print("number of Nans at chosen date:", np.sum(np.isnan(temp.t.values)))





#pip install pymap3d

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import binned_statistic
import pymap3d
from pymap3d import vincenty

# Compute the pairwise distances
lats_st = fpd_st.Latitude.to_numpy()
lons_st = fpd_st.Longitude.to_numpy()
pdist = np.zeros((lats_st.size, lats_st.size))
for i in range(lats_st.size):
    for j in range(i):
        pdist[i, j] = vincenty.vdist(lats_st[i], lons_st[i], lats_st[j], lons_st[j])[0]
        pdist[j, i] = pdist[i, j]
        
# Compute the raw variogram values
dissim = np.abs(temp["t"].values[:, None] - temp["t"].values[None, :])**2 / 2
dissim_ = squareform(dissim)
pdist_ = squareform(pdist)

# Create a binned variogram
variogram, edges, _ = binned_statistic(pdist_, dissim_,
                                       statistic='mean', bins=20)
dist_binned = edges[:-1] + (edges[1] - edges[0])/2
plt.figure(figsize=(8, 4))
plt.title("Variogram")
plt.plot(squareform(pdist)/1e3, squareform(dissim), ".", markersize=1, label="raw variogram")
plt.xlabel("distance [km]")
plt.ylabel(r"dissimilarity [K$^2$]")    
plt.scatter(dist_binned/1e3, variogram,
            marker="*", color="black", label="binned variogram")
plt.xlabel("distance [km]")
plt.ylabel(r"dissimilarity [K$^2$]")
plt.legend()




from scipy import optimize

def covariance_gaussian(h, a, b):
    return b*np.exp(-h**2 / (2*a**2))

def variogram_gaussian(h, a, b):
    return b*(1 - np.exp(-h**2 / (2*a**2)))

def residual_variogram_gaussian(arr, variogram_data, variogram_dist,
                                weight_res=None):
    a, b = arr
    res =  variogram_gaussian(variogram_dist, a, b) - variogram_data
    if weight_res is None:
        return res
    else:
        return weight_res * res

# For fitting, I weight the data by the number of points in each bin:
weight_res = np.histogram(squareform(pdist), bins=edges)[0].astype(float)
# ...or not:
#weight_res = None

# Fitting here
res_fit = {}
metrics = ["linear", "huber", "soft_l1"]
for metric_ in metrics:  # linear, soft_l1, huber, cauchy, arctan
    res = optimize.least_squares(residual_variogram_gaussian, [300e3, 60],
                                 bounds=([0., 0.], [np.inf, np.inf]),
                                 loss=metric_,
                                 args=[variogram, dist_binned, weight_res])
    res_fit[metric_] = res

plt.figure(figsize=(8, 4))
plt.title("Variogram fitting")
plt.scatter(dist_binned/1e3, variogram,
            s=weight_res,
            marker="*", color="black", label="data")
for metric_, res_ in res_fit.items():
    plt.plot(dist_binned/1e3, variogram_gaussian(dist_binned, *res_.x), "--", label=metric_)
plt.grid()
plt.ylim(ymin=0)
plt.margins(x=1e-3)
plt.xlabel("distance [km]")
plt.ylabel(r"dissimilarity [K$^2$]")
plt.legend()

for metric_, res_ in res_fit.items():
    print(f"Fitted parameters with {metric_} metric:\t a={res_.x[0]} \t b={res_.x[1]}", )






a_learned, b_learned = res_fit["soft_l1"].x
cov_learned = b_learned * np.exp(-pdist**2 / (2*a_learned**2))

# Check the condition number before using cov_learned, as it will be
# inversed the covariance matrix
print("covariance condition number is", np.linalg.cond(cov_learned))
cov_learned_inv = np.linalg.pinv(cov_learned)  # 

plt.figure()
plt.title("Covariance matrix at sensing points")
plt.imshow(cov_learned, vmin=0)
plt.colorbar()




ones_ = np.ones(temp.shape[0])
weights_mean = cov_learned_inv @ ones_ / cov_learned_inv.sum()

mean_krig = temp.t.values @ weights_mean
var_mean_krig = 1 / cov_learned_inv.sum()

print("Temperature \t Mean [°C] \t\t Variance [°C²]")
print("Kriged  \t", mean_krig - const_k2c, " \t {}".format(var_mean_krig))
print("Vanilla \t", temp.t.mean() - const_k2c, "\t {}".format(temp.t.var()))





from tqdm import tqdm

# Create points for interpolation
n_width = 100  # Takes time if n_width > 50...
lat_map_uni = np.linspace(bound_map[0][0], bound_map[1][0], n_width)
lon_map_uni = np.linspace(bound_map[0][1], bound_map[1][1], n_width)
lat_map, lon_map = np.meshgrid(lat_map_uni, lon_map_uni)
lat_map_ = lat_map.ravel()
lon_map_ = lon_map.ravel()

# Create the matrix of parwise distances between latitudes. It takes some time...
rdist = np.zeros((lats_st.size, lat_map_.size))
for i in tqdm(range(lats_st.size)):
    for j in range(lat_map_.size):
        rdist[i, j] = pymap3d.vincenty.vdist(lats_st[i], lons_st[i], lat_map_[j], lon_map_[j])[0]



# Compute the gaussian covariance model
cov_krig_meas = covariance_gaussian(rdist, a_learned, b_learned)

# Do interpolation by simple kriging
weights_sk = cov_learned_inv @ cov_krig_meas
temp_sk = mean_krig + (temp.t.values - mean_krig) @ weights_sk
var_sk = cov_learned[0, 0] - (cov_krig_meas * weights_sk).sum(axis=0)  # hypothesis : cov_learned diagonal elements are equal.




#pip install catopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature

map_temp_sk = temp_sk.reshape((n_width, n_width))
map_var_sk = var_sk.reshape((n_width, n_width))

map_type = "pcolor"  # map style of kriging (expectation)
fig = plt.figure(figsize=(14, 5))

# Kriging - Excpectation
temp_sk_min = 5*np.floor((temp.t.min() - const_k2c)/5)
temp_sk_max = 5*np.ceil((temp.t.max() - const_k2c)/5)
ax = fig.add_subplot(121, projection=ccrs.PlateCarree())
if map_type == "pcolor":
    im = ax.pcolormesh(lon_map_uni, lat_map_uni, (map_temp_sk - const_k2c).T,
                      vmin=temp_sk_min, vmax=temp_sk_max, cmap="jet")
elif map_type == "contourf":
    levels = np.arange(temp_sk_min, temp_sk_max+1e-6, 1)
    im = ax.contourf(lon_map_uni, lat_map_uni, (map_temp_sk - const_k2c).T,
                     levels=levels,
                     extend="min",
                     cmap="jet")
ax.scatter(lons_st, lats_st, c=temp.t.values - const_k2c,
           cmap='jet', edgecolor="black",
           vmin=im.get_clim()[0], vmax=im.get_clim()[1])
plt.colorbar(im)
ax.margins(0)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax.coastlines()
ax.set_title("Simple kriging - temperature [°C]")
ax.set_aspect("auto")

np.min(map_var_sk)
# Kriging - Variance
ax = fig.add_subplot(122, projection=ccrs.PlateCarree())
im = ax.pcolormesh(lon_map_uni, lat_map_uni, (map_var_sk**0.5).T,
                  vmin=0, vmax=0.3, cmap="magma")
ax.scatter(lons_st, lats_st, edgecolor="white", facecolor="none")
plt.colorbar(im)
ax.margins(0)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax.coastlines()
ax.set_title("Simple kriging - standard deviation  [°C]")
ax.set_aspect("auto")             
             


from shapely.geometry import Point, Polygon

lonlat_poly = np.load("file_polygon_mainland_France.npy")
poly = Polygon(lonlat_poly.T)  # create a Polygon

# create Point objects
point_collection = [Point(lon_, lat_)
                    for (lon_, lat_) in zip(lon_map_, lat_map_)]

# create the mask with the point-in-polygon algorithm
mask = np.array([p.within(poly) for p in point_collection])
poly  # visualize the polygon







data=temp.merge(fpd_st ,right_on='ID', left_on='numer_sta')
data=data[['Latitude','Longitude','t']]


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([-5, 10, 42, 52])

ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.plot(data.Longitude, data.Latitude, '.')
ax.set_title('France')




import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.io.img_tiles import OSM

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([-5, 10, 42, 52])

imagery = OSM()
ax.add_image(imagery, 5)
# plus c'est grand, plus c'est précis, plus ça prend du temps

ax.plot(data.Longitude, data.Latitude, '.')
ax.set_title('France')



#%pip install -U numpy

import gstools as gs
bins = gs.standard_bins((data.Latitude, data.Longitude), max_dist=np.deg2rad(8), latlon=True)
bin_c, vario = gs.vario_estimate((data.Latitude, data.Longitude), data.t, bins, latlon=True)



model = gs.Spherical(latlon=True, rescale=gs.EARTH_RADIUS)
model.fit_variogram(bin_c, vario, nugget=False)
ax = model.plot("vario_yadrenko", x_max=bins[-1])
ax.scatter(bin_c, vario)
print(model)

def north_south_drift(lat, lon):
    return lat


uk = gs.krige.Universal(
    model=model,
    cond_pos=(data.Latitude, data.Longitude),
    cond_val=data.t,
    drift_functions=north_south_drift,
)

from pykrige.uk import UniversalKriging

from pykrige.ok import OrdinaryKriging


UK = UniversalKriging(
    data.Latitude,
    data.Longitude,
    data.t,
    variogram_model="gaussian",
    drift_terms=["regional_linear"],
    #enable_plotting=True,
)

#gridx = np.linspace(min(data.Longitude)-0.5,max(data.Longitude)+0.5,100)#,
#gridy = np.linspace(min(data.Latitude)-0.5,max(data.Latitude)+0.5,100)

gridx= np.linspace(42.3,51.1,100)
gridy=np.linspace(-5, 8.3,100)

z, ss = UK.execute("grid", gridx, gridy)
plt.imshow(z)
plt.show()

# Do interpolation by simple kriging
#weights_uk = cov_learned_inv @ cov_krig_meas
temp_sk ,var_sk=UK.execute("grid", gridx, gridy)



np.min(var_sk.data)

var_sk=np.ravel(var_sk.data)
min(var_sk)
var_sk.shape





#pip install catopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature

map_temp_sk = temp_sk.reshape((n_width, n_width))
map_var_sk = var_sk.reshape((n_width, n_width))

map_type = "pcolor"  # map style of kriging (expectation)
fig = plt.figure(figsize=(14, 5))

# Kriging - Excpectation
temp_sk_min = 5*np.floor((temp.t.min() - const_k2c)/5)
temp_sk_max = 5*np.ceil((temp.t.max() - const_k2c)/5)
ax = fig.add_subplot(121, projection=ccrs.PlateCarree())
if map_type == "pcolor":
    im = ax.pcolormesh(lon_map_uni, lat_map_uni, (map_temp_sk - const_k2c).T,
                      vmin=temp_sk_min, vmax=temp_sk_max, cmap="jet")
elif map_type == "contourf":
    levels = np.arange(temp_sk_min, temp_sk_max+1e-6, 1)
    im = ax.contourf(lon_map_uni, lat_map_uni, (map_temp_sk - const_k2c).T,
                     levels=levels,
                     extend="min",
                     cmap="jet")
ax.scatter(lons_st, lats_st, c=temp.t.values - const_k2c,
           cmap='jet', edgecolor="black",
           vmin=im.get_clim()[0], vmax=im.get_clim()[1])
plt.colorbar(im)
ax.margins(0)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax.coastlines()
ax.set_title("Simple kriging - temperature [°C]")
ax.set_aspect("auto")

np.min(map_var_sk)
# Kriging - Variance
ax = fig.add_subplot(122, projection=ccrs.PlateCarree())
im = ax.pcolormesh(lon_map_uni, lat_map_uni, (map_var_sk**0.5).T,
                  vmin=2, vmax=5, cmap="magma")
ax.contourf(lon_map_uni, lat_map_uni,(map_var_sk**0.5).T)
ax.scatter(lons_st, lats_st, edgecolor="white", facecolor="none")
plt.colorbar(im)
ax.margins(0)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax.coastlines()
ax.set_title("Simple kriging - standard deviation  [°C]")
ax.set_aspect("auto")             
             


