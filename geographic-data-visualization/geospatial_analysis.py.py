#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import folium

from folium.plugins import HeatMap


# In[13]:


import os

os.listdir()


# In[14]:


import pandas as pd

df = pd.read_csv("Geospatial_Big_Data__sample_rows_.csv")

df.head()


# In[15]:


print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# In[16]:


lat_col = "lat"
lon_col = "lon"

print("Latitude:", lat_col)
print("Longitude:", lon_col)


# In[17]:


geo_df = df.dropna(subset=[lat_col, lon_col]).copy()

geo_df[lat_col] = pd.to_numeric(geo_df[lat_col], errors="coerce")
geo_df[lon_col] = pd.to_numeric(geo_df[lon_col], errors="coerce")

geo_df = geo_df.dropna(subset=[lat_col, lon_col])

print("Valid geographic records:", len(geo_df))
geo_df.head()


# In[18]:


gdf = gpd.GeoDataFrame(
    geo_df,
    geometry=gpd.points_from_xy(
        geo_df[lon_col],
        geo_df[lat_col]
    ),
    crs="EPSG:4326"
)

gdf.head()


# In[19]:


print("Number of points:", len(gdf))
print("Coordinate system:", gdf.crs)

gdf[["city", "lat", "lon", "geometry"]]


# In[20]:


import folium

center = [
    gdf["lat"].mean(),
    gdf["lon"].mean()
]

m = folium.Map(
    location=center,
    zoom_start=5,
    tiles="CartoDB positron"
)

m


# In[21]:


for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        popup=f"{row['city']} | {row['category']}",
        fill=True
    ).add_to(m)

m


# In[22]:


from folium.plugins import HeatMap

heat_data = gdf[["lat", "lon"]].values.tolist()

HeatMap(
    heat_data,
    radius=12,
    blur=18,
    max_zoom=12,
    min_opacity=0.2
).add_to(m)

m


# In[23]:


m.save("heatmap.html")


# In[24]:


print("Heatmap saved successfully.")


# In[25]:


scatter_map = folium.Map(
    location=center,
    zoom_start=5,
    tiles="CartoDB positron"
)

for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        popup=f"{row['city']} | {row['category']}",
        fill=True
    ).add_to(scatter_map)

scatter_map


# In[26]:


scatter_map.save("scatter_map.html")

