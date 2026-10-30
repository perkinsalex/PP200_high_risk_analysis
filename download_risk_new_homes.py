#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:41:32 2026

@author: alexandraperkins
"""


import os
import requests
import geopandas as gpd

URL = (
    "https://gis.water.ca.gov/arcgis/rest/services/"
    "Planning/i15_Parcels_Assessor_Lightbox/MapServer/0/query"
)

TEST_FILE = os.path.expanduser(
    "~/Downloads/ca_parcels_test_batch.gpkg"
)

params = {
    "where": "OBJECTID > 3442000",
    "outFields": "*",
    "returnGeometry": "true",
    "outSR": 4326,
    "f": "geojson",
    "resultRecordCount": 10,
    "orderByFields": "OBJECTID ASC",
}

response = requests.get(URL, params=params, timeout=180)
response.raise_for_status()
data = response.json()

if "error" in data:
    raise RuntimeError(data["error"])

features = data.get("features", [])
print("Features received:", len(features))

if features:
    gdf = gpd.GeoDataFrame.from_features(
        features, crs="EPSG:4326"
    )
    print("First OBJECTID:", gdf["OBJECTID"].min())
    print("Last OBJECTID:", gdf["OBJECTID"].max())

    gdf.to_file(
        TEST_FILE,
        layer="parcels",
        driver="GPKG",
        index=False,
        promote_to_multi=True,
        engine="pyogrio",
    )

    check = gpd.read_file(TEST_FILE, layer="parcels")
    print("Test write succeeded:", len(check), "parcels")
    print("Test file:", TEST_FILE)
else:
    print("No features returned after OBJECTID 3442000.")

