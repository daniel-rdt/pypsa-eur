# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Cluster gas transmission network to clustered model regions.
"""

import logging

logger = logging.getLogger(__name__)

import geopandas as gpd
import pandas as pd
import numpy as np
from packaging.version import Version, parse
from pypsa.geo import haversine_pts, haversine
from shapely import wkt
from shapely.geometry import LineString


def concat_gdf(gdf_list, crs="EPSG:4326"):
    """
    Concatenate multiple geopandas dataframes with common coordinate reference
    system (crs).
    """
    return gpd.GeoDataFrame(pd.concat(gdf_list), crs=crs)


def load_bus_regions(onshore_path, offshore_path):
    """
    Load pypsa-eur on- and offshore regions and concat.
    """
    bus_regions_offshore = gpd.read_file(offshore_path)
    bus_regions_onshore = gpd.read_file(onshore_path)
    bus_regions = concat_gdf([bus_regions_offshore, bus_regions_onshore])
    bus_regions = bus_regions.dissolve(by="name", aggfunc="sum")

    return bus_regions


def load_custom_gas_network(path, **indexcol):
    """
    Loads and cleans custom gas network dataset.
    """
    df = pd.read_csv(path, decimal=",", sep=";", **indexcol)
    # drop columns that are not needed
    to_drop = list(set(df.columns).intersection({"From_Node", "To_Node", "node0_pypsa_region", "node1_pypsa_region", "node0", "node1", "node0_x", "node0_y", "node1_x", "node1_y"}))
    df.drop(columns=to_drop, inplace=True)
    # turn coordinates into points and drop redundant columns
    df["point0"] = gpd.points_from_xy(df.node0_long, df.node0_lat)
    df["point1"] = gpd.points_from_xy(df.node1_long, df.node1_lat)
    df.drop(columns=["node0_long", "node0_lat", "node1_long", "node1_lat"], inplace=True)
    # convert from GWh/day to MWh/h
    cap_col = list(set(df.columns).intersection({"CH4_80bar_GWh_d", "Cap_H2_GWh_d", "Cap_CH4_GWh_d"}))[0]
    df[cap_col] = df[cap_col] * 1e3 / 24
    # rename columns to pypsa names
    df.rename(columns={cap_col: "p_nom", "diameter": "diameter_mm"}, inplace=True)
    # set all pipes in custom dataset as bidirectional
    df["bidirectional"] = True

    return df


def build_clustered_gas_network(df, bus_regions, length_factor=1.25, **kwargs):
    for i in [0, 1]:
        gdf = gpd.GeoDataFrame(geometry=df[f"point{i}"], crs="EPSG:4326")

        kws = (
            dict(op="within")
            if parse(gpd.__version__) < Version("0.10")
            else dict(predicate="within")
        )
        bus_mapping = gpd.sjoin(gdf, bus_regions, how="left", **kws).index_right
        bus_mapping = bus_mapping.groupby(bus_mapping.index).first()

        df[f"bus{i}"] = bus_mapping

        df[f"point{i}"] = df[f"bus{i}"].map(
            bus_regions.to_crs(3035).centroid.to_crs(4326)
        )

    # drop control valves, valves and compressor stations
    if "type" in df.columns:
        df = df.loc[df.type == "pipe"]
    else:
        pipes = ~((df.name.str.startswith("VA")) | (df.name.str.startswith("CV")) | (df.name.str.startswith("CS")))
        df = df.loc[pipes]

    # drop pipes where not both buses are inside regions
    df = df.loc[~df.bus0.isna() & ~df.bus1.isna()]

    # drop pipes within the same region
    df = df.loc[df.bus1 != df.bus0]

    # recalculate lengths as center to center * length factor
    df["length"] = df.apply(
        lambda p: length_factor
                  * haversine_pts([p.point0.x, p.point0.y], [p.point1.x, p.point1.y]),
        axis=1,
    )

    # tidy and create new numbered index
    df.drop(["point0", "point1"], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    def swap_buses(df):
        # create positive order of buses for bidirectional links
        positive_order = (df.bus0 < df.bus1) | (~df.bidirectional)
        df_p = df[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        df_n = df[~positive_order].rename(columns=swap_buses)
        return pd.concat([df_p, df_n])

    df = swap_buses(df)

    reindex_pipes(df, **kwargs)

    strategies = {
        "bus0": "first",
        "bus1": "first",
        "p_nom": "sum",
        "diameter_mm": "mean",
        "length": "mean",
        "name": " ".join,
        "p_min_pu": "min",
    }
    df = aggregate_parallel_pipes(df, strategies)

    return df


def reindex_pipes(df, carrier="gas pipeline", year=""):
    def make_index(x):
        connector = " <-> " if x.bidirectional else " -> "
        if year:
            return f"{carrier} " + x.bus0 + connector + x.bus1 + f"-{year}"
        else:
            return f"{carrier} " + x.bus0 + connector + x.bus1

    df.index = df.apply(make_index, axis=1)

    df["p_min_pu"] = df.bidirectional.apply(lambda bi: -1 if bi else 0)
    df.drop("bidirectional", axis=1, inplace=True)

    df.sort_index(axis=1, inplace=True)


def aggregate_parallel_pipes(df, strategies):
    """
    Aggregate pipelines and dataset according to given strategies
    """
    return df.groupby(df.index).agg(strategies)


def add_geometries(df, bus_regions, map=False):
    """
    returns a gpd.GeoDataFrame with a linestring as geometry
    """
    # add points corresponding to centroid of bus regions
    if map:
        for i in [0, 1]:
            df[f"point{i}"] = df[f"bus{i}"].map(
                bus_regions.to_crs(3035).centroid.to_crs(4326)
            )
    # Create a linestring column
    df["line"] = df.apply(lambda row: LineString([row['point0'], row['point1']]), axis=1)

    return gpd.GeoDataFrame(data=df, geometry=df.line, crs="EPSG:4326")


def filter_for_country(df, ctry):
    """
    Filters a DataFrame with pypsa bus nodes for a specific country
    """
    return df.loc[(df.bus0.str.startswith(ctry)) & (df.bus1.str.startswith(ctry))]


def substitute_country_network(df1, df2, ctry):
    """
    Substitutes gas pipelines for given country in pypsa network
    """
    return pd.concat(
        [
            (df1[~(df1.bus0.str.startswith(ctry) & df1.bus1.str.startswith(ctry))]),
            df2
        ], axis=0).sort_index()
def find_neighbors(gdf):
    """
    Finds neighbors for regions in GeoDataFrame and stores in separate column
    """
    for index, row in gdf.iterrows():
        neighbors = gdf[gdf.geometry.touches(row['geometry'])].index.to_list()
        gdf.at[index, "neighbors"] = neighbors
        # add again as list if only one entry
        if len(neighbors) == 1:
            gdf.at[index, "neighbors"] = list(neighbors)
    return gdf[["neighbors"]]


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("cluster_gas_network_custom", simpl="", clusters="180")

    logging.basicConfig(level=snakemake.config["logging"]["level"])

    fn_custom = snakemake.input.gas_network_custom
    fn = snakemake.input.cleaned_gas_network
    df_default = pd.read_csv(fn, index_col=0)
    for col in ["point0", "point1"]:
        df_default[col] = df_default[col].apply(wkt.loads)

    df_custom = load_custom_gas_network(fn_custom)

    bus_regions = load_bus_regions(
        snakemake.input.regions_onshore, snakemake.input.regions_offshore
    )
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    offshore_regions = gpd.read_file(snakemake.input.regions_offshore).set_index("name")

    # clustered gas network from old dataset
    gas_network = build_clustered_gas_network(df_default, bus_regions)

    # clustered gas network from custom addition
    gas_network_custom = build_clustered_gas_network(df_custom, bus_regions)
    gas_network_custom_de = filter_for_country(gas_network_custom, "DE")

    # merge gas networks to one
    gas_network_merged = substitute_country_network(gas_network, gas_network_custom_de, "DE")

    gas_network_merged.to_csv(snakemake.output.clustered_gas_network_custom)

    # prepare bus onshore regions neighbors for h2 network preparation
    regions_neighbors = find_neighbors(onshore_regions)
    regions_neighbors.to_parquet(snakemake.output.bus_regions_neighbors)

    # calculate centroid distances in km between all bus regions
    bus_regions_centroids = gpd.GeoDataFrame(geometry=bus_regions.to_crs(3035).centroid.to_crs(4326))
    distancematrix = haversine(np.dstack((bus_regions_centroids.geometry.x, bus_regions_centroids.geometry.y))[0],
              np.dstack((bus_regions_centroids.geometry.x, bus_regions_centroids.geometry.y))[0])
    distance_df = pd.DataFrame(data=distancematrix,
                               index=bus_regions_centroids.index.values,
                               columns=bus_regions_centroids.index.values)
    distance_df.to_csv(snakemake.output.bus_regions_centroid_distances)
