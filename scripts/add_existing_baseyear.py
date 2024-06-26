# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Adds existing power and heat generation capacities for initial planning
horizon.
"""

import logging

logger = logging.getLogger(__name__)

import pandas as pd

idx = pd.IndexSlice

from types import SimpleNamespace

import country_converter as coco
import numpy as np
import pypsa
import xarray as xr
from _helpers import override_component_attrs, update_config_with_sector_opts
from prepare_sector_network import cluster_heat_buses, define_spatial, prepare_costs
from add_brownfield import add_brownfield, add_build_year_to_new_assets, add_ocgt_retro, load_custom_gas_stores, add_custom_gas_stores
from cluster_gas_network_custom import filter_for_country

cc = coco.CountryConverter()

spatial = SimpleNamespace()


def add_existing_renewables(df_agg):
    """
    Append existing renewables to the df_agg pd.DataFrame with the conventional
    power plants.
    """
    carriers = {"solar": "solar", "onwind": "onwind", "offwind": "offwind-ac"}

    for tech in ["solar", "onwind", "offwind"]:
        carrier = carriers[tech]

        df = pd.read_csv(snakemake.input[f"existing_{tech}"], index_col=0).fillna(0.0)
        df.columns = df.columns.astype(int)
        df.index = cc.convert(df.index, to="iso2")

        # calculate yearly differences
        df.insert(loc=0, value=0.0, column="1999")
        df = df.diff(axis=1).drop("1999", axis=1).clip(lower=0)

        # distribute capacities among nodes according to capacity factor
        # weighting with nodal_fraction
        elec_buses = n.buses.index[n.buses.carrier == "AC"].union(
            n.buses.index[n.buses.carrier == "DC"]
        )
        nodal_fraction = pd.Series(0.0, elec_buses)

        for country in n.buses.loc[elec_buses, "country"].unique():
            gens = n.generators.index[
                (n.generators.index.str[:2] == country)
                & (n.generators.carrier == carrier)
            ]
            cfs = n.generators_t.p_max_pu[gens].mean()
            cfs_key = cfs / cfs.sum()
            nodal_fraction.loc[n.generators.loc[gens, "bus"]] = cfs_key.values

        nodal_df = df.loc[n.buses.loc[elec_buses, "country"]]
        nodal_df.index = elec_buses
        nodal_df = nodal_df.multiply(nodal_fraction, axis=0)

        for year in nodal_df.columns:
            for node in nodal_df.index:
                name = f"{node}-{tech}-{year}"
                capacity = nodal_df.loc[node, year]
                if capacity > 0.0:
                    df_agg.at[name, "Fueltype"] = tech
                    df_agg.at[name, "Capacity"] = capacity
                    df_agg.at[name, "DateIn"] = year
                    df_agg.at[name, "cluster_bus"] = node


def add_power_capacities_installed_before_baseyear(n, grouping_years, costs, baseyear):
    """
    Parameters
    ----------
    n : pypsa.Network
    grouping_years :
        intervals to group existing capacities
    costs :
        to read lifetime to estimate YearDecomissioning
    baseyear : int
    """
    logger.debug(
        f"Adding power capacities installed before {baseyear} from powerplants.csv"
    )

    df_agg = pd.read_csv(snakemake.input.powerplants, index_col=0)

    rename_fuel = {
        "Hard Coal": "coal",
        "Lignite": "lignite",
        "Nuclear": "nuclear",
        "Oil": "oil",
        "OCGT": "OCGT",
        "CCGT": "CCGT",
        "Natural Gas": "gas",
        "Bioenergy": "urban central solid biomass CHP",
    }

    fueltype_to_drop = [
        "Hydro",
        "Wind",
        "Solar",
        "Geothermal",
        "Waste",
        "Other",
        "CCGT, Thermal",
    ]

    technology_to_drop = ["Pv", "Storage Technologies"]

    # drop unused fueltyps and technologies
    df_agg.drop(df_agg.index[df_agg.Fueltype.isin(fueltype_to_drop)], inplace=True)
    df_agg.drop(df_agg.index[df_agg.Technology.isin(technology_to_drop)], inplace=True)
    df_agg.Fueltype = df_agg.Fueltype.map(rename_fuel)

    # Intermediate fix for DateIn & DateOut
    # Fill missing DateIn
    biomass_i = df_agg.loc[df_agg.Fueltype == "urban central solid biomass CHP"].index
    mean = df_agg.loc[biomass_i, "DateIn"].mean()
    df_agg.loc[biomass_i, "DateIn"] = df_agg.loc[biomass_i, "DateIn"].fillna(int(mean))
    # Fill missing DateOut
    dateout = (
        df_agg.loc[biomass_i, "DateIn"]
        + snakemake.params.costs["fill_values"]["lifetime"]
    )
    df_agg.loc[biomass_i, "DateOut"] = df_agg.loc[biomass_i, "DateOut"].fillna(dateout)

    # drop assets which are already phased out / decommissioned
    phased_out = df_agg[df_agg["DateOut"] < baseyear].index
    df_agg.drop(phased_out, inplace=True)

    # calculate remaining lifetime before phase-out (+1 because assuming
    # phase out date at the end of the year)
    df_agg["lifetime"] = df_agg.DateOut - df_agg.DateIn + 1

    # assign clustered bus
    busmap_s = pd.read_csv(snakemake.input.busmap_s, index_col=0).squeeze()
    busmap = pd.read_csv(snakemake.input.busmap, index_col=0).squeeze()

    inv_busmap = {}
    for k, v in busmap.items():
        inv_busmap[v] = inv_busmap.get(v, []) + [k]

    clustermaps = busmap_s.map(busmap)
    clustermaps.index = clustermaps.index.astype(int)

    df_agg["cluster_bus"] = df_agg.bus.map(clustermaps)

    # include renewables in df_agg
    add_existing_renewables(df_agg)

    df_agg["grouping_year"] = np.take(
        grouping_years, np.digitize(df_agg.DateIn, grouping_years, right=True)
    )

    df = df_agg.pivot_table(
        index=["grouping_year", "Fueltype"],
        columns="cluster_bus",
        values="Capacity",
        aggfunc="sum",
    )

    lifetime = df_agg.pivot_table(
        index=["grouping_year", "Fueltype"],
        columns="cluster_bus",
        values="lifetime",
        aggfunc="mean",  # currently taken mean for clustering lifetimes
    )

    carrier = {
        "OCGT": "gas",
        "CCGT": "gas",
        "coal": "coal",
        "oil": "oil",
        "lignite": "lignite",
        "nuclear": "uranium",
        "urban central solid biomass CHP": "biomass",
    }

    for grouping_year, generator in df.index:
        # capacity is the capacity in MW at each node for this
        capacity = df.loc[grouping_year, generator]
        capacity = capacity[~capacity.isna()]
        capacity = capacity[
            capacity > snakemake.params.existing_capacities["threshold_capacity"]
        ]
        suffix = "-ac" if generator == "offwind" else ""
        name_suffix = f" {generator}{suffix}-{grouping_year}"
        asset_i = capacity.index + name_suffix
        if generator in ["solar", "onwind", "offwind"]:
            # to consider electricity grid connection costs or a split between
            # solar utility and rooftop as well, rather take cost assumptions
            # from existing network than from the cost database
            capital_cost = n.generators.loc[
                n.generators.carrier == generator + suffix, "capital_cost"
            ].mean()
            marginal_cost = n.generators.loc[
                n.generators.carrier == generator + suffix, "marginal_cost"
            ].mean()
            # check if assets are already in network (e.g. for 2020)
            already_build = n.generators.index.intersection(asset_i)
            new_build = asset_i.difference(n.generators.index)

            # this is for the year 2020
            if not already_build.empty:
                n.generators.loc[already_build, "p_nom_min"] = capacity.loc[
                    already_build.str.replace(name_suffix, "")
                ].values
            new_capacity = capacity.loc[new_build.str.replace(name_suffix, "")]

            if "m" in snakemake.wildcards.clusters:
                for ind in new_capacity.index:
                    # existing capacities are split evenly among regions in every country
                    inv_ind = [i for i in inv_busmap[ind]]

                    # for offshore the splitting only includes coastal regions
                    inv_ind = [
                        i for i in inv_ind if (i + name_suffix) in n.generators.index
                    ]

                    p_max_pu = n.generators_t.p_max_pu[
                        [i + name_suffix for i in inv_ind]
                    ]
                    p_max_pu.columns = [i + name_suffix for i in inv_ind]

                    n.madd(
                        "Generator",
                        [i + name_suffix for i in inv_ind],
                        bus=ind,
                        carrier=generator,
                        p_nom=new_capacity[ind]
                        / len(inv_ind),  # split among regions in a country
                        marginal_cost=marginal_cost,
                        capital_cost=capital_cost,
                        efficiency=costs.at[generator, "efficiency"],
                        p_max_pu=p_max_pu,
                        build_year=grouping_year,
                        lifetime=costs.at[generator, "lifetime"],
                    )

            else:
                p_max_pu = n.generators_t.p_max_pu[
                    capacity.index + f" {generator}{suffix}-{baseyear}"
                ]

                if not new_build.empty:
                    n.madd(
                        "Generator",
                        new_capacity.index,
                        suffix=" " + name_suffix,
                        bus=new_capacity.index,
                        carrier=generator,
                        p_nom=new_capacity,
                        marginal_cost=marginal_cost,
                        capital_cost=capital_cost,
                        efficiency=costs.at[generator, "efficiency"],
                        p_max_pu=p_max_pu.rename(columns=n.generators.bus),
                        build_year=grouping_year,
                        lifetime=costs.at[generator, "lifetime"],
                    )

        else:
            bus0 = vars(spatial)[carrier[generator]].nodes
            if "EU" not in vars(spatial)[carrier[generator]].locations:
                bus0 = bus0.intersection(capacity.index + " gas")

            already_build = n.links.index.intersection(asset_i)
            new_build = asset_i.difference(n.links.index)
            lifetime_assets = lifetime.loc[grouping_year, generator].dropna()

            # this is for the year 2020
            if not already_build.empty:
                n.links.loc[already_build, "p_nom_min"] = capacity.loc[
                    already_build.str.replace(name_suffix, "")
                ].values

            if not new_build.empty:
                new_capacity = capacity.loc[new_build.str.replace(name_suffix, "")]

                if generator != "urban central solid biomass CHP":
                    n.madd(
                        "Link",
                        new_capacity.index,
                        suffix=name_suffix,
                        bus0=bus0,
                        bus1=new_capacity.index,
                        bus2="co2 atmosphere",
                        carrier=generator,
                        marginal_cost=costs.at[generator, "efficiency"]
                        * costs.at[generator, "VOM"],  # NB: VOM is per MWel
                        capital_cost=costs.at[generator, "efficiency"]
                        * costs.at[generator, "fixed"],  # NB: fixed cost is per MWel
                        p_nom=new_capacity / costs.at[generator, "efficiency"],
                        efficiency=costs.at[generator, "efficiency"],
                        efficiency2=costs.at[carrier[generator], "CO2 intensity"],
                        build_year=grouping_year,
                        lifetime=lifetime_assets.loc[new_capacity.index],
                    )
                else:
                    key = "central solid biomass CHP"
                    n.madd(
                        "Link",
                        new_capacity.index,
                        suffix=name_suffix,
                        bus0=spatial.biomass.df.loc[new_capacity.index]["nodes"].values,
                        bus1=new_capacity.index,
                        bus2=new_capacity.index + " urban central heat",
                        carrier=generator,
                        p_nom=new_capacity / costs.at[key, "efficiency"],
                        capital_cost=costs.at[key, "fixed"]
                        * costs.at[key, "efficiency"],
                        marginal_cost=costs.at[key, "VOM"],
                        efficiency=costs.at[key, "efficiency"],
                        build_year=grouping_year,
                        efficiency2=costs.at[key, "efficiency-heat"],
                        lifetime=lifetime_assets.loc[new_capacity.index],
                    )
        # check if existing capacities are larger than technical potential
        existing_large = n.generators[
            n.generators["p_nom_min"] > n.generators["p_nom_max"]
        ].index
        if len(existing_large):
            logger.warning(
                f"Existing capacities larger than technical potential for {existing_large},\
                           adjust technical potential to existing capacities"
            )
            n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
                existing_large, "p_nom_min"
            ]


def add_heating_capacities_installed_before_baseyear(
    n,
    baseyear,
    grouping_years,
    ashp_cop,
    gshp_cop,
    time_dep_hp_cop,
    costs,
    default_lifetime,
):
    """
    Parameters
    ----------
    n : pypsa.Network
    baseyear : last year covered in the existing capacities database
    grouping_years : intervals to group existing capacities
        linear decommissioning of heating capacities from 2020 to 2045 is
        currently assumed heating capacities split between residential and
        services proportional to heating load in both 50% capacities
        in rural busess 50% in urban buses
    """
    logger.debug(f"Adding heating capacities installed before {baseyear}")

    # Add existing heating capacities, data comes from the study
    # "Mapping and analyses of the current and future (2020 - 2030)
    # heating/cooling fuel deployment (fossil/renewables) "
    # https://ec.europa.eu/energy/studies/mapping-and-analyses-current-and-future-2020-2030-heatingcooling-fuel-deployment_en?redir=1
    # file: "WP2_DataAnnex_1_BuildingTechs_ForPublication_201603.xls" -> "existing_heating_raw.csv".
    # TODO start from original file

    # retrieve existing heating capacities
    techs = [
        "gas boiler",
        "oil boiler",
        "resistive heater",
        "air heat pump",
        "ground heat pump",
    ]
    df = pd.read_csv(snakemake.input.existing_heating, index_col=0, header=0)

    # data for Albania, Montenegro and Macedonia not included in database
    df.loc["Albania"] = np.nan
    df.loc["Montenegro"] = np.nan
    df.loc["Macedonia"] = np.nan

    df.fillna(0.0, inplace=True)

    # convert GW to MW
    df *= 1e3

    df.index = cc.convert(df.index, to="iso2")

    # coal and oil boilers are assimilated to oil boilers
    df["oil boiler"] = df["oil boiler"] + df["coal boiler"]
    df.drop(["coal boiler"], axis=1, inplace=True)

    # distribute technologies to nodes by population
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)

    nodal_df = df.loc[pop_layout.ct]
    nodal_df.index = pop_layout.index
    nodal_df = nodal_df.multiply(pop_layout.fraction, axis=0)

    # split existing capacities between residential and services
    # proportional to energy demand
    ratio_residential = pd.Series(
        [
            (
                n.loads_t.p_set.sum()[f"{node} residential rural heat"]
                / (
                    n.loads_t.p_set.sum()[f"{node} residential rural heat"]
                    + n.loads_t.p_set.sum()[f"{node} services rural heat"]
                )
            )
            # if rural heating demand for one of the nodes doesn't exist,
            # then columns were dropped before and heating demand share should be 0.0
            if (f"{node} residential rural heat" in n.loads_t.p_set.sum().index)
               & (f"{node} services rural heat" in n.loads_t.p_set.sum().index)
            else 0.0
            for node in nodal_df.index
        ],
        index=nodal_df.index,
    )

    for tech in techs:
        nodal_df["residential " + tech] = nodal_df[tech] * ratio_residential
        nodal_df["services " + tech] = nodal_df[tech] * (1 - ratio_residential)

    names = [
        "residential rural",
        "services rural",
        "residential urban decentral",
        "services urban decentral",
        "urban central",
    ]

    nodes = {}
    p_nom = {}
    for name in names:
        name_type = "central" if name == "urban central" else "decentral"
        nodes[name] = pd.Index(
            np.unique([
                n.buses.at[index, "location"]
                for index in n.buses.index[
                    n.buses.index.str.contains(name)
                    & n.buses.index.str.contains("heat")
                ]
            ])
        )
        heat_pump_type = "air" if "urban" in name else "ground"
        heat_type = "residential" if "residential" in name else "services"

        if name == "urban central":
            p_nom[name] = nodal_df["air heat pump"][nodes[name]]
        else:
            p_nom[name] = nodal_df[f"{heat_type} {heat_pump_type} heat pump"][
                nodes[name]
            ]

        # Add heat pumps
        costs_name = f"decentral {heat_pump_type}-sourced heat pump"

        cop = {"air": ashp_cop, "ground": gshp_cop}

        if time_dep_hp_cop:
            efficiency = cop[heat_pump_type][nodes[name]]
        else:
            efficiency = costs.at[costs_name, "efficiency"]

        for i, grouping_year in enumerate(grouping_years):
            if int(grouping_year) + default_lifetime <= int(baseyear):
                continue

            # installation is assumed to be linear for the past 25 years (default lifetime)
            ratio = (int(grouping_year) - int(grouping_years[i - 1])) / default_lifetime

            n.madd(
                "Link",
                nodes[name],
                suffix=f" {name} {heat_pump_type} heat pump-{grouping_year}",
                bus0=nodes[name],
                bus1=nodes[name] + " " + name + " heat",
                carrier=f"{name} {heat_pump_type} heat pump",
                efficiency=efficiency,
                capital_cost=costs.at[costs_name, "efficiency"]
                * costs.at[costs_name, "fixed"],
                p_nom=p_nom[name] * ratio / costs.at[costs_name, "efficiency"],
                build_year=int(grouping_year),
                lifetime=costs.at[costs_name, "lifetime"],
            )

            # add resistive heater, gas boilers and oil boilers
            # (50% capacities to rural buses, 50% to urban buses)
            n.madd(
                "Link",
                nodes[name],
                suffix=f" {name} resistive heater-{grouping_year}",
                bus0=nodes[name],
                bus1=nodes[name] + " " + name + " heat",
                carrier=name + " resistive heater",
                efficiency=costs.at[name_type + " resistive heater", "efficiency"],
                capital_cost=costs.at[name_type + " resistive heater", "efficiency"]
                * costs.at[name_type + " resistive heater", "fixed"],
                p_nom=0.5
                * nodal_df[f"{heat_type} resistive heater"][nodes[name]]
                * ratio
                / costs.at[name_type + " resistive heater", "efficiency"],
                build_year=int(grouping_year),
                lifetime=costs.at[costs_name, "lifetime"],
            )

            n.madd(
                "Link",
                nodes[name],
                suffix=f" {name} gas boiler-{grouping_year}",
                bus0=spatial.gas.nodes,
                bus1=nodes[name] + " " + name + " heat",
                bus2="co2 atmosphere",
                carrier=name + " gas boiler",
                efficiency=costs.at[name_type + " gas boiler", "efficiency"],
                efficiency2=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at[name_type + " gas boiler", "efficiency"]
                * costs.at[name_type + " gas boiler", "fixed"],
                p_nom=0.5
                * nodal_df[f"{heat_type} gas boiler"][nodes[name]]
                * ratio
                / costs.at[name_type + " gas boiler", "efficiency"],
                build_year=int(grouping_year),
                lifetime=costs.at[name_type + " gas boiler", "lifetime"],
            )

            n.madd(
                "Link",
                nodes[name],
                suffix=f" {name} oil boiler-{grouping_year}",
                bus0=spatial.oil.nodes,
                bus1=nodes[name] + " " + name + " heat",
                bus2="co2 atmosphere",
                carrier=name + " oil boiler",
                efficiency=costs.at["decentral oil boiler", "efficiency"],
                efficiency2=costs.at["oil", "CO2 intensity"],
                capital_cost=costs.at["decentral oil boiler", "efficiency"]
                * costs.at["decentral oil boiler", "fixed"],
                p_nom=0.5
                * nodal_df[f"{heat_type} oil boiler"][nodes[name]]
                * ratio
                / costs.at["decentral oil boiler", "efficiency"],
                build_year=int(grouping_year),
                lifetime=costs.at[name_type + " gas boiler", "lifetime"],
            )

            # delete links with p_nom=nan corresponding to extra nodes in country
            n.mremove(
                "Link",
                [
                    index
                    for index in n.links.index.to_list()
                    if str(grouping_year) in index and np.isnan(n.links.p_nom[index])
                ],
            )

            # delete links with capacities below threshold
            threshold = snakemake.params.existing_capacities["threshold_capacity"]
            n.mremove(
                "Link",
                [
                    index
                    for index in n.links.index.to_list()
                    if str(grouping_year) in index and n.links.p_nom[index] < threshold
                ],
            )


def set_gas_network(net, fn_gas):
    """
    Sets gas network pipeline capacities according to input gas network mapped to clustered nodes.
    """
    gas_clustered = pd.read_csv(fn_gas, index_col=0)
    # filter out interconnectors
    gas_clustered_de = filter_for_country(gas_clustered, "DE")

    gas_de_i = net.links.loc[
        (net.links.carrier == "gas pipeline")
        & (net.links.bus0.str.startswith("DE"))
        & (net.links.bus1.str.startswith("DE"))].index

    gas_old = net.links.loc[gas_de_i, ["p_nom", "p_nom_min", "p_nom_max"]]

    # if baseyear != year_first:
    #     # set German gas pipelines from custom input to corresponding p_noms
    #     net.links.loc[gas_de_i, ["p_nom_opt", "p_nom", "p_nom_min", "p_nom_max"]] = gas_clustered_de.p_nom.reindex(gas_de_i).fillna(0)

    # then existing gas links need to be rounded up to adjust for some rounding inaccuracy in custom H2 dataset
    net.links.loc[gas_de_i, ["p_nom_opt", "p_nom", "p_nom_min", "p_nom_max"]] = np.ceil(
        net.links.loc[gas_de_i, ["p_nom_opt", "p_nom", "p_nom_min", "p_nom_max"]])

    return gas_old, gas_clustered_de


def set_h2_network(net, fn_new, fn_retro, exchange_year, h2_retrofit_capacity_per_ch4, costs, reoptimise_h2=False):
    """
    Sets H2 network pipeline capacities according to input gas network mapped to clustered nodes.
    """

    h2_retro_clustered = pd.read_csv(fn_retro, index_col=0)
    h2_new_clustered = pd.read_csv(fn_new, index_col=0)

    # substitute H2 network capacities by setting p_nom values
    # also if H2 shall be fixed, the rest of German pipelines, p_nom_min and p_nom_max are set to 0 first
    # p_nom and p_nom_opt are already 0
    carrier = ["H2 pipeline", "H2 pipeline retrofitted"]
    if reoptimise_h2:
        # then same year's base H2 infrastructure will be set
        # set p_nom_min and p_nom_max, so pipelines will be extended exactly to nominal value
        p_noms = ["p_nom_min", "p_nom_max"]
    else:
        # then previous optimization year's infrastructure will be replaced and all p_nom values need to be set
        # since those links are no longer extendable in next period
        p_noms = ["p_nom_opt", "p_nom", "p_nom_min", "p_nom_max"]

    year = str(exchange_year)
    net.links.loc[
        (net.links.carrier.isin(carrier))
        & (net.links.bus0.str.startswith("DE"))
        & (net.links.bus1.str.startswith("DE"))
        & (net.links.index.str.contains(year)),
        p_noms] = 0.0

    # h2_retro_clustered["p_nom"] = np.floor(h2_retro_clustered.p_nom)
    # convert p_nom from CH4 to H2 capacity
    h2_retro_clustered.loc[:, ["p_nom"]] = h2_retro_clustered.loc[:, ["p_nom"]] * h2_retrofit_capacity_per_ch4

    net.links.loc[h2_retro_clustered.index, p_noms] \
        = h2_retro_clustered.p_nom

    # set p_noms for existing h2_new_links
    # if some new pipelines are not in old set -> add new links
    h2_pipes_new = h2_new_clustered[~h2_new_clustered.index.isin(net.links.index)]
    if not h2_pipes_new.empty:
        net.madd(
            "Link",
            h2_pipes_new.index,
            bus0=h2_pipes_new.bus0.values + " H2",
            bus1=h2_pipes_new.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=True,
            length=h2_pipes_new.length.values,
            capital_cost=costs.at["H2 (g) pipeline", "fixed"] * h2_pipes_new.length.values,
            carrier="H2 pipeline",
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
        )
        logger.info(f"Added {h2_pipes_new.index.values} as new links to the network.")

    # set p_noms
    net.links.loc[h2_new_clustered.index, p_noms] \
        = h2_new_clustered.p_nom


def _add_brownfield(n, year):
    """
    # Calls functions of add_brownfield and prepares network for next iteration step.
    # """
    logger.info(f"Preparing brownfield from the file {snakemake.input.network_p}")

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p, override_component_attrs=overrides)

    # if set in config custom H2 network can be added as base infrastructure such as FNB H2 core network
    # gas network is later optimized accordingly
    if snakemake.params.H2_network_custom:
        _ = set_gas_network(n_p, fn_gas)
        if reoptimise_h2:
            # if current year should be reoptimised then year that shall be exchanged is also current year
            exchange_year = year
        else:
            # otherwise last year's H2 infrastructure will be exchanged
            exchange_year = year_p
        set_h2_network(n_p, fn_new, fn_retro, exchange_year, H2_retrofit_capacity_per_CH4, costs, reoptimise_h2)

    # call add_bownfield function from myopic workflow to add previous year's optimization results
    add_brownfield(n, n_p, year, year_p, snakemake.params.threshold_capacity, snakemake.params.H2_retrofit,
                   snakemake.params.H2_retrofit_capacity_per_CH4, build_back_FT_factor, OCGT_H2_retrofitting)


# %%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_existing_baseyear",
            simpl="",
            clusters="180",
            ll="vopt",
            opts="",
            sector_opts="730SEG-T-H-B-I-A-solar+p3-linemaxext10-onwind+p0.4-gas+m2.5",
            planning_horizons=2035,
        )

    logging.basicConfig(level=snakemake.config["logging"]["level"])

    update_config_with_sector_opts(snakemake.config, snakemake.wildcards.sector_opts)

    options = snakemake.params.sector
    build_back_FT_factor = options.get("build_back_FT_factor")
    OCGT_H2_retrofitting = options.get("OCGT_H2_retrofitting")
    H2_retrofit_capacity_per_CH4 = options.get("H2_retrofit_capacity_per_CH4")
    reoptimise_h2 = options.get("reoptimise_h2")
    custom_gas_stores = options.get("custom_gas_stores")
    opts = snakemake.wildcards.sector_opts.split("-")

    baseyear = snakemake.params.baseyear
    if snakemake.params.foresight == "myopic_stepwise":
        planning_horizons = "planning_horizons_all"
    else:
        planning_horizons = "planning_horizons"
    years_all = snakemake.config["scenario"][planning_horizons]
    year_first = years_all[0]
    year_p = years_all[years_all.index(baseyear)-1]

    if snakemake.params.H2_network_custom:
        fn_gas = snakemake.input.clustered_gas_custom
        fn_retro = snakemake.input.clustered_h2_retro_custom
        fn_new = snakemake.input.clustered_h2_new_custom

    overrides = override_component_attrs(snakemake.input.overrides)

    if (snakemake.params.name_base) and (snakemake.params.foresight == "myopic_stepwise"):
        n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
        Nyears = n.snapshot_weightings.generators.sum() / 8760.0
        costs = prepare_costs(
            snakemake.input.costs,
            snakemake.params.costs,
            Nyears,
        )
        _add_brownfield(n, baseyear)

    else:
        n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
        # define spatial resolution of carriers
        spatial = define_spatial(n.buses[n.buses.carrier == "AC"].index, options)

        Nyears = n.snapshot_weightings.generators.sum() / 8760.0
        costs = prepare_costs(
            snakemake.input.costs,
            snakemake.params.costs,
            Nyears,
        )

        add_build_year_to_new_assets(n, baseyear)

        grouping_years_power = snakemake.params.existing_capacities["grouping_years_power"]
        grouping_years_heat = snakemake.params.existing_capacities["grouping_years_heat"]
        add_power_capacities_installed_before_baseyear(
            n, grouping_years_power, costs, baseyear
        )

        # set existing renewable capacities to p_nom min instead of IRENA data

        if "H" in opts:
            time_dep_hp_cop = options["time_dep_hp_cop"]
            ashp_cop = (
                xr.open_dataarray(snakemake.input.cop_air_total)
                .to_pandas()
                .reindex(index=n.snapshots)
            )
            gshp_cop = (
                xr.open_dataarray(snakemake.input.cop_soil_total)
                .to_pandas()
                .reindex(index=n.snapshots)
            )
            default_lifetime = snakemake.params.costs["fill_values"]["lifetime"]
            add_heating_capacities_installed_before_baseyear(
                n,
                baseyear,
                grouping_years_heat,
                ashp_cop,
                gshp_cop,
                time_dep_hp_cop,
                costs,
                default_lifetime,
            )

        # if set in config custom H2 network can be added as base infrastructure such as FNB H2 core network
        # gas network is later optimized accordingly
        if snakemake.params.H2_network_custom:
            gas_old, gas_new = set_gas_network(n, fn_gas)
            set_h2_network(n, fn_new, fn_retro, baseyear, H2_retrofit_capacity_per_CH4, costs, reoptimise_h2)

    if options.get("cluster_heat_buses", False):
        cluster_heat_buses(n)

    if OCGT_H2_retrofitting:
        add_ocgt_retro(n, baseyear)

    if custom_gas_stores:
        fn_gas_stores = snakemake.params.custom_gas_stores
        add_custom_gas_stores(n, fn=fn_gas_stores, costs=costs)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

    n.export_to_netcdf(snakemake.output[0])
