# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Creates plots for optimised network topologies, including electricity, gas and
hydrogen networks, and regional generation, storage and conversion capacities
built.

This rule plots a map of the network with technology capacities at the
nodes.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from _helpers import override_component_attrs
from plot_summary import preferred_order, rename_techs
from plot_network import assign_location, rename_techs_tyndp, group_pipes
from prepare_sector_network import prepare_costs
from add_existing_baseyear import set_gas_network, set_h2_network
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'sans-serif': ['Computer Modern Sans serif']})
plt.style.use(["ggplot"])

def plot_h2_demand_flow(network, regions, path, save_plot=True, show_fig=True):
    n = network.copy()
    map_opts = map_opts_params.copy()
    if "H2 pipeline" not in n.links.carrier.unique():
        return

    linewidth_factor = 2e7
    # MW below which not drawn
    line_lower_threshold = 1e2
    min_energy = 0
    lim = 50
    link_color = "#499a9c"
    flow_factor = 100

    assign_location(n)

    # get H2 energy balance per node
    carrier = "H2"
    h2_energy_balance = n.statistics.energy_balance(aggregate_bus=False).loc[:, :, carrier].droplevel(0).swaplevel()
    # make a fake MultiIndex so that area is correct for legend
    h2_energy_balance.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)

    to_drop = ["H2 pipeline retrofitted", "H2 pipeline"]
    # drop pipelines and storages from energy balance
    h2_energy_balance.drop(h2_energy_balance.loc[:, to_drop].index, inplace=True)

    regions["H2"] = (
        h2_energy_balance
        .groupby(level=0)
        .sum()
        .div(1e6)  # TWh
        # .mul(-1)  # so demand is positive and supply is negative
    )
    # regions["H2"] = regions["H2"].where(regions["H2"] > 0.1)

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    if only_DE:
        h2_energy_balance = h2_energy_balance.filter(like="DE", axis=0)
        regions = regions.filter(like="DE", axis=0)
        # n.buses.drop(n.buses.index[~n.buses.index.str.startswith("DE")], inplace=True)
        for c in n.iterate_components(n.branch_components):
            c.df.drop(c.df.index[~((c.df.bus0.str.startswith("DE")) | (c.df.bus1.str.startswith("DE")))], inplace=True)
        n.stores.drop(n.stores.index[~n.stores.bus.str.startswith("DE")], inplace=True)
        n.storage_units.drop(n.storage_units.index[~n.storage_units.bus.str.startswith("DE")], inplace=True)
        map_opts["boundaries"] = [4, 17, 46, 56]
        flow_factor = 20
        lim = 8

    # drop all links which are not H2 pipelines
    n.links.drop(
        n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True
    )
    n.links.loc[n.links.p_nom_opt < line_lower_threshold, "p_nom_opt"] = 0.0

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    n.links["flow"] = n.snapshot_weightings.generators @ n.links_t.p0

    positive_order = n.links.bus0 < n.links.bus1
    swap_buses = {"bus0": "bus1", "bus1": "bus0"}
    n.links.loc[~positive_order] = n.links.rename(columns=swap_buses)
    n.links.loc[~positive_order, "flow"] = -n.links.loc[~positive_order, "flow"]
    n.links.index = n.links.apply(lambda x: f"H2 pipeline {x.bus0} -> {x.bus1}", axis=1)
    n.links = n.links.groupby(n.links.index).agg(
        dict(flow="sum", bus0="first", bus1="first", carrier="first", p_nom_opt="sum")
    )

    n.links.flow = n.links.flow.where(n.links.flow.abs() > min_energy)

    proj = ccrs.EqualEarth()
    regions = regions.to_crs(proj.proj4_init)

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": proj})

    link_widths_flows = n.links.flow.div(linewidth_factor).fillna(0)
    # cap link width
    link_widths_flows = (
        link_widths_flows
        .where((link_widths_flows.abs() < 1) | (link_widths_flows < 0), 1)  # 1 where not between 0 and 1 or negative
        .where((link_widths_flows.abs() < 1) | (link_widths_flows > 0), -1)  # -1 where not between 0 and -1 or positive
    )

    n.plot(
        geomap=True,
        bus_sizes=0,
        link_colors=link_color,
        link_widths=link_widths_flows,
        branch_components=["Link"],
        ax=ax,
        flow=pd.concat({"Link": link_widths_flows*flow_factor}),
        **map_opts,
    )

    regions.plot(
        ax=ax,
        column="H2",
        cmap="BrBG",
        linewidths=0,
        legend=True,
        vmax=lim,
        vmin=-lim,
        legend_kwds={
            "label": "Hydrogen balance [TWh]",
            "shrink": 0.7,
            "extend": "max",
        },
    )
    if only_DE:
        legend_x = -0.17
    else:
        legend_x = -0.37

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(legend_x, 1.13),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    sizes = [20, 10, 5]
    sizes_str = {20: " $\geq$", 10: "", 5: ""}
    labels = [f"Hydrogen flows{sizes_str[s]} {s} TWh" for s in sizes]
    scale = 1e6 / linewidth_factor
    sizes = [s * scale for s in sizes]
    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color=link_color),
        legend_kw=legend_kw,
    )

    ax.set_facecolor("white")

    if show_fig:
        fig.show()
    if save_plot:
        fig.savefig(path.replace("pdf", "png"), bbox_inches="tight", dpi=1000)


def plot_h2_custom(network, regions, path, rcm_storages=False, save_plot=True, show_fig=True):
    n = network.copy()
    map_opts = map_opts_params.copy()
    if "H2 pipeline" not in n.links.carrier.unique():
        return

    assign_location(n)

    h2_storage = n.stores.query("carrier == 'H2 Store'")
    regions["H2"] = (
        h2_storage.groupby("bus")
        .sum()
        .rename(index=n.buses.location[h2_storage.groupby("bus").sum().index])
        .e_nom_opt.div(1e6)  # TWh
    )

    # rcm storages
    if rcm_storages:
        # load rcm storages in TWh
        rcm_storages = read_rcm_storages(scenario=scenario, year=year)
        de_i = regions.filter(like="DE", axis=0).index
        # set German storage values to 0
        regions.loc[de_i, "H2"] = 0
        # replace with new values
        regions.loc[rcm_storages.index, "H2"] = rcm_storages.p_nom

    regions["H2"] = regions["H2"].where(regions["H2"] > 0.1)

    bus_size_factor = 3e8
    linewidth_factor = 9e3  #1.5e4 # 7e3
    # MW below which not drawn
    line_lower_threshold = 62
    energy_threshold = 1e3  # MWh

    # get H2 energy balance per node
    carrier = "H2"
    h2_energy_balance = n.statistics.energy_balance(aggregate_bus=False).loc[:, :, carrier].droplevel(0).swaplevel()
    # make a fake MultiIndex so that area is correct for legend
    h2_energy_balance.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)

    to_drop = ["H2 pipeline retrofitted", "H2 pipeline", "H2 Store"]
    # drop pipelines and storages from energy balance
    h2_energy_balance.drop(h2_energy_balance.loc[:, to_drop].index, inplace=True)

    # make demand values positive so they can be plotted with demand
    h2_energy_balance = h2_energy_balance.abs()

    to_drop = h2_energy_balance.index[
        h2_energy_balance < energy_threshold
        ]

    logger.info(
        f"Dropping all H2 supply and demand below {energy_threshold} MWh/a"
    )

    h2_energy_balance = h2_energy_balance.drop(to_drop)

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    if only_DE:
        h2_energy_balance = h2_energy_balance.filter(like="DE", axis=0)
        regions = regions.filter(like="DE", axis=0)
        for c in n.iterate_components(n.branch_components):
            c.df.drop(c.df.index[~((c.df.bus0.str.startswith("DE")) | (c.df.bus1.str.startswith("DE")))], inplace=True)
        neighbor_countries = ("DE", "DK", "SE", "PL", "CZ", "AT", "CH", "FR", "LU", "BE", "NL")
        n.buses.drop(n.buses.index[~n.buses.index.str.startswith(neighbor_countries)], inplace=True)
        n.stores.drop(n.stores.index[~n.stores.bus.str.startswith("DE")], inplace=True)
        n.storage_units.drop(n.storage_units.index[~n.storage_units.bus.str.startswith("DE")], inplace=True)
        map_opts["boundaries"] = [4, 17, 46, 56]

    carriers = ["H2 Electrolysis", "H2 Fuel Cell"]

    elec = n.links[n.links.carrier.isin(carriers)].index

    bus_sizes = (
        n.links.loc[elec, "p_nom_opt"].groupby([n.links["bus0"], n.links.carrier]).sum()
        / bus_size_factor
    )

    bus_sizes = h2_energy_balance / bus_size_factor

    # make a fake MultiIndex so that area is correct for legend
    # drop all links which are not H2 pipelines
    n.links.drop(
        n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True
    )

    h2_new = n.links[n.links.carrier == "H2 pipeline"]
    h2_retro = n.links[n.links.carrier == "H2 pipeline retrofitted"]

    # sum capacity for pipelines from different investment periods
    h2_new = group_pipes(h2_new)

    if not h2_retro.empty:
        h2_retro = (
            group_pipes(h2_retro, drop_direction=True)
            .fillna(0)
        )

    if not h2_retro.empty:
        # create positive order of buses
        positive_order = h2_retro.bus0 < h2_retro.bus1
        h2_retro_p = h2_retro[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        h2_retro_n = h2_retro[~positive_order].rename(columns=swap_buses)
        h2_retro = pd.concat([h2_retro_p, h2_retro_n])

        h2_retro["index_orig"] = h2_retro.index
        h2_retro.index = h2_retro.apply(
            lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
            axis=1,
        )

        retro_w_new_i = h2_retro.index.intersection(h2_new.index)
        h2_retro_w_new = h2_retro.loc[retro_w_new_i]

        retro_wo_new_i = h2_retro.index.difference(h2_new.index)
        h2_retro_wo_new = h2_retro.loc[retro_wo_new_i]
        h2_retro_wo_new.index = h2_retro_wo_new.index_orig.apply(lambda x: x.split('-2')[0])

        to_concat = [h2_new, h2_retro_w_new, h2_retro_wo_new]
        h2_total = pd.concat(to_concat).p_nom_opt.groupby(level=0).sum()

    else:
        h2_total = h2_new.p_nom_opt

    link_widths_total = h2_total / linewidth_factor

    n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)

    # group links by summing up p_nom values and taking the first value of the rest of the columns
    other_cols = dict.fromkeys(n.links.columns.drop(["p_nom_opt", "p_nom"]), "first")
    n.links = n.links.groupby(level=0).agg({"p_nom_opt": "sum", "p_nom": "sum", **other_cols})

    link_widths_total = link_widths_total.reindex(n.links.index).fillna(0.0)
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    retro = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline retrofitted", other=0.0
    )
    link_widths_retro = retro / linewidth_factor
    link_widths_retro[n.links.p_nom_opt < line_lower_threshold] = 0.0

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    proj = ccrs.EqualEarth()
    regions = regions.to_crs(proj.proj4_init)

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": proj})

    color_h2_pipe = "#81e6da"  # "#93e3f5"  # "#b3f3f4"
    color_retrofit = "#499a9c"

    n.plot(
        geomap=True,
        bus_sizes=bus_sizes,
        bus_colors=tech_colors,
        link_colors=color_h2_pipe,
        link_widths=link_widths_total,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    n.plot(
        geomap=True,
        bus_sizes=0,
        link_colors=color_retrofit,
        link_widths=link_widths_retro,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    regions.plot(
        ax=ax,
        column="H2",
        cmap="Blues",
        linewidths=0,
        legend=True,
        vmax=6,
        vmin=0,
        legend_kwds={
            "label": "Hydrogen Storage [TWh]",
            "shrink": 0.7,
            "extend": "max",
        },
    )
    if only_DE:
        legend_x = -0.22
        legend_y = 0.39
        sizes = [30, 10, 5]
    else:
        legend_x = -0.37
        legend_y = 0.57
        sizes = [300, 100, 30]

    labels = [f"{s} TWh" for s in sizes]
    sizes = [s / bus_size_factor * 1e6 for s in sizes]



    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(legend_x + 0.24, 1.01),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [60, 30, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(legend_x, 1.01),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color="lightgrey"),
        legend_kw=legend_kw,
    )

    colors = [color_h2_pipe, color_retrofit]
    labels = ["H2 pipeline constructed", "H2 pipeline retrofitted"]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(legend_x, 1.13),
        frameon=False,
    )

    add_legend_patches(ax, colors, labels, legend_kw=legend_kw)

    preferred_order_supply = pd.Index(
        [
            "H2 Electrolysis",
            "SMR",
            "SMR CC",
        ]
    )
    preferred_order = preferred_order_supply.intersection(h2_energy_balance.groupby(level=1).sum().index)
    h2_carriers = h2_energy_balance.groupby(level=1).sum().loc[preferred_order].index
    colors = [tech_colors[c] for c in h2_carriers]
    labels = list(h2_carriers)

    labels = list(pd.Series(labels).replace(pretty_names))

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(legend_x, legend_y + 0.21),
        frameon=False,
        title=r"\textbf{Supply}",
        alignment="left",
    )

    add_legend_patches(
        ax,
        colors,
        labels,
        legend_kw=legend_kw
    )

    preferred_order_demand = pd.Index(
        [
            "H2 Fuel Cell",
            "OCGT H2 retrofitted",
            "land transport fuel cell",
            "Fischer-Tropsch",
            "Sabatier",
            "H2 for industry",
            "H2 for shipping",
            "methanolisation",
        ]
    )

    preferred_order = preferred_order_demand.intersection(h2_energy_balance.groupby(level=1).sum().index)
    h2_carriers = h2_energy_balance.groupby(level=1).sum().loc[preferred_order].index
    colors = [tech_colors[c] for c in h2_carriers]
    labels = list(h2_carriers)

    labels = list(pd.Series(labels).replace(pretty_names))

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(legend_x, legend_y),
        frameon=False,
        title=r"\textbf{Demand}",
        alignment="left",
    )

    add_legend_patches(
        ax,
        colors,
        labels,
        legend_kw=legend_kw
    )

    ax.set_facecolor("white")

    if show_fig:
        fig.show()
    if save_plot:
        fig.savefig(path, bbox_inches="tight")

def plot_costs(snakemake, n_header):
    cost_df = pd.read_csv(
        snakemake.input.costs, index_col=list(range(3)), header=list(range(n_header))
    )

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    # convert to billions
    df = df / 1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < snakemake.params.plotting["costs_threshold"]]

    logger.info(
        f"Dropping technology with costs below {snakemake.params['plotting']['costs_threshold']} EUR billion per year"
    )
    logger.debug(df.loc[to_drop])

    df = df.drop(to_drop)

    for row in df.sum().items():
        logger.info(f"Total system cost of {round(row[1])} EUR billion per year for {row[0]}")

    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )

    new_columns = df.sum().index.sort_values()
    column_years = df.sum().index.get_level_values("planning_horizon").sort_values()

    fig, ax = plt.subplots(figsize=(12, 8))

    df_plot = df.loc[new_index, new_columns].T.set_index(column_years)

    df_plot.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[tech_colors[i] for i in new_index],
    )

    ax.bar_label(ax.containers[-1], fmt="{:,.0f}")
    ax.tick_params(axis='x', labelrotation=0)
    # for i, total in enumerate(df_plot.sum(axis=1)):
    #     total = round(total, 1)
    #     ax.text(df_plot.index[i], total+4, total, ha="center")

    handles, labels = ax.get_legend_handles_labels()
    labels = list(pd.Series(labels).replace(pretty_names))

    handles.reverse()
    labels.reverse()

    ax.set_ylim([0, snakemake.params.plotting["costs_max"]])
    if only_DE:
        ax.set_ylim([0, snakemake.params.plotting["costs_max"]*0.4])

    ax.set_ylabel("System Cost [EUR billion per year]")

    # after plotting the data, format the labels
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    ax.set_xlabel("")

    ax.grid(axis="x")

    ax.legend(
        handles, labels, ncol=2, loc="upper left", bbox_to_anchor=[1, 1], frameon=False
    )

    fig.savefig(snakemake.output.costs, bbox_inches="tight")


def plot_energy(snakemake, n_header):
    energy_df = pd.read_csv(
        snakemake.input.energy, index_col=list(range(2)), header=list(range(n_header))
    )

    df = energy_df.groupby(energy_df.index.get_level_values(1)).sum()

    # convert MWh to TWh
    df = df / 1e6

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[
        df.abs().max(axis=1) < snakemake.params.plotting["energy_threshold"]
    ]

    logger.info(
        f"Dropping all technology with energy consumption or production below {snakemake.params['plotting']['energy_threshold']} TWh/a"
    )
    logger.debug(df.loc[to_drop])

    df = df.drop(to_drop)

    for row in df.loc[df.values > 0].sum().items():
        logger.info(f"Total energy of {round(row[1])} TWh/a for {row[0]}")

    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )

    new_columns = df.columns.sort_values()
    column_years = df.columns.get_level_values("planning_horizon").sort_values()

    fig, ax = plt.subplots(figsize=(12, 8))

    logger.debug(df.loc[new_index, new_columns])

    df_plot = df.loc[new_index, new_columns].T.set_index(column_years)

    df_plot.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[tech_colors[i] for i in new_index],
    )

    ax.bar_label(ax.containers[-1], fmt="{:,.0f}")
    ax.tick_params(axis='x', labelrotation=0)

    handles, labels = ax.get_legend_handles_labels()
    labels = list(pd.Series(labels).replace(pretty_names))

    handles.reverse()
    labels.reverse()

    ax.set_ylim(
        [
            snakemake.params.plotting["energy_min"],
            snakemake.params.plotting["energy_max"],
        ]
    )

    if only_DE:
        ax.set_ylim(
            [
                snakemake.params.plotting["energy_min"],
                snakemake.params.plotting["energy_max"]*0.4,
            ]
        )

    ax.set_ylabel("Energy [TWh/a]")

    # after plotting the data, format the labels
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    ax.set_xlabel("")

    ax.grid(axis="x")

    ax.legend(
        handles, labels, ncol=2, loc="upper left", bbox_to_anchor=[1, 1], frameon=False
    )

    fig.savefig(snakemake.output.energy, bbox_inches="tight")


def plot_balances(snakemake, n_header, spatial):
    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df = pd.read_csv(
        snakemake.input.balances, index_col=list(range(3)), header=list(range(n_header))
    )

    balances = {i.replace(" ", "_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [
        i for i in balances_df.index.levels[0] if i not in co2_carriers
    ]

    plt.rcParams['font.size'] = 18
    fig, ax = plt.subplots(figsize=(12, 8))

    for k, v in balances.items():
        df = balances_df.loc[v]
        df = df.groupby(df.index.get_level_values(2)).sum()

        # convert MWh to TWh
        df = df / 1e6

        # remove trailing link ports
        df.index = [
            i[:-1]
            if ((i not in ["co2", "NH3", "H2"]) and (i[-1:] in ["0", "1", "2", "3"]))
            else i
            for i in df.index
        ]

        df = df.groupby(df.index.map(rename_techs)).sum()

        to_drop = df.index[
            df.abs().max(axis=1) < snakemake.params.plotting["energy_threshold"] / 100
        ]

        units = "MtCO2/a" if v[0] in co2_carriers else "TWh/a"

        logger.debug(
            f"Dropping technology energy balance smaller than {snakemake.params['plotting']['energy_threshold']/100} {units}"
        )
        logger.debug(df.loc[to_drop])

        df = df.drop(to_drop)

        logger.debug(
            f"Total energy balance for {v} of {round(df.sum().iloc[0],2)} {units}")

        if df.empty:
            continue

        new_index = preferred_order.intersection(df.index).append(
            df.index.difference(preferred_order)
        )

        new_columns = df.columns.sort_values()
        column_years = df.columns.get_level_values("planning_horizon").sort_values()
        df_plot = df.loc[new_index, new_columns].T.set_index(column_years)

        if (k == "oil") and (scenario == "default"):
            df_plot.loc["2045", ["oil"]] = 1e-12
        df_plot.plot(
            kind="bar",
            ax=ax,
            stacked=True,
            color=[tech_colors[i] for i in new_index],
        )

        ax.tick_params(axis='x', labelrotation=0)

        if k in ["H2", "oil"]:
            max = 0
            max_i = 0
            for i, item in enumerate(ax.containers):
                if item.datavalues[0] > max:
                    max = item.datavalues[0]
                    max_i = i
            ax.bar_label(ax.containers[max_i], fmt="{:,.0f}")
            if k == "H2" and only_DE:
                i = {"low": 3, "default": 4, "high": 5}
                ax.bar_label(ax.containers[i[scenario]], fmt="{:,.0f}")
        elif k == "gas":
            if scenario == "high":
                ax.bar_label(ax.containers[-5], fmt="{:,.0f}")
            else:
                ax.bar_label(ax.containers[-4], fmt="{:,.0f}")
        elif k == "methanol":
            ax.bar_label(ax.containers[0], fmt="{:,.0f}")

        handles, labels = ax.get_legend_handles_labels()
        labels = list(pd.Series(labels).replace(pretty_names))

        handles.reverse()
        labels.reverse()

        if v[0] in co2_carriers:
            ax.set_ylabel("CO2 [MtCO2/a]")
        else:
            ax.set_ylabel("Energy [TWh/a]")

        # after plotting the data, format the labels
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        ax.set_xlabel("")

        ax.grid(axis="x")

        ax.legend(
            handles,
            labels,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=[1, 1],
            frameon=False,
        )

        fig.savefig(snakemake.output.balances[:-10] + k + spatial + ".pdf", bbox_inches="tight")

        plt.cla()


def plot_network_links_color_coded(network, weight, label, filename="", percentage=True, show_fig=False, save_plot=True, **kwargs):
    n = network.copy()
    map_opts = map_opts_params.copy()
    if "H2 pipeline" not in n.links.carrier.unique():
        return

    assign_location(n)

    # MW below which not drawn
    line_lower_threshold = 62

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    # h2_energy_balance = h2_energy_balance.filter(like="DE", axis=0)
    # regions = regions.filter(like="DE", axis=0)
    for c in n.iterate_components(n.branch_components):
        c.df.drop(c.df.index[~((c.df.bus0.str.startswith("DE")) | (c.df.bus1.str.startswith("DE")))],
                  inplace=True)
    neighbor_countries = ("DE", "DK", "SE", "PL", "CZ", "AT", "CH", "FR", "LU", "BE", "NL")
    n.buses.drop(n.buses.index[~n.buses.index.str.startswith(neighbor_countries)], inplace=True)
    map_opts["boundaries"] = [6, 15, 47, 55]


    carriers = ["H2 Electrolysis", "H2 Fuel Cell"]

    elec = n.links[n.links.carrier.isin(carriers)].index

    # make a fake MultiIndex so that area is correct for legend
    # drop all links which are not H2 pipelines
    n.links.drop(
        n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True
    )

    n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)

    # group links by summing up p_nom values and taking the first value of the rest of the columns
    other_cols = dict.fromkeys(n.links.columns.drop(["p_nom_opt", "p_nom"]), "first")
    n.links = n.links.groupby(level=0).agg({"p_nom_opt": "sum", "p_nom": "sum", **other_cols})

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    proj = ccrs.EqualEarth()

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": proj})

    link_widths = (n.links.p_nom_opt / n.links.p_nom_opt).fillna(0)
    link_widths[n.links.p_nom_opt < line_lower_threshold] = 0.0

    # filter utilisation for German pipelines
    pipelines_de = pd.Index(pd.concat([pd.Series(weight.filter(like="H2 pipeline DE", axis=0).filter(like="<-> DE", axis=0).index),
                                       pd.Series(weight.filter(like="H2 pipeline retrofitted DE", axis=0).filter(like="<-> DE", axis=0).index)]))
    weight = weight[pipelines_de]
    logger.info(f"Plot color coded network links with min value of {weight.min():,.2f} and max value of {weight.max():,.2f}.")
    if percentage:
        # fill non-existent pipeline values with 100, so that percentage plotting, is scaled to 100
        weight = weight.fillna(100)

    n.plot(
        geomap=True,
        bus_sizes=0,
        link_colors=weight,
        link_cmap=kwargs["cmap"],
        link_widths=link_widths,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    ax.set_facecolor("white")

    if save_plot and percentage:
        save_path = f"data_exchange_pypsa/{'_'.join(run_name.split('_')[:2])}/results/{scenario}"
        fig.savefig(f"{save_path}/{filename.replace('.pdf', '_nolegend.pdf')}", bbox_inches="tight")

    kwargs["vmin"] = 0
    kwargs["vmax"] = weight.round(2).max()
    plot = ax.pcolormesh(weight.mul(100).values[:, None], **kwargs)
    cbar = fig.colorbar(plot, shrink=0.9, extend="max")
    cbar.ax.locator_params(nbins=5)
    cbar.set_label(label=f"{label}", size=16)
    cbar.ax.tick_params(labelsize=16)

    if show_fig:
        fig.show()
    if save_plot:
        save_path = f"data_exchange_pypsa/{'_'.join(run_name.split('_')[:2])}/results/{scenario}"
        fig.savefig(f"{save_path}/{filename}", bbox_inches="tight")

def plot_osc_bar_chart(osc, label, year_label, filename):
    # sort osc
    osc_sorted = osc.sort_values(ascending=False)
    top_i = osc_sorted[:5].index
    osc_sorted_grouped = osc_sorted.rename(lambda x: x.replace("<", "").replace(">", "") if x in top_i else "other").groupby(level=0).sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rcParams['font.size'] = 18
    osc_sorted_grouped.to_frame().T.set_index(pd.Series(year_label)).plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=["#91d7d1", "#22172e", "#5f8f96", "#edba1c", "#c6dfa2", "#e6b89c", "#fe938c", "#4281a4", "#de9e46"]
    )

    # format plot
    ax.tick_params(axis='x', labelrotation=0, labelsize=18)
    ax.set_ylabel(label, size=18)
    ax.tick_params(axis="y", labelsize=18)

    # after plotting the data, format the labels
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    ax.set_xlabel("")

    ax.grid(axis="x")

    ax.legend(
        ncol=1,
        loc="upper left",
        bbox_to_anchor=[1, 1],
        frameon=False,
    )


    save_path = f"data_exchange_pypsa/{'_'.join(run_name.split('_')[:2])}/results/{scenario}"
    fig.savefig(f"{save_path}/{filename}", bbox_inches="tight")

def calculate_economic_indicators(networks_dict):
    df = pd.DataFrame()
    ft_assets = {}
    for label, filename in networks_dict.items():
        logger.info(f"Calculate economic indicators for scenario {label}, using {filename}")

        n = pypsa.Network(filename)

        #####################
        # CAPEX
        #####################
        c = (
            n.links
            .query("carrier == 'H2 pipeline' or carrier == 'H2 pipeline retrofitted'")
            .rename(lambda x: f"{x.split('-2')[0]}")
        )

        h2_retro_i = c.filter(like="H2 pipeline retro", axis=0).index
        h2_new_i = pd.Index(set(c.filter(like="H2 pipeline").index).difference(set(h2_retro_i)))

        # implement rcm cost parameter
        c_new_fix = 1417589.58  # € / km
        c_new = 7946.83 * 24/1000  # € / (km * GWh / d)
        c_retro_fix = 546004.22  # € / km
        c_retro = 3543.04 * 24/1000  # € / (km * GWh / d)

        # annualise values (and add fix OPEX)
        from add_electricity import calculate_annuity

        lifetime_new = c.loc[h2_retro_i, "lifetime"][0]
        lifetime_retro = c.loc[h2_new_i, "lifetime"][0]
        r = 0.02
        fom = {"2030": 3.1667,
               "2035": 2.75,
               "2040": 2.3333,
               "2045": 1.9167,
               }
        def annuity_factor(r, lifetime, fom):
            return calculate_annuity(lifetime, r) + fom / 100

        c_new_ann = annuity_factor(r, lifetime_new, fom[str(label[-1])]) * c_new
        c_new_fix_ann = annuity_factor(r, lifetime_new, fom[str(label[-1])]) * c_new_fix
        c_retro_ann = annuity_factor(r, lifetime_retro, fom[str(label[-1])]) * c_retro
        c_retro_fix_ann = annuity_factor(r, lifetime_retro, fom[str(label[-1])]) * c_retro_fix

        c.loc[:, "p_nom_opt"] = c.p_nom_opt.mask(c.p_nom_opt < threshold, 0)
        c.loc[h2_retro_i, "capex_rcm"] = c.loc[h2_retro_i, "length"] * c_retro_ann
        c.loc[h2_retro_i, "capex_rcm_fix"] = c.loc[h2_retro_i, "length"] * c_retro_fix_ann
        c.loc[h2_new_i, "capex_rcm"] = c.loc[h2_new_i, "length"] * c_new_ann
        c.loc[h2_new_i, "capex_rcm_fix"] = c.loc[h2_new_i, "length"] * c_new_fix_ann
        # set fix cost of pipelines that have no capacity to 0
        c.loc[c.query("p_nom_opt==0").index, "capex_rcm_fix"] = 0
        capital_costs = c.capex_rcm * c.p_nom_opt + c.capex_rcm_fix
        capital_costs_old = c.capital_cost * c.p_nom_opt
        capital_costs = capital_costs.groupby(level=0).sum()
        indicators = pd.concat([capital_costs], keys=["CAPEX"])

        #####################
        # Capacities
        #####################
        caps = c.p_nom_opt.groupby(level=0).sum()
        indicators = pd.concat([indicators,
                   pd.concat([caps], keys=["capacity_MW"])])

        #####################
        # utilisation
        #####################
        # yearly flow in MWh
        flow_yearly = (
            n.links_t.p0
            .filter(like="H2 pipeline")
            .rename(lambda x: f"{x.split('-2')[0]}", axis=1)
            .multiply(n.snapshot_weightings.generators, axis=0)
            .abs().sum()
            .groupby(level=0).sum()  # group over different build years
        )
        flow_yearly_directed = (
            n.links_t.p0
            .filter(like="H2 pipeline")
            .rename(lambda x: f"{x.split('-2')[0]}", axis=1)
            .multiply(n.snapshot_weightings.generators, axis=0)
            .sum()
            .groupby(level=0).sum()  # group over different build years
        )
        # max flow in MWh/h
        flow_max = (
            n.links_t.p0
            .filter(like="H2 pipeline")
            .rename(lambda x: f"{x.split('-2')[0]}", axis=1)
            .abs().max()
            .groupby(level=0).sum()  # group over different build years
        )
        utilisation_yearly = (
            flow_yearly.div(
                c.p_nom_opt
                .groupby(level=0)
                .sum()
                .multiply(n.snapshot_weightings.generators.sum())
            ).replace(np.inf, 0)  # replace inf with 0 since if no p_nom_opt, then division by 0
            .round(6)
        )
        utilisation_max = (
            np.floor(flow_max).div(
                np.floor(c.p_nom_opt
                         .groupby(level=0)
                         .sum())
            ).replace(np.inf, 0)  # replace inf with 0 since if no p_nom_opt, then division by 0
            .round(2)
        )


        if label[-1] == 2045:
            # calculate pipeline lengths of pipelines with utilisation below 5% and capacity below 5 GW
            # for 2045
            lowest_utilisation_i = utilisation_max[(~utilisation_max.isna()) & (utilisation_max > 0) & (utilisation_max < 0.05)].index
            lowest_utilisation = (
                c.filter(like="DE", axis=0)
                .query("p_nom_opt > 0")
                .groupby(level=0)
                .aggregate({"p_nom_opt": "sum", "length": "mean"})
                .loc[lowest_utilisation_i]
                .query("p_nom_opt < 5000").sum()
            )
            lowest_utilisation.to_csv(f"{path}/lowest_utilisation_network_length_2045.csv")

        indicators = pd.concat([indicators,
                                pd.concat([flow_yearly], keys=["flow_yearly"]),
                                pd.concat([flow_yearly_directed], keys=["flow_yearly_directed"]),
                                pd.concat([flow_max], keys=["flow_max"]),
                                pd.concat([utilisation_yearly], keys=["utilisation_yearly"]),
                                pd.concat([utilisation_max], keys=["utilisation_max"])])

        # plot utilisation
        kwargs = {"vmin": 0, "vmax": 100, "cmap": "coolwarm"}
        plot_network_links_color_coded(n, utilisation_yearly*100, label="Yearly Utilisation [\%]",
                                       filename=f"h2_pipeline_yearly_utilisation_de_clustered_{label[-1]}.pdf",
                                       **kwargs)
        plot_network_links_color_coded(n, utilisation_max*100, label="Max Utilisation [\%]",
                                       filename=f"h2_pipeline_max_utilisation_de_clustered_{label[-1]}.pdf",
                                       **kwargs)

        #####################
        # LCOHT & OSC
        #####################
        lcoht = capital_costs.div(flow_yearly).replace(np.inf, 0)  # replace inf with 0 since if no flow, then division by 0
        osc = c.p_nom_opt.groupby(level=0).sum().mul(1 - utilisation_max).mul(c.length.groupby(level=0).first())  # length weighted OSC in MWkm
        indicators = pd.concat([indicators,
                                pd.concat([lcoht], keys=["LCOHT"]),
                                pd.concat([osc], keys=["OSC"])])
        # plot osc
        kwargs = {"cmap": "BuPu"}
        plot_network_links_color_coded(n, osc.div(1e6), label="Oversize-Capacity (OSC) [TWkm]",
                                       percentage=False,
                                       filename=f"h2_pipeline_OSC_de_clustered_{label[-1]}.pdf",
                                       **kwargs)
        plot_osc_bar_chart(osc.div(1e6),
                           label="Oversize-Capacity (OSC) [TWkm]",
                           year_label=label[-1],
                           filename=f"h2_pipeline_total_OSC_de_stacked_{label[-1]}.pdf")

        df = df.reindex(index=indicators.index)
        df[f"{scenario}_{label[-1]}"] = indicators

        #####################
        # Stranded assets Fischer Tropsch
        #####################
        ft_assets[label[-1]] = (
            n.links.filter(like="Fischer-Tropsch", axis=0).rename(lambda x: f"{x.split('-2')[0]}")
            .groupby(level=0)
            .sum()
        )

    stranded_assets = pd.DataFrame({
        year: ft_assets[int(years[i])].p_nom_opt - ft_assets[int(year)].p_nom_opt for i, year in enumerate(years[1:])
    })
    # positive value means stranded assets, since previous year had more optimal capacity for node

    stranded_capex_per_MW = (
            n.links.filter(like="Fischer-Tropsch", axis=0).rename(lambda x: f"{x.split('-2')[0]}")
            .capital_cost
            .groupby(level=0)
            .first()
            .div(1 + (annuity_factor(r, 20, 0) / fom["2045"]))  # calculate out fix opex
        )
    stranded_assets_cost = stranded_assets.mul(stranded_capex_per_MW, axis=0).mul(20)
    stranded_assets_cost[stranded_assets_cost < 0] = 0
    stranded_assets_cost.to_csv(f"{path}/stranded_assets_cost_ft.csv")
    stranded_assets_cost_sum = stranded_assets_cost.sum().div(1e6).to_frame().T.set_index(pd.Series("Stranded assets cost [EUR million]"))
    stranded_assets_cost_sum.to_csv(f"{path}/stranded_assets_cost_ft_aggregated.csv")

    return df

def read_rcm_flow(scenario, year):
    df = pd.read_csv(
        f"data_exchange/20240201_730SEG_{scenario}_{year}/P_Trp_H2_{year}.csv",
        index_col=0
    ).rename(columns={"From_Node": "bus0", "To_Node": "bus1"})

    # create positive order of buses
    positive_order = df.bus0 < df.bus1
    df_p = df[positive_order]
    swap_buses = {"bus0": "bus1", "bus1": "bus0"}
    df_n = df[~positive_order].rename(columns=swap_buses)
    # for swapped buses, also swap flow direction
    df_n.iloc[:, 5:] = ((df_n.iloc[:, 5:]*(-1))
                        .mask(df_n.iloc[:, 5:].abs() == 0, 0.0)  # make zeros positive zeros
                        )
    # concat with positive order buses
    df = pd.concat([df_p, df_n])

    # filter for h2 pipelines
    h2_retro = df.query("Retrof_Status == 1 and not index.str.startswith('CP')")
    h2_new = df.query("Retrof_Status == 1 and index.str.startswith('CP')")

    # rename indices to pypsa naming and drop columns that are not needed anymore. Then aggregate parallel pipelines
    h2_retro.index = h2_retro.apply(
        lambda x: f"H2 pipeline retrofitted {x.bus0} <-> {x.bus1}-{x.build_year}",
        axis=1,
    )
    h2_retro.drop(columns=["bus0", "bus1", "H2", "Retrof_Status", "build_year"], inplace=True)
    # aggregate parallel pipelines
    h2_retro_grouped = h2_retro.groupby(level=0).sum()
    h2_new.index = h2_new.apply(
        lambda x: f"H2 pipeline {x.bus0} <-> {x.bus1}-{x.build_year}",
        axis=1,
    )
    h2_new.drop(columns=["bus0", "bus1", "H2", "Retrof_Status", "build_year"], inplace=True)
    # aggregate parallel pipelines
    h2_new_grouped = h2_new.groupby(level=0).sum()

    # concat both tables in to one
    flow = pd.concat([h2_retro_grouped, h2_new_grouped]).T

    # set index to datetime index
    yearly_i = pd.date_range(f'2013-01-01', f'2013-12-31', freq='1D', name="snapshot")
    flow.index = yearly_i

    # turn from GWh/d to MW
    flow *= 1000/24

    return flow

def read_rcm_storages(scenario, year):
    return pd.read_csv(
        f"data_exchange/20240201_730SEG_{scenario}_{year}/H2_storages_{year}.csv",
            index_col=0,
    ).rename(columns={"cap_slv_[GWh]": "p_nom"}).div(1e3)  # convert from GWh to TWh


if __name__ == "__main__":

    scenario = "default"
    ll = "lvopt"
    sector_opts = "730SEG-T-H-B-I-A-solar+p3-linemaxext10-onwind+p0.4-gas+m2.5"
    years = ["2030", "2035", "2040", "2045"]
    run = f"20240201_730SEG_{scenario}demand"
    simpl = ""
    clusters = "180"

    # custom tech colors
    tech_colors_custom = {
        "H2 Electrolysis": "#ff29d9",
        "SMR": "#f073da",
        "SMR CC": "#c251ae",
        "Fischer-Tropsch": "#25c49a",
        "H2 Fuel Cell": "#2d8077",
        "H2 for industry": "#cd4f41",
        "H2 for shipping": "#238fc4",
        "OCGT H2 retrofitted": "#1c404c",
        "Sabatier": "#de9e46",
        "methanation": "#de9e46",
        "land transport fuel cell": "#c6dfa2",
        "methanolisation": "#edba1c",
    }

    # choose pretty names
    pretty_names = {
        "H2 Electrolysis": "H2 electrolysis",
        "H2 pipeline": "H2 pipeline constructed",
        "H2 pipeline retrofitted": "H2 pipeline retrofitted",
        "SMR": "SMR",
        "SMR CC": "SMR CC",
        "Fischer-Tropsch": "Fischer-Tropsch process",
        "H2 Fuel Cell": "H2 fuel cell",
        "H2 for industry": "Industry H2 demand",
        "H2 for shipping": "Shipping H2 demand",
        "OCGT H2 retrofitted": "OCGT H2 retrofitted",
        "Sabatier": "Methanation (Sabatier)",
        "methanation": "Methanation (Sabatier)",
        "land transport fuel cell": "Land transport H2 demand",
        "methanolisation": "Methanol synthesis",
        "offshore wind (AC)": "Offshore Wind (AC)",
        "offshore wind (DC)": "Offshore Wind (DC)",
        "offwind-ac": "Offshore Wind (AC)",
        "offwind-dc": "Offshore Wind (DC)",
        "offshore wind": "Offshore Wnd",
        "onwind": "Onshore Wind",
        "onshore wind": "Onshore Wind",
        "solar PV": "Solar PV (utility)",
        "solar": "Solar PV (utility)",
        "solar rooftop": "Solar PV (rooftop)",
        "hydroelectricity": "Hydroelectricity",
        "uranium": "Uranium",
        "solid biomass": "Solid biomass",
        "solid biomass for industry": "Industry biomass demand",
        "solid biomass for industry CC": "Industry biomass demand CC",
        "gas for industry": "Industry methane demand",
        "gas for industry CC": "Industry methane demand CC",
        "shipping oil": "Shipping oil demand",
        "shipping methanol": "Shipping methanol demand",
        "oil emissions": "Oil emissions",
        "shipping oil emissions": "Shipping oil emissions",
        "shipping methanol emissions": "Shipping methanol emissions",
        "process emissions": "Process emissions",
        "process emissions CC": "Process emissions CC",
        "agriculture machinery oil emissions": "Agriculture machinery oil emissions",
        "gas": "Methane",
        "oil": "Oil",
        "coal": "Coal",
        "oil boiler": "Oil boiler",
        "gas boiler": "Gas boiler",
        "nuclear": "Nuclear",
        "lignite": "Lignite",
        "land transport oil": "Land transport oil demand",
        "land transport EV": "Land transport EV",
        "naphtha for industry": "Industry naphtha demand",
        "low-temperature heat for industry": "Industry low-temperature heat demand",
        "kerosene for aviation": "Aviation kerosene demand",
        "industry electricity": "Industry electricity demand",
        "heat": "Heat demand",
        "electricity": "Electricity demand",
        "co2": "CO2 emissions",
        "biomass boiler": "Biomass boiler",
        "agriculture machinery oil": "Agriculture machinery oil demand",
        "agriculture heat": "Agriculture heat demand",
        "agriculture electricity": "Agriculture electricity demand",
        "hot water storage": "Hot water storage",
        "resistive heater": "Resistive heater",
        "air heat pump": "Heat pump (air)",
        "ground heat pump": "Heat pump (ground)",
        "electricity distribution grid": "Electricity distribution grid",
        "transmission lines": "Transmission lines",
    }

    for year in years:
        # set run name for year
        run_name = f"{run}_myopic_stepwise_{year}"

        if "snakemake" not in globals():
            from _helpers import mock_snakemake

            snakemake = mock_snakemake(
                "plot_network",
                simpl=simpl,
                opts="",
                clusters=clusters,
                ll=ll,
                sector_opts=sector_opts,
                planning_horizons=year,
            )

        # define tech colors and update with custom
        tech_colors = snakemake.config["plotting"]["tech_colors"]
        tech_colors.update(tech_colors_custom)

        # get threshold capacity
        threshold = snakemake.config["existing_capacities"]["threshold_capacity"]

        logging.basicConfig(level=snakemake.config["logging"]["level"])

        map_opts_params = snakemake.params.plotting["map"]
        H2_retrofit_capacity_per_CH4 = snakemake.config["sector"].get("H2_retrofit_capacity_per_CH4")
        regions = gpd.read_file("resources/regions_onshore_elec_s_180.geojson").set_index("name")

        fn_gas = f"results/{run_name}/h2_networks_custom/clustered_gas_network_custom_s{simpl}_{clusters}_"+year+".csv"
        fn_retro = f"results/{run_name}/h2_networks_custom/clustered_h2_network_retro_custom_s{simpl}_{clusters}_"+year+".csv"
        fn_new = f"results/{run_name}/h2_networks_custom/clustered_h2_network_new_custom_s{simpl}_{clusters}_"+year+".csv"

        overrides = override_component_attrs(snakemake.input.overrides)
        opts = sector_opts.split("-")

        n = pypsa.Network(f"results/{run_name}/postnetworks/elec_s_180_lvopt__{sector_opts}_{year}.nc")
        Nyears = n.snapshot_weightings.generators.sum() / 8760.0
        costs = prepare_costs(
            f"data/costs_{year}.csv",
            snakemake.config["costs"],
            Nyears,
        )

        # get index of current years German H2 pipelines to replace
        h2_de = (n.links
                 .filter(like="H2 pipeline", axis=0)
                 .filter(like=year, axis=0)
                 .query("bus0.str.startswith('DE') and bus1.str.startswith('DE')")
                 .index
                 )
        gas_old, gas_new = set_gas_network(n, fn_gas)
        set_h2_network(n, fn_new, fn_retro,
                       exchange_year=year,
                       h2_retrofit_capacity_per_ch4=H2_retrofit_capacity_per_CH4,
                       reoptimise_h2=False,
                       costs=costs,
                       )

        # exchange with new flow values for German H2 network
        flow_de = read_rcm_flow(scenario, year)
        de_i = pd.Index(pd.concat([pd.Series(n.links_t.p0.filter(like="H2 pipeline DE").filter(like="<-> DE").columns),
                                   pd.Series(n.links_t.p0.filter(like="H2 pipeline retrofitted DE").filter(like="<-> DE").columns)]))
        n.links_t.p0.loc[:, de_i] = 0
        n.links_t.p1.loc[:, de_i] = 0

        # reindex flow to pypsa index
        snapshots_yearly = pd.date_range(f'2013-01-01', f'2013-12-31 23:00:00', freq='1H', name="snapshot")
        snapshots_pypsa = n.links_t.p0.loc[:, de_i].index

        flow_de = flow_de.reindex(snapshots_yearly).ffill().reindex(snapshots_pypsa)  # just forward fill with values since MW in hourly is MWh
        n.links_t.p0.loc[:, flow_de.columns] = flow_de
        n.links_t.p1.loc[:, flow_de.columns] = -flow_de

        # save network with custom h2 for later use
        logger.info("Custom H2 network was added to postnetwork and new adjusted postnetwork was saved.")
        n.export_to_netcdf(f"results/{run_name}/postnetworks/elec_s_180_lvopt__{sector_opts}_{year}_custom.nc")
        # save network with only_DE for later use
        n_de = n.copy()
        # drop non German buses, links, stores, storages
        n_de.buses.drop(n_de.buses.index[~(n_de.buses.index.str.startswith("DE") | n_de.buses.index.str.startswith("EU"))], inplace=True)
        bus_map = n.buses.carrier == "H2"
        bus_map.at[""] = False
        for c in n_de.iterate_components(n_de.branch_components):
            c.df.drop(c.df.index[~((c.df.bus0.str.startswith("DE") | c.df.bus0.str.startswith("EU")) & ((c.df.bus1.str.startswith("DE") | c.df.bus1.str.startswith("EU"))))], inplace=True)
            for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
                items = c.df["bus" + str(end)].index
                if len(items) == 0:
                    continue
                c.pnl["p" + end] = c.pnl["p" + end][items]
        for c in n_de.iterate_components(n_de.one_port_components):
            c.df.drop(c.df.index[~(c.df.bus.str.startswith("DE") | c.df.bus.str.startswith("EU"))], inplace=True)
            items = c.df.bus.index
            if len(items) == 0:
                continue
            c.pnl.p = c.pnl.p[items]

        # n_de.stores.drop(n_de.stores.index[~n_de.stores.bus.str.startswith("DE")], inplace=True)
        # n_de.storage_units.drop(n_de.storage_units.index[~n_de.storage_units.bus.str.startswith("DE")], inplace=True)

        logger.info("Custom H2 network was added to postnetwork and new adjusted German postnetwork was saved.")
        n_de.export_to_netcdf(f"results/{run_name}/postnetworks/elec_s_180_lvopt__{sector_opts}_{year}_custom_DE.nc")

        # create save path for figures
        save_path = f"data_exchange_pypsa/{'_'.join(run_name.split('_')[:2])}/results/{scenario}"
        os.makedirs(save_path, exist_ok=True)

        for only_DE in [True, False]:
            # plot and save custom h2 network
            if only_DE:
                spatial = "DE"
            else:
                spatial = "Europe"
            plot_h2_demand_flow(n, regions,
                                path=f"{save_path}/h2_demand_flow_{spatial}_dual_model_{year}.pdf",
                                show_fig=False)
            plot_h2_custom(n, regions,
                           path=f"{save_path}/h2_network_{spatial}_dual_model_{year}.pdf",
                           show_fig=False,
                           rcm_storages=True)


    ####################
    # Updating summaries
    ####################
    from make_summary import make_summaries, calculate_cumulative_cost, assign_locations, assign_carriers
    for only_DE in [True, False]:
        # plot and save custom h2 network
        if only_DE:
            spatial = "_DE"
        else:
            spatial = ""
        # define all network from the run in dict to create summaries
        networks_dict = {
            (clusters, ll, sector_opts,
             int(year)): f"results/{run_name.replace(years[-1], year)}/postnetworks/elec_s_180_lvopt__{sector_opts}_{year}_custom{spatial}.nc"
            for year in years
        }
        df = make_summaries(networks_dict)

        df["metrics"].loc["total costs"] = df["costs"].sum()

        path = f"data_exchange_pypsa/{'_'.join(run_name.split('_')[:2])}/results/csvs/{scenario}"
        os.makedirs(path, exist_ok=True)
        for key in df:
            df[key].to_csv(f"{path}/{key}{spatial}.csv")

        cumulative_cost = calculate_cumulative_cost(df, [int(year) for year in years])
        cumulative_cost.to_csv(f"{path}/cumulative_cost{spatial}.csv")

        n_header = 4

        snakemake = mock_snakemake(
            "plot_summary",
            simpl=simpl,
            opts="",
            clusters=clusters,
            ll=ll,
            sector_opts=sector_opts,
            planning_horizons=year,
        )
        # set input and output paths for costs, energy and balances
        # inputs
        snakemake.input.costs = path + f"/costs{spatial}.csv"
        snakemake.input.energy = path + f"/energy{spatial}.csv"
        snakemake.input.balances = path + f"/supply_energy{spatial}.csv"
        # outputs
        path_graphs = path.replace("/csvs", "") + "/graphs"
        os.makedirs(path_graphs, exist_ok=True)
        snakemake.output.costs = path_graphs + f"/costs{spatial}.pdf"
        snakemake.output.energy = path_graphs + f"/energy{spatial}.pdf"
        snakemake.output.balances = path_graphs + f"/balances-energy.pdf"

        plot_costs(snakemake, n_header)
        plot_energy(snakemake, n_header)
        plot_balances(snakemake, n_header, spatial=spatial)

    networks_dict = {
        (clusters, ll, sector_opts,
         int(year)): f"results/{run_name.replace(years[-1], year)}/postnetworks/elec_s_180_lvopt__{sector_opts}_{year}_custom.nc"
        for year in years
    }
    economic_indicators = calculate_economic_indicators(networks_dict)
    # total lcoht is the same as capacity weighted mean lcoht
    lcoht_total = economic_indicators.loc["CAPEX", :].sum() / economic_indicators.loc["flow_yearly", :].sum()
    # total utilisation is the same as capacity weighted mean utilisation
    utilisation_total_max = economic_indicators.loc["flow_max", :].sum() / (economic_indicators.loc["capacity_MW", :].sum()).round(2)

    # add capacity weighted indicators
    economic_indicators = pd.concat([economic_indicators,
                                     pd.concat([lcoht_total.to_frame().T.set_index(pd.Index(["total"]))], keys=["LCOHT_total"]),
                                     pd.concat([utilisation_total_max.to_frame().T.set_index(pd.Index(["total"]))], keys=["utilisation_total_max"])])
    economic_indicators.to_csv(f"{path}/economic_indicators.csv")

    # save aggregated economic indicators
    economic_indicators_aggregated = economic_indicators.groupby(level=0).agg(
        Max_2030=pd.NamedAgg(column=f"{scenario}_2030", aggfunc='max'),
        Min_2030=pd.NamedAgg(column=f"{scenario}_2030", aggfunc='min'),
        Sum_2030=pd.NamedAgg(column=f"{scenario}_2030", aggfunc='sum'),
        Mean_2030=pd.NamedAgg(column=f"{scenario}_2030", aggfunc='mean'),
        Median_2030=pd.NamedAgg(column=f"{scenario}_2030", aggfunc='median'),
        Max_2035=pd.NamedAgg(column=f"{scenario}_2035", aggfunc='max'),
        Min_2035=pd.NamedAgg(column=f"{scenario}_2035", aggfunc='min'),
        Sum_2035=pd.NamedAgg(column=f"{scenario}_2035", aggfunc='sum'),
        Mean_2035=pd.NamedAgg(column=f"{scenario}_2035", aggfunc='mean'),
        Median_2035=pd.NamedAgg(column=f"{scenario}_2035", aggfunc='median'),
        Max_2040=pd.NamedAgg(column=f"{scenario}_2040", aggfunc='max'),
        Min_2040=pd.NamedAgg(column=f"{scenario}_2040", aggfunc='min'),
        Sum_2040=pd.NamedAgg(column=f"{scenario}_2040", aggfunc='sum'),
        Mean_2040=pd.NamedAgg(column=f"{scenario}_2040", aggfunc='mean'),
        Median_2040=pd.NamedAgg(column=f"{scenario}_2040", aggfunc='median'),
        Max_2045=pd.NamedAgg(column=f"{scenario}_2045", aggfunc='max'),
        Min_2045=pd.NamedAgg(column=f"{scenario}_2045", aggfunc='min'),
        Sum_2045=pd.NamedAgg(column=f"{scenario}_2045", aggfunc='sum'),
        Mean_2045=pd.NamedAgg(column=f"{scenario}_2045", aggfunc='mean'),
        Median_2045=pd.NamedAgg(column=f"{scenario}_2045", aggfunc='median'),
    )
    economic_indicators_aggregated.to_csv(f"{path}/economic_indicators_aggregated.csv")

    # only for Germany
    de_i = (
        economic_indicators
        .filter(like="H2 pipeline DE", axis=0)
        .filter(like="<-> DE", axis=0)
        .index
        .union(
            economic_indicators
            .filter(like="H2 pipeline retrofitted DE", axis=0)
            .filter(like="<-> DE", axis=0)
            .index
        )
    )
    economic_indicators_de = economic_indicators.loc[de_i]
    lcoht_total_de = economic_indicators_de.loc["CAPEX", :].sum() / economic_indicators_de.loc["flow_yearly", :].sum()
    # total utilisation is the same as capacity weighted mean utilisation
    utilisation_total_max_de = economic_indicators_de.loc["flow_max", :].sum() / (economic_indicators_de.loc["capacity_MW", :].sum()).round(2)

    economic_indicators_de = pd.concat([economic_indicators_de,
                                        pd.concat([lcoht_total_de.to_frame().T.set_index(pd.Index(["total"]))], keys=["LCOHT_total"]),
                                        pd.concat([utilisation_total_max_de.to_frame().T.set_index(pd.Index(["total"]))], keys=["utilisation_total_max"])])
    economic_indicators_aggregated_de = economic_indicators_de.groupby(level=0).agg(
        Max_2030=pd.NamedAgg(column=f"{scenario}_2030", aggfunc='max'),
        Min_2030=pd.NamedAgg(column=f"{scenario}_2030", aggfunc='min'),
        Sum_2030=pd.NamedAgg(column=f"{scenario}_2030", aggfunc='sum'),
        Mean_2030=pd.NamedAgg(column=f"{scenario}_2030", aggfunc='mean'),
        Median_2030=pd.NamedAgg(column=f"{scenario}_2030", aggfunc='median'),
        Max_2035=pd.NamedAgg(column=f"{scenario}_2035", aggfunc='max'),
        Min_2035=pd.NamedAgg(column=f"{scenario}_2035", aggfunc='min'),
        Sum_2035=pd.NamedAgg(column=f"{scenario}_2035", aggfunc='sum'),
        Mean_2035=pd.NamedAgg(column=f"{scenario}_2035", aggfunc='mean'),
        Median_2035=pd.NamedAgg(column=f"{scenario}_2035", aggfunc='median'),
        Max_2040=pd.NamedAgg(column=f"{scenario}_2040", aggfunc='max'),
        Min_2040=pd.NamedAgg(column=f"{scenario}_2040", aggfunc='min'),
        Sum_2040=pd.NamedAgg(column=f"{scenario}_2040", aggfunc='sum'),
        Mean_2040=pd.NamedAgg(column=f"{scenario}_2040", aggfunc='mean'),
        Median_2040=pd.NamedAgg(column=f"{scenario}_2040", aggfunc='median'),
        Max_2045=pd.NamedAgg(column=f"{scenario}_2045", aggfunc='max'),
        Min_2045=pd.NamedAgg(column=f"{scenario}_2045", aggfunc='min'),
        Sum_2045=pd.NamedAgg(column=f"{scenario}_2045", aggfunc='sum'),
        Mean_2045=pd.NamedAgg(column=f"{scenario}_2045", aggfunc='mean'),
        Median_2045=pd.NamedAgg(column=f"{scenario}_2045", aggfunc='median'),
    )
    economic_indicators_aggregated_de.to_csv(f"{path}/economic_indicators_aggregated_de.csv")

    # calculate net import statistics for Germany
    interconnectors_i = (economic_indicators
                         .filter(like="DE", axis=0)
                         .index.get_level_values(1).unique()
                         .difference(de_i.get_level_values(1)))
    net_interconnector_flows = economic_indicators.loc["flow_yearly_directed", interconnectors_i, :]
    # positive flow direction should be uniformly defined as flow into Germany
    # filter for negative flow direction
    new_n_i = net_interconnector_flows.filter(like="pipeline DE", axis=0).index
    retro_n_i = net_interconnector_flows.filter(like="pipeline retrofitted DE", axis=0).index
    # TODO: rename index for these pipelines?
    # change to positive flow direction
    net_interconnector_flows.loc[new_n_i.union(retro_n_i)] *= (-1)
    total_net_interconnector_flows = net_interconnector_flows.sum().div(1e6).to_frame().T.set_index(pd.Series("Net imports [TWh]"))  # in TW
    total_imports = net_interconnector_flows[net_interconnector_flows > 0].sum().div(1e6).to_frame().T.set_index(pd.Series("Imports [TWh]"))
    total_exports = net_interconnector_flows[net_interconnector_flows < 0].sum().div(-1e6).to_frame().T.set_index(pd.Series("Exports [TWh]"))
    transit_shares = total_exports.reset_index(drop=True).div(total_imports.reset_index(drop=True)).mul(1e2).set_index(pd.Series("Export shares [%]"))
    pd.concat([total_imports, total_exports, total_net_interconnector_flows, transit_shares]).to_csv(f"{path}/total_net_imports_de.csv")

