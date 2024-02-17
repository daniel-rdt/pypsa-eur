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

logger = logging.getLogger(__name__)

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from _helpers import override_component_attrs
from make_summary import assign_carriers
from plot_summary import preferred_order, rename_techs
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
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
        map_opts["boundaries"] = [3, 18, 45, 57]
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

    link_widths_flows = n.links.flow.div(linewidth_factor)
    # cap link width
    link_widths_flows = (
        link_widths_flows
        .where((link_widths_flows.abs() < 1) | (link_widths_flows < 0), 1)
        .where((link_widths_flows.abs() < 1) | (link_widths_flows > 0), -1)
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
        legend_x = -0.15
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
    sizes_str = {20: " or more", 10: "", 5: ""}
    labels = [f"Hydrogen flows of {s} TWh{sizes_str[s]}" for s in sizes]
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
        # fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace("pdf", "png"), bbox_inches="tight", dpi=1000)

def plot_h2_custom(network, regions, path, save_plot=True, show_fig=True):
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
    regions["H2"] = regions["H2"].where(regions["H2"] > 0.1)

    bus_size_factor = 3e8
    linewidth_factor = 9e3 #1.5e4 # 7e3
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
        n.buses.drop(n.buses.index[~n.buses.index.str.startswith("DE")], inplace=True)
        for c in n.iterate_components(n.branch_components):
            c.df.drop(c.df.index[~((c.df.bus0.str.startswith("DE")) & (c.df.bus1.str.startswith("DE")))], inplace=True)
        n.stores.drop(n.stores.index[~n.stores.bus.str.startswith("DE")], inplace=True)
        n.storage_units.drop(n.storage_units.index[~n.storage_units.bus.str.startswith("DE")], inplace=True)
        map_opts["boundaries"] = [3, 18, 45, 57]

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

    bus_colors = {"Supply": "#ff29d9", "Demand": "#805394"}
    tech_colors = snakemake.config["plotting"]["tech_colors"]
    tech_colors_custom = {
        "H2 Electrolysis": "#ff29d9",
        "SMR": "#f073da",
        "SMR CC": "#c251ae",
        "Fischer-Tropsch": "#25c49a",
        "H2 Fuel Cell": "#2d8077",
        "H2 for industry": "#cd4f41",
        "H2 for shipping": "#26b28a",
        "OCGT H2 retrofitted": "#1c404c",
        "Sabatier": "#de9e46",
        "land transport fuel cell": "#c6dfa2",
        "methanolisation": "#238fc4",
    }
    tech_colors.update(tech_colors_custom)

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
        legend_x = -0.15
        legend_y = 0.47
        sizes = [30, 10, 5]
    else:
        legend_x = -0.37
        legend_y = 0.65
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
    labels = ["H2 pipeline (total)", "H2 pipeline (repurposed)"]

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

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(legend_x, 0.84),
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
            "Fischer-Tropsch",
            "H2 Fuel Cell",
            "H2 for industry",
            "H2 for shipping",
            "OCGT H2 retrofitted",
            "land transport fuel cell",
            "Sabatier",
            "methanolisation",
        ]
    )

    preferred_order = preferred_order_demand.intersection(h2_energy_balance.groupby(level=1).sum().index)
    h2_carriers = h2_energy_balance.groupby(level=1).sum().loc[preferred_order].index
    colors = [tech_colors[c] for c in h2_carriers]
    labels = list(h2_carriers)

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
        # fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace("pdf", "png"), bbox_inches="tight", dpi=1000)

if __name__ == "__main__":

    scenario = "low"
    ll = "lvopt"
    sector_opts = "23H-T-H-B-I-A-solar+p3-linemaxext10-onwind+p0.4-gas+m2.5"
    years = ["2030", "2035", "2040", "2045"]
    run = f"20240126_23h_{scenario}demand"
    simpl = ""
    clusters = "180"

    for year in years:
        for only_DE in [True, False]:
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
            save_path = f"data_exchange_pypsa/{'_'.join(run_name.split('_')[:2])}/results/{scenario}"
            os.makedirs(save_path, exist_ok=True)
            # plot and save custom h2 network
            if only_DE:
                spatial = "DE"
            else:
                spatial = "Europe"
            plot_h2_demand_flow(n, regions, path=f"{save_path}/h2_demand_flow_{spatial}_dual_model_{year}.pdf", show_fig=False)
            plot_h2_custom(n, regions, path=f"{save_path}/h2_network_{spatial}_dual_model_{year}.pdf", show_fig=False)
