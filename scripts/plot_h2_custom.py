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
from scripts.plot_network import assign_location, rename_techs_tyndp, group_pipes
from add_existing_baseyear import set_gas_network, set_H2_network
from add_brownfield import add_brownfield, add_build_year_to_new_assets, add_ocgt_retro

plt.style.use(["ggplot", "matplotlibrc"])


# def plot_custom(n, regions):
#     fig, ax = plt.subplots(figsize=(15,15), subplot_kw={"projection": ccrs.PlateCarree()})
#     ax.coastlines()
#     ax.add_feature(cartopy.feature.BORDERS, color='black', linewidth=1)
#     ax.add_feature(cartopy.feature.OCEAN, color='azure')
#     ax.add_feature(cartopy.feature.LAND, color='#FAF9F6') #'cornsilk'
#     # ax.set_extent([5, 16, 47, 55])
#     # ax.set_extent([-10, 25, 40, 60])
#     ax.set_extent([-2, 23, 45, 57])

#     onshore_regions.set_crs(crs="EPSG:4326").plot(ax=ax, facecolor="None", edgecolor="grey", alpha=0.2)
#     linewidth_factor = 5e3
#     line_lower_threshold = 750
#     link_widths_retro = h2_pipes_pypsa_retro_gdf.p_nom_opt / linewidth_factor
#     link_widths_retro[h2_pipes_pypsa_retro_gdf.p_nom_opt < line_lower_threshold] = 0.0
#     link_widths_new = h2_pipes_pypsa_new_gdf.p_nom_opt / linewidth_factor
#     link_widths_new[h2_pipes_pypsa_new_gdf.p_nom_opt < line_lower_threshold] = 0.0
#     h2_pipes_pypsa_retro_gdf.set_crs(crs="EPSG:4326").plot(ax=ax, color="#499a9c", label="H2 pipeline repurposed", linewidth=link_widths_retro)
#     h2_pipes_pypsa_new_gdf.set_crs(crs="EPSG:4326").plot(ax=ax, color="#CF9FFF", label="H2 pipeline new", linewidth=link_widths_new)
#     h2_nodes_gdf_de.set_crs(crs="EPSG:4326").plot(ax=ax, color="green", markersize=30, alpha=0.5)
#     # h2_nodes_gdf_de.apply(lambda x: ax.annotate(text=x['name'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1);

#     ########
#     # add legend line color info
#     leg = plt.legend(fontsize=25, loc="upper right")
#     # change the line width for the legend
#     for line in leg.get_lines():
#         line.set_linewidth(2.0)
#     ax.get_figure().add_artist(leg)

#     # add legend line width info
#     sizes = [30, 10, 5, 1]
#     labels = [f"{s} GW" for s in sizes]
#     scale = 1e3 / linewidth_factor
#     sizes = [s * scale for s in sizes]
#     legend_kw = dict(
#         loc="upper center",
#         bbox_to_anchor=(0.8, 0.9),
#         ncol=2,
#         frameon=False,
#         labelspacing=0.8,
#         handletextpad=1,
#         fontsize=25,
#     )

#     patch_kw=dict(color="lightgrey")
#     sizes = np.atleast_1d(sizes)
#     labels = np.atleast_1d(labels)

#     assert len(sizes) == len(labels), "Sizes and labels must have the same length."

#     handles = [plt.Line2D([0], [0], linewidth=s, **patch_kw) for s in sizes]

#     legend = ax.legend(handles, labels, **legend_kw)

#     ax.get_figure().add_artist(legend)
#     ########

#     plt.tight_layout()
#     # plt.savefig(f"data_exchange/{run_name}_H2_network_substituted.png", dpi=300)

def plot_H2_custom(network, regions, path, save_plot=True, show_fig=True):
    n = network.copy()
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

    bus_size_factor = 4e4
    linewidth_factor = 2e4 # 7e3
    # MW below which not drawn
    line_lower_threshold = 62

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    carriers = ["H2 Electrolysis", "H2 Fuel Cell"]

    elec = n.links[n.links.carrier.isin(carriers)].index

    bus_sizes = (
        n.links.loc[elec, "p_nom_opt"].groupby([n.links["bus0"], n.links.carrier]).sum()
        / bus_size_factor
    )

    # make a fake MultiIndex so that area is correct for legend
    bus_sizes.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
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
    n.links = n.links.groupby(level=0).agg({"p_nom_opt": sum, "p_nom": sum, **other_cols})

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

    color_h2_pipe = "#b3f3f4"
    color_retrofit = "#499a9c"

    bus_colors = {"H2 Electrolysis": "#ff29d9", "H2 Fuel Cell": "#805394"}

    n.plot(
        geomap=True,
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
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

    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1),
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
        bbox_to_anchor=(0.23, 1),
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

    colors = [bus_colors[c] for c in carriers] + [color_h2_pipe, color_retrofit]
    labels = carriers + ["H2 pipeline (total)", "H2 pipeline (repurposed)"]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1.13),
        ncol=2,
        frameon=False,
    )

    add_legend_patches(ax, colors, labels, legend_kw=legend_kw)

    ax.set_facecolor("white")

    if show_fig:
        fig.show()
    if save_plot:
        fig.savefig(path, bbox_inches="tight"
        )

if __name__ == "__main__":

    scenario = "default"
    ll = "lvopt"
    sector_opts = "100H-T-H-B-I-A-solar+p3-linemaxext10-onwind+p0.4"
    year = "2035"
    add_brownfield = False
    run_name = f"20240113_100h_defaultdemand_myopic_stepwise_{year}"
    simpl = ""
    clusters = "180"

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

map_opts = snakemake.params.plotting["map"]

if snakemake.params.foresight == "myopic_stepwise":
    planning_horizons = "planning_horizons_all"
else:
    planning_horizons = "planning_horizons"
years_all = snakemake.config["scenario"][planning_horizons]
year_first = years_all[0]
year_p = years_all[years_all.index(int(year)) - 1]
H2_retrofit_capacity_per_CH4 = snakemake.config["sector"].get("H2_retrofit_capacity_per_CH4")

regions = gpd.read_file("resources/regions_onshore_elec_s_180.geojson").set_index("name")

if add_brownfield:
    fn_gas = f"resources/clustered_gas_network_custom_s{simpl}_{clusters}_"+year+".csv"
    fn_retro = f"resources/clustered_h2_network_retro_custom_s{simpl}_{clusters}_"+year+".csv"
    fn_new = f"resources/clustered_h2_network_new_custom_s{simpl}_{clusters}_"+year+".csv"

    overrides = override_component_attrs(snakemake.input.overrides)
    opts = sector_opts.split("-")

    # n_pre = pypsa.Network(f"results/{run_name}/prenetworks-brownfield/elec_s_180_lvopt__100H-T-H-B-I-A-solar+p3-linemaxext10-onwind+p0.4_2045.nc")

    n = pypsa.Network(f"results/{run_name}/postnetworks/elec_s_180_lvopt__{sector_opts}_{year}.nc")
    # remove German H2 pipelines and add German H2 pipelines from pre network
    h2_de = n.links.filter(like="H2 pipeline", axis=0).filter(like=year, axis=0).query("bus0.str.startswith('DE') and bus1.str.startswith('DE')").index
    # n.mremove(
    #     "Link",
    #     h2_de,
    # )

    # h2_pre_df = n_pre.links.filter(like="H2 pipeline", axis=0).filter(like=year, axis=0).query("bus0.str.startswith('DE') and bus1.str.startswith('DE')")
    # n.import_components_from_dataframe(h2_pre_df, "Link")

    gas_old, gas_new = set_gas_network(n, fn_gas)
    set_H2_network(n, fn_new, fn_retro,
                   baseyear=year,
                   year_first=year_first,
                   year_p=year,
                   H2_retrofit_capacity_per_CH4=H2_retrofit_capacity_per_CH4
                   )
    save_path = f"data_exchange_pypsa/{'_'.join(run_name.split('_')[:2])}/results/h2_network_dual_model_{year}.pdf"

else:
    # load pypsa network
    n = pypsa.Network(f"results/{run_name}/prenetworks-brownfield/elec_s_180_{ll}__{sector_opts}_{year}.nc")
    save_path = f"results/{run_name.replace(year, str(year_p))}/maps/h2_network_dual_model_{year_p}.pdf"
    save_path = f"data_exchange_pypsa/{'_'.join(run_name.split('_')[:2])}/results/h2_network_dual_model_{year_p}.pdf"
# n.links.filter(like="H2 pipeline", axis=0).filter(like="2030", axis=0).query("bus0.str.startswith('DE') and bus1.str.startswith('DE')").loc[:,["p_nom", "p_nom_min", "p_nom_max"]]

plot_H2_custom(n, regions, path=save_path, show_fig=False)