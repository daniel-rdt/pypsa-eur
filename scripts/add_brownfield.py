# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Prepares brownfield data from previous planning horizon.
"""

import logging

logger = logging.getLogger(__name__)

import pandas as pd

idx = pd.IndexSlice

import numpy as np
import pypsa
from _helpers import override_component_attrs, update_config_with_sector_opts
from pypsa.io import import_components_from_dataframe


def add_build_year_to_new_assets(n, baseyear):
    """
    Parameters
    ----------
    n : pypsa.Network
    baseyear : int
        year in which optimized assets are built
    """
    # Give assets with lifetimes and no build year the build year baseyear
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        assets = c.df.index[(c.df.lifetime != np.inf) & (c.df.build_year == 0)]
        c.df.loc[assets, "build_year"] = baseyear

        # add -baseyear to name
        rename = pd.Series(c.df.index, c.df.index)
        rename[assets] += "-" + str(baseyear)
        c.df.rename(index=rename, inplace=True)

        # rename time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for attr in n.component_attrs[c.name].index[selection]:
            c.pnl[attr].rename(columns=rename, inplace=True)


def add_brownfield(n, n_p, year, threshold, H2_retrofit, H2_retrofit_capacity_per_CH4, build_back_FT_factor, OCGT_H2_retrofitting):
    logger.info(f"Preparing brownfield for the year {year}")

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]

    for c in n_p.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.mremove(c.name, c.df.index[c.df.lifetime == np.inf])

        # remove assets whose build_year + lifetime < year
        n_p.mremove(c.name, c.df.index[c.df.build_year + c.df.lifetime < year])

        # remove assets if their optimized nominal capacity is lower than a threshold
        # since CHP heat Link is proportional to CHP electric Link, make sure threshold is compatible
        chp_heat = c.df.index[
            (
                c.df[attr + "_nom_extendable"]
                & c.df.index.str.contains("urban central")
                & c.df.index.str.contains("CHP")
                & c.df.index.str.contains("heat")
            )
        ]

        if not chp_heat.empty:
            threshold_chp_heat = (
                threshold
                * c.df.efficiency[chp_heat.str.replace("heat", "electric")].values
                * c.df.p_nom_ratio[chp_heat.str.replace("heat", "electric")].values
                / c.df.efficiency[chp_heat].values
            )
            n_p.mremove(
                c.name,
                chp_heat[c.df.loc[chp_heat, attr + "_nom_opt"] < threshold_chp_heat],
            )

        n_p.mremove(
            c.name,
            c.df.index[
                c.df[attr + "_nom_extendable"]
                & ~c.df.index.isin(chp_heat)
                & (c.df[attr + "_nom_opt"] < threshold)
            ],
        )

        # copy over assets but fix their capacity
        c.df[attr + "_nom"] = c.df[attr + "_nom_opt"]
        c.df[attr + "_nom_extendable"] = False

        # option to build back FT by factor
        if build_back_FT_factor:
            if c.name == "Link":
                logger.info(f"Allow for Fischer-Tropsch build back down to {(1-build_back_FT_factor)*100}% of p_nom_opt.")
                ft_i = (c.df.carrier == "Fischer-Tropsch")
                # set p_nom_extendable back to true to allow for build back
                c.df.loc[ft_i, [attr + "_nom_extendable"]] = True
                # limit to build back by setting p_nom_max to former p_nom_opt value
                c.df.loc[ft_i, [attr + "_nom_max"]] = c.df.loc[ft_i, attr + "_nom_opt"]
                # set build back range according to build_back_FT_factor
                c.df.loc[ft_i, [attr + "_nom_min"]] = c.df.loc[ft_i, attr + "_nom_opt"] * (1 - build_back_FT_factor)
                # stranded assets costs are then capex of build back capacities minus FOM

        n.import_components_from_dataframe(c.df, c.name)

        # copy time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for tattr in n.component_attrs[c.name].index[selection]:
            n.import_series_from_dataframe(c.pnl[tattr], c.name, tattr)

    # deal with gas network
    pipe_carrier = ["gas pipeline"]
    if H2_retrofit:
        # drop capacities of previous year to avoid duplicating
        to_drop = n.links.carrier.isin(pipe_carrier) & (n.links.build_year != year)
        n.mremove("Link", n.links.loc[to_drop].index)

        # subtract the already retrofitted from today's gas grid capacity
        h2_retrofitted_fixed_i = n.links[
            (n.links.carrier == "H2 pipeline retrofitted")
            & (n.links.build_year != year)
        ].index
        gas_pipes_i = n.links[n.links.carrier.isin(pipe_carrier)].index
        CH4_per_H2 = 1 / H2_retrofit_capacity_per_CH4
        fr = "H2 pipeline retrofitted"
        to = "gas pipeline"
        # today's pipe capacity
        pipe_capacity = n.links.loc[gas_pipes_i, "p_nom"]
        # already retrofitted capacity from gas -> H2
        already_retrofitted = (
            n.links.loc[h2_retrofitted_fixed_i, "p_nom"]
            .rename(lambda x: f"{x.split('-2')[0].replace(fr, to)}-{year}")
            .groupby(level=0)
            .sum()
        )
        remaining_capacity = (
            pipe_capacity
            - CH4_per_H2
            * already_retrofitted.reindex(index=pipe_capacity.index).fillna(0)
        )
        # set p_nom and p_nom_max to new p_nom value
        n.links.loc[gas_pipes_i, ["p_nom", "p_nom_max"]] = remaining_capacity
        # also new p_nom_max values for retrofitted H2 pipelines for current year need to be set accordingly
        n.links.loc[
            (n.links.carrier == "H2 pipeline retrofitted") & (n.links.build_year == year),
            ["p_nom_max"]
        ] = remaining_capacity.rename(index=lambda x: x.replace(to, fr)) * H2_retrofit_capacity_per_CH4

        # drop gas pipelines with capacity less than threshold due to infeasible size.
        # Also drop corresponding H2 retro pipeline since can't be retrofitted anymore
        to_drop_gas = (n.links.loc[gas_pipes_i]
                       .query("p_nom < @threshold")
                       .index
                       )
        to_drop_h2_retro = (n.links.loc[gas_pipes_i]
                            .query("p_nom < @threshold")
                            .rename(lambda x: f"{x.replace('gas pipeline', 'H2 pipeline retrofitted')}")
                            .index
                            )
        # drop both set of links
        n.mremove("Link", to_drop_gas.append(to_drop_h2_retro))

    else:
        new_pipes = n.links.carrier.isin(pipe_carrier) & (
            n.links.build_year == year
        )
        n.links.loc[new_pipes, "p_nom"] = 0.0
        n.links.loc[new_pipes, "p_nom_min"] = 0.0

    if OCGT_H2_retrofitting:
        # subtract retrofitted ocgt from existing ocgt
        ocgt_i = n.links.query("carrier == 'OCGT' and ~p_nom_extendable").index
        year_p_str = str(year_p)
        h2_retrofitted_p = n.links.query("carrier == 'OCGT H2 retrofitted' and index.str.endswith(@year_p_str)").index
        fr = "OCGT H2 retrofitted"
        to = "OCGT"
        already_retrofitted = (
            n.links.loc[h2_retrofitted_p, "p_nom"]
            .rename(lambda x: f"{x.replace(fr, to)[:-5]}")
        )
        ocgt_gas_capacity = n.links.loc[ocgt_i, "p_nom"]
        remaining_capacity = (
                ocgt_gas_capacity
                - already_retrofitted.reindex(index=ocgt_gas_capacity.index).fillna(0)
        ).round(6)
        # set p_nom and p_nom_max to new p_nom value
        n.links.loc[ocgt_i, ["p_nom", "p_nom_max"]] = remaining_capacity

        # drop links below threshold
        to_drop_gas = (n.links.loc[ocgt_i]
                       .query("p_nom < @threshold")
                       .index
                       )
        # drop both set of links
        n.mremove("Link", to_drop_gas)

def add_ocgt_retro(n, year):
    """
    Function to add OCGT H2 retrofitting of existing plants.
    """
    logger.info(
        "Add OCGT H2 retrofitting."
    )
    # repurposing cost of OCGT gas to H2 in % investment cost in EUR/MW
    # source: Christidis et al (2023) - H2-Ready-Gaskraftwerke,
    # https://reiner-lemoine-institut.de/wp-content/uploads/2023/11/RLI-Studie-H2-ready_DE.pdf
    retro_factor_ocgt = 0.015
    efficiency_ocgt = 0.39

    # existing OCGT gas boilers
    ocgt_i = n.links.query("carrier == 'OCGT' and ~p_nom_extendable and p_nom > 0").index
    if ocgt_i.empty:
        logger.info(
            "No more OCGT retrofitting potential."
        )
        return

    df = n.links.loc[ocgt_i].copy()
    # adjust bus 0
    df["bus0"] = df.bus0.map(n.buses.location) + " H2"
    # rename carrier and index
    df["carrier"] = df.carrier.apply(
        lambda x: x.replace("OCGT", "OCGT H2 retrofitted")
    )
    df.rename(
        index=lambda x: x.replace("OCGT", "OCGT H2 retrofitted") + f"-{year}", inplace=True
    )
    df.loc[:, "capital_cost"] *= retro_factor_ocgt
    df.loc[:, "efficiency"] = efficiency_ocgt
    # set p_nom_max to OCGT gas p_nom and existing capacity to zero
    df.loc[:, "p_nom_max"] = df["p_nom"]
    df.loc[:, "p_nom"] = 0
    df.loc[:, "p_nom_extendable"] = True
    # set co2 emissions to 0
    df.loc[:, "efficiency2"] = 0.0
    # build_year and lifetime will stay the same as decommissioning of gas plant
    # will also lead to decommissioning of retrofitted use
    # TODO: research extension of lifetime
    # add OCGT H2 to network
    import_components_from_dataframe(n, df, "Link")

# %%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_brownfield",
            simpl="",
            clusters="180",
            opts="",
            ll="vopt",
            sector_opts="100H-T-H-B-I-A-solar+p3-linemaxext10-onwind+p0.4-gas+m2",
            planning_horizons=2040,
        )

    logging.basicConfig(level=snakemake.config["logging"]["level"])

    update_config_with_sector_opts(snakemake.config, snakemake.wildcards.sector_opts)

    logger.info(f"Preparing brownfield from the file {snakemake.input.network_p}")

    year = int(snakemake.wildcards.planning_horizons)
    years_all = snakemake.config["scenario"]["planning_horizons"]
    year_p = years_all[years_all.index(year)-1]

    options = snakemake.params.sector

    build_back_FT_factor = options.get("build_back_FT_factor")
    OCGT_H2_retrofitting = options.get("OCGT_H2_retrofitting")

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p, override_component_attrs=overrides)

    add_brownfield(n, n_p, year, snakemake.params.threshold_capacity, snakemake.params.H2_retrofit,
                   snakemake.params.H2_retrofit_capacity_per_CH4, build_back_FT_factor, OCGT_H2_retrofitting)

    if OCGT_H2_retrofitting:
        add_ocgt_retro(n, year)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
