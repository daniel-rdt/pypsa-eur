import logging

logger = logging.getLogger(__name__)

from _helpers import update_config_with_sector_opts
import geopandas as gpd
from cluster_gas_network_custom import load_custom_gas_network, add_geometries, build_clustered_gas_network, filter_for_country, aggregate_clustered_gas_network_custom


def load_and_cluster(**fns):
    """
    loads and clusters custom networks
    @param fns: dictionary specifying one or multiple paths to custom networks
    @return returns tuple of clustered networks
    """
    nets = []
    for fn_name, fn in fns.items():
        # load network
        df = load_custom_gas_network(fn)
        # filter for baseyear (only add new pipelines)
        # TODO: add option to choose if only new pipelines should be exchanged
        if "build_year" in df.columns:
            df = df.loc[df.build_year == baseyear]

        # select link name as kwarg
        if "retro" in fn_name:
            kwargs = dict(carrier="H2 pipeline retrofitted", year=baseyear)
        elif "new" in fn_name:
            kwargs = dict(carrier="H2 pipeline", year=baseyear)
        elif "gas" in fn_name:
            kwargs = dict(carrier="gas pipeline", year=baseyear)

        # either cluster or only aggregate custom h2 network
        if snakemake.params.cluster_H2_network_custom:
            gdf = add_geometries(df, onshore_regions, map=False)
            # cluster network
            df_clustered = build_clustered_gas_network(gdf, onshore_regions, **kwargs)
        else:
            # if already clustered to pypsa regions only parallel pipelines need to be aggregated to total capacity
            df_clustered = aggregate_clustered_gas_network_custom(df, **kwargs)

        nets.append(df_clustered)
        # df_clustered_de = filter_for_country(gas_pipes_rcm_clustered, "DE")

    return tuple(nets)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "cluster_h2_custom",
            simpl="",
            clusters="180",
            ll="vopt",
            opts="",
            sector_opts="100H-T-H-B-I-A-solar+p3-linemaxext10-onwind+p0.4",
            planning_horizons=2035,
        )

    logging.basicConfig(level=snakemake.config["logging"]["level"])

    baseyear = snakemake.params.baseyear

    fn_regions = snakemake.input.onshore_regions
    onshore_regions = gpd.read_file(fn_regions).set_index("name")

    # if set in config custom H2 network can be added as base infrastructure such as FNB H2 core network
    # gas network is also replaced accordingly
    fn_gas = snakemake.input.gas_network_custom
    fn_retro = snakemake.input.h2_network_custom_retro
    fn_new = snakemake.input.h2_network_custom_new
    h2_new_clustered, h2_retro_clustered, gas_clustered = load_and_cluster(
        **dict(fn_new=fn_new, fn_retro=fn_retro, fn_gas=fn_gas))

    # save clustered networks
    h2_new_clustered.to_csv(snakemake.output.clustered_h2_new_custom)
    h2_retro_clustered.to_csv(snakemake.output.clustered_h2_retro_custom)
    gas_clustered.to_csv(snakemake.output.clustered_gas_custom)
