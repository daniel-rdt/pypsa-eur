import pandas as pd
import pypsa
import os


def swap_direction(df):
    df["Bus0"] = df.index.get_level_values("Bus0")
    df["Bus1"] = df.index.get_level_values("Bus1")
    df = df.droplevel(["Bus0","Bus1"])
    swap_buses = {"Bus0": "Bus1", "Bus1": "Bus0"}
    df_n = df.rename(columns=swap_buses)
    df_n.set_index([df_n.index, "Bus0", "Bus1"], drop=True, inplace=True)
    return df_n * (-1)


def disaggregate_hourly(df, year, baseyear=2013):

    # disaggregate loads to hourly values in MWh
    yearly_i = pd.date_range(f'{baseyear}-01-01', f'{baseyear}-12-31 23:00:00', freq='1H', name="snapshot")
    df_hourly = df.T.reindex(yearly_i).ffill()  # just forward fill with values since MW in hourly is MWh

    # change timestamp year to year of analysis
    df_hourly.index = df_hourly.index.shift(freq=pd.DateOffset(years=int(year)-int(baseyear)))

    return df_hourly


def positive_interconnector_flow(df, ctry="DE"):

    # filter for flows with DE as Bus 1 to get following convention --> positive flow = into DE
    # unit of flow is [MWh/h] = [MW]
    bus0 = (~df.index.get_level_values(level=1).str.startswith(ctry))
    bus1 = (df.index.get_level_values(level=2).str.startswith(ctry))

    # pipelines with only either Bus0 or Bus1 as DE nodes are interconnectors
    intercon_bus1 = df.loc[(bus0 & bus1)].copy()
    intercon_bus0 = df.loc[(~bus0 & ~bus1)].copy()

    # swap direction for pipelines with DE as Bus 0 so that convention --> positive flow = into DE
    intercon_bus0 = swap_direction(intercon_bus0)

    return pd.concat([intercon_bus1, intercon_bus0])


def set_bus_index(df, splits="pipeline|retrofitted"):

    # transpose to get add new column for index
    df = df.T

    # extract Bus information from str index
    df["Bus0"] = df.index.str.split("-|<").str[0].str.split(splits).str[-1].str[1:-1].values
    df["Bus1"] = df.index.str.split("-|<").str[-2].str[2:].values

    # add Bus information to index
    return df.set_index([df.index, "Bus0", "Bus1"], drop=True)


if __name__ == "__main__":

    ###############################
    # Settings
    ###############################

    scenario = "low"
    run_name = f"20240201_730SEG_{scenario}demand_myopic_stepwise_2030"
    ll = "lvopt"
    sector_opts = "730SEG-T-H-B-I-A-solar+p3-linemaxext10-onwind+p0.4-gas+m2.5"
    year = run_name[-4:]
    simpl = ""
    clusters = "180"

    save_dir = "data_exchange_pypsa"
    save_path = f"{save_dir}/{run_name[:8]}_{sector_opts.split('-')[0]}/{run_name[:8]}_{sector_opts.split('-')[0]}_{scenario}_{year}"
    if not os.path.isdir(f"{save_path}"):
        print(f"Created directory {save_path}.")
    os.makedirs(f"{save_path}", exist_ok=True)
    print(f"Results will be saved to {save_path}.")

    ###############################
    # load pypsa network
    ###############################

    # load network
    n = pypsa.Network(f"results/{run_name}/postnetworks/elec_s_180_{ll}__{sector_opts}_{year}.nc")
    snapshot_weights = n.snapshot_weightings

    # get energy balance
    balance = n.statistics.energy_balance(aggregate_bus=False, aggregate_time=False)

    ###############################
    # Energy balance H2
    ###############################

    # get H2 energy balance by node and snapshots from statistics module
    carrier = 'H2'
    df_h2 = balance.loc[:, :, carrier].copy()

    # positive values represent supply, negative values signify consumption of H2
    # unit is MW, i.e for energy values in MWh has to be multiplied with number of hours (weight) of snapshot
    # sorting index for consistent order
    loads_h2 = df_h2.sort_index(level=2)

    # rename bus index to only include bus node name instead of H2
    loads_h2["bus"] = loads_h2.index.get_level_values(level=2).str[:-3]
    loads_h2.set_index([loads_h2.index.droplevel(2), "bus"], drop=True, inplace=True)

    # disaggregate loads to hourly values in MWh and change timestamp to year of analysis
    loads_h2_MWh = disaggregate_hourly(loads_h2, year=year, baseyear=2013)

    # save H2 supply and demand data as csv and parquet
    # unit is MWh
    loads_h2_MWh.T.to_csv(f"{save_path}/H2_supply_demand.csv")
    loads_h2_MWh.to_parquet(f"{save_path}/H2_supply_demand.parquet")

    ###############################
    # Interconnector flows H2
    ###############################

    df_pipeline_flow_h2 = n.links_t.p0.filter(like="H2 pipeline")

    df_pipeline_flow_h2 = set_bus_index(df_pipeline_flow_h2, splits="pipeline|retrofitted")

    # filter for flows with DE as Bus 1 to get following convention --> positive flow = into DE
    # unit of flow is [MWh/h] = [MW]
    intercon_flows_DE_h2 = positive_interconnector_flow(df_pipeline_flow_h2, "DE")

    # disaggregate loads to hourly values in MWh and change timestamp year to year of analyis
    intercon_flows_DE_h2_MWh = disaggregate_hourly(intercon_flows_DE_h2, year=year, baseyear=2013)

    # save intercon_flows to csv and parquet
    intercon_flows_DE_h2_MWh.T.to_csv(f"{save_path}/H2_interconnector_flows.csv")
    intercon_flows_DE_h2_MWh.to_parquet(f"{save_path}/H2_interconnector_flows.parquet")

    ###############################
    # Energy balance gas
    ###############################

    # get H2 energy balance by node and snapshots from statistics module
    carrier = 'gas'
    df_gas = balance.loc[:, :, carrier].copy()

    # sorting index for consistent order
    loads_gas = df_gas.sort_index(level=2)

    # rename bus index to only include bus node name instead of gas
    loads_gas["bus"] = loads_gas.index.get_level_values(level=2).str[:-4]
    loads_gas.set_index([loads_gas.index.droplevel(2), "bus"], drop=True, inplace=True)

    # disaggregate loads to hourly values in MWh
    loads_gas_MWh = disaggregate_hourly(loads_gas, year=year, baseyear=2013)

    # save loads gas to csv and parquet
    loads_gas_MWh.T.to_csv(f"{save_path}/Gas_supply_demand.csv")
    loads_gas_MWh.to_parquet(f"{save_path}/Gas_supply_demand.parquet")

    ###############################
    # Newly build gas pipelines
    ###############################

    year_int = int(year)
    gas_pipelines_new = n.links.query(
        " \
        carrier == 'gas pipeline new' \
        and p_nom_opt > 0 \
        and (bus0.str.startswith('DE') \
        or bus1.str.startswith('DE')) \
        and build_year == @year_int \
        "
    )
    if not gas_pipelines_new.empty:
        gas_pipelines_new.to_csv(f"{save_path}/Gas_pipelines_new.csv")

    ###############################
    # Interconnector flows gas
    ###############################

    df_pipeline_flow_gas = n.links_t.p0.filter(like="gas pipeline")

    df_pipeline_flow_gas = set_bus_index(df_pipeline_flow_gas, splits="pipeline|new")

    # filter for flows with DE as Bus 1 to get following convention --> positive flow = into DE
    # unit of flow is [MWh/h] = [MW]
    intercon_flows_DE_gas = positive_interconnector_flow(df_pipeline_flow_gas, "DE")

    # disaggregate loads to hourly values in MWh and change timestamp year to year of analysis
    intercon_flows_DE_gas_MWh = disaggregate_hourly(intercon_flows_DE_gas, year=year, baseyear=2013)

    # save intercon_flows to csv and parquet
    intercon_flows_DE_gas_MWh.T.to_csv(f"{save_path}/Gas_interconnector_flows.csv")
    intercon_flows_DE_gas_MWh.to_parquet(f"{save_path}/Gas_interconnector_flows.parquet")

    ###############################
    ## All gas pipeline flows
    ###############################

    # disaggregate loads to hourly values in MWh
    df_pipeline_flow_gas_MWh = disaggregate_hourly(df_pipeline_flow_gas, year=year, baseyear=2013)

    # group by buses to aggregate parallel pipelines
    df_pipeline_flow_gas_MWh_aggregated = df_pipeline_flow_gas_MWh.T.groupby(["Bus0", "Bus1"]).sum()

    # save intercon_flows to csv and parquet
    df_pipeline_flow_gas_MWh_aggregated.to_csv(f"{save_path}/Gas_flows_all.csv")
    df_pipeline_flow_gas_MWh_aggregated.to_parquet(f"{save_path}/Gas_flows_all.parquet")