# SPDX-FileCopyrightText: : 2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

if config["sector"]["H2_network_custom"]:
    planning_horizons_all = config["scenario"]["planning_horizons_all"]
    i = planning_horizons_all.index(int(config["scenario"]["planning_horizons"][0]))
    if (config["sector"]["reoptimise_h2"]) or (config["run"]["name"] == config["run"]["name_base"]):
        H2_INFRA_YEAR = str(planning_horizons_all[i])
        DIR = RDIR
    else:
        H2_INFRA_YEAR = str(planning_horizons_all[i - 1])
        DIR = PRDIR

    rule cluster_h2_custom:
        params:
            baseyear=H2_INFRA_YEAR,
            H2_network_custom=config["sector"]["H2_network_custom"],
            cluster_H2_network_custom=config["sector"]["cluster_H2_network_custom"],
            gas_network_custom=config["sector"]["gas_network_custom"],
            result_dir=RESULTS,
        input:
            onshore_regions=RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
            gas_network_custom="results/"
                                + DIR
                                + "h2_networks_custom/gas_network_custom_"+H2_INFRA_YEAR+".csv",
            h2_network_custom_new="results/"
                                   + DIR
                                   + "h2_networks_custom/h2_network_new_custom_"+H2_INFRA_YEAR+".csv",
            h2_network_custom_retro="results/"
                                   + DIR
                                   + "h2_networks_custom/h2_network_retro_custom_"+H2_INFRA_YEAR+".csv",
        output:
            clustered_h2_retro_custom="results/"
                                      + DIR
                                      + "h2_networks_custom/clustered_h2_network_retro_custom_s{simpl}_{clusters}_"
                                      + H2_INFRA_YEAR
                                      + ".csv",
            clustered_h2_new_custom="results/"
                                    + DIR
                                    + "h2_networks_custom/clustered_h2_network_new_custom_s{simpl}_{clusters}_"
                                    + H2_INFRA_YEAR
                                    + ".csv",
            clustered_gas_custom="results/"
                                 + DIR
                                 + "h2_networks_custom/clustered_gas_network_custom_s{simpl}_{clusters}_"
                                 + H2_INFRA_YEAR
                                 + ".csv",
        threads: 1
        resources:
            mem_mb=2000,
        log:
            LOGS
            + "cluster_h2_custom_s{simpl}_{clusters}.log",
        benchmark:
            (
                    BENCHMARKS
                    + "cluster_h2_custom/elec_s{simpl}_{clusters}"
            )
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/cluster_h2_custom.py"

    h2_infrastructure = {
        **rules.cluster_h2_custom.output,
    }

if not config["sector"]["H2_network_custom"]:
    # this is effectively an `else` statement which is however not liked by snakefmt

    h2_infrastructure = {}

rule add_existing_baseyear:
    params:
        baseyear=config["scenario"]["planning_horizons"][0],
        planning_horizons=config["scenario"]["planning_horizons_all"],
        name_base=config["run"].get("name_base"),
        foresight=config["foresight"],
        sector=config["sector"],
        existing_capacities=config["existing_capacities"],
        costs=config["costs"],
        H2_retrofit=config["sector"]["H2_retrofit"],
        H2_retrofit_capacity_per_CH4=config["sector"]["H2_retrofit_capacity_per_CH4"],
        threshold_capacity=config["existing_capacities"]["threshold_capacity"],
        H2_network_custom=config["sector"]["H2_network_custom"],
        gas_network_custom=config["sector"]["gas_network_custom"],
    input:
        overrides="data/override_component_attrs",
        network=RESULTS
        + "prenetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
        network_p= solved_previous_horizon_stepwise,#solved network at previous time step
        costs = "data/costs_{planning_horizons}.csv",
        onshore_regions = RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
        cop_soil_total = RESOURCES + "cop_soil_total_elec_s{simpl}_{clusters}.nc",
        cop_air_total = RESOURCES + "cop_air_total_elec_s{simpl}_{clusters}.nc",
        powerplants=RESOURCES + "powerplants.csv",
        busmap_s=RESOURCES + "busmap_elec_s{simpl}.csv",
        busmap=RESOURCES + "busmap_elec_s{simpl}_{clusters}.csv",
        clustered_pop_layout=RESOURCES + "pop_layout_elec_s{simpl}_{clusters}.csv",
        existing_heating="data/existing_infrastructure/existing_heating_raw.csv",
        existing_solar="data/existing_infrastructure/solar_capacity_IRENA.csv",
        existing_onwind="data/existing_infrastructure/onwind_capacity_IRENA.csv",
        existing_offwind="data/existing_infrastructure/offwind_capacity_IRENA.csv",
        **h2_infrastructure,
    output:
        RESULTS
        + "prenetworks-brownfield/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
    wildcard_constraints:
        planning_horizons=config["scenario"]["planning_horizons"][0],  #only applies to baseyear
    threads: 1
    resources:
        mem_mb=2000,
    log:
        LOGS
        + "add_existing_baseyear_elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.log",
    benchmark:
        (
            BENCHMARKS
            + "add_existing_baseyear/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/add_existing_baseyear.py"


rule solve_sector_network_myopic:
    params:
        solving=config["solving"],
        foresight=config["foresight"],
        planning_horizons=config["scenario"]["planning_horizons"],
        co2_sequestration_potential=config["sector"].get(
            "co2_sequestration_potential", 200
        ),
    input:
        overrides="data/override_component_attrs",
        network=RESULTS
        + "prenetworks-brownfield/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
        costs="data/costs_{planning_horizons}.csv",
        config=RESULTS + "config/config.yaml",
    output:
        RESULTS
        + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
    shadow:
        "shallow"
    log:
        solver=LOGS
        + "elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}_solver.log",
        python=LOGS
        + "elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}_python.log",
        memory=LOGS
        +"elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}_memory.log",
    threads: 4
    resources:
        mem_mb=config["solving"]["mem"],
    benchmark:
        (
            BENCHMARKS
            + "solve_sector_network/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"
