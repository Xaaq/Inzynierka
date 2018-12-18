from glob import glob

import numpy as np
import pandas as pd


def extract_params_from_agent_csv(file_path: str) -> np.ndarray:
    parameter_function = lambda data_frame: data_frame["EvacTime[s]"].max()
    agent_data_frame = pd.read_csv(file_path, decimal=",")
    value_vector = parameter_function(agent_data_frame)
    return value_vector


for simulation_dir in sorted(glob("tested_output_data/test*")):
    single_simulation_results = [extract_params_from_agent_csv(f"{single_simulation_dir}/Stat#Lev0_agentSummary.csv")
                                 for single_simulation_dir in glob(f"{simulation_dir}/Simulation*")]
    print(np.mean(single_simulation_results))
