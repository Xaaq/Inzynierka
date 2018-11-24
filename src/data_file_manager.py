import os
import xml.etree.ElementTree
import xml.etree.ElementTree as ET
from itertools import chain
from os import listdir
from random import randint, random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.set_printoptions(linewidth=400, precision=2, threshold=np.nan, suppress=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class DataFilesManager:
    independent_parameters = [("rootObj/globalConfiguration/panicSpreadFctr", (0.5, 3), float),
                              ("rootObj/globalConfiguration/panicCancelZone", (0.01, 0.95), float),
                              ("rootObj/globalConfiguration/cancelPanicChance", (10, 99), int),
                              ("rootObj/globalConfiguration/choosingEvacPathMode", (1, 4), int),
                              ("rootObj/level/numOfPeds", (100, 458), int),
                              ("rootObj/level/chaosLevel", (1, 100), int),
                              ("rootObj/level/penaltyFact/densFactor", (1, 10), float),
                              ("rootObj/level/penaltyFact/freqFactor", (1, 10), float),
                              ("rootObj/level/penaltyFact/panicFactor", (0.1, 3), float),
                              ("rootObj/level/penaltyFact/distanceFactor", (0.1, 5), float),
                              ("rootObj/level/penaltyFact/randFactor", (0, 1), float)]

    dependent_parameters = [
        ("rootObj/level/preMovementTime/meanValue", "rootObj/level/preMovementTime/stdDeviation", (1, 10), float),
        ("rootObj/level/speedDistribution/meanValue", "rootObj/level/speedDistribution/stdDeviation", (1, 10),
         float)
    ]

    @classmethod
    def create_random_parameters_file(cls, file_path: str):
        xml_root = ET.parse("template_params.xml").getroot()

        for parameter_path, (min_val, max_val), val_type in cls.independent_parameters:
            if val_type == int:
                random_value = randint(min_val, max_val)
            else:
                random_value = random() * (max_val - min_val) + min_val

            xml_root.findall(parameter_path)[0].text = str(random_value)

        for mean_parameter_path, std_parameter_path, (min_val, max_val), val_type in cls.dependent_parameters:
            mean_random = random() * (max_val - min_val) + min_val
            std_random = random() * mean_random / 5

            xml_root.findall(mean_parameter_path)[0].text = str(mean_random)
            xml_root.findall(std_parameter_path)[0].text = str(std_random)

        ET.ElementTree(xml_root).write(file_path)

    @classmethod
    def extract_params_from_xml(cls, file_path: str) -> np.ndarray:
        independent_parameters = [parameter_path for parameter_path, _, _ in cls.independent_parameters]
        dependent_parameters_tuples = [(parameter_path_1, parameter_path_2)
                                       for parameter_path_1, parameter_path_2, _, _ in cls.dependent_parameters]
        dependent_parameters = list(chain(*dependent_parameters_tuples))
        parameters = independent_parameters + dependent_parameters

        xml_root = xml.etree.ElementTree.parse(file_path).getroot()
        value_vector = [float(xml_root.findall(parameter)[0].text) for parameter in parameters]
        return np.array(value_vector)

    @staticmethod
    def extract_params_from_agent_csv(file_path: str, output_parameters_count: int) -> np.ndarray:
        parameter_functions = [
            lambda data_frame: data_frame["EvacTime[s]"].max(),
            lambda data_frame: ((data_frame["AverageSpeed[m/s]"]
                                 - np.ones_like(data_frame["AverageSpeed[m/s]"])
                                 * data_frame["AverageSpeed[m/s]"].max()) ** 2).sum()
                               / data_frame["AverageSpeed[m/s]"].count()
        ]

        agent_data_frame = pd.read_csv(file_path, decimal=",")
        value_vector = [function(agent_data_frame) for function in parameter_functions[:output_parameters_count]]
        return np.array(value_vector)

    @classmethod
    def create_parameters_file(cls, parameters: np.ndarray, file_path: str):
        xml_root = ET.parse("template_params.xml").getroot()
        independent_parameters = parameters[:len(cls.independent_parameters)]

        for parameter, (parameter_path, _, val_type) in zip(independent_parameters, cls.independent_parameters):
            xml_root.findall(parameter_path)[0].text = str(val_type(parameter))

        dependent_parameters = parameters[len(cls.independent_parameters):]

        for index, (mean_parameter_path, std_parameter_path, _, val_type) in enumerate(cls.dependent_parameters):
            xml_root.findall(mean_parameter_path)[0].text = str(dependent_parameters[index * 2])
            xml_root.findall(std_parameter_path)[0].text = str(dependent_parameters[index * 2 + 1])

        xml_root.findall("rootObj/globalConfiguration/simStatDir")[0].text = \
            "/home/xaaq/my-projects/inzynierka/tested_output_data"
        ET.ElementTree(xml_root).write(file_path)

    @classmethod
    def extract_simulation_means_data(cls, all_simulations_path: str, output_parameters_count: int):
        input_data_list = []
        output_data_list = []

        for simulation_dir in listdir(all_simulations_path):
            input_data = cls.extract_params_from_xml(f"{all_simulations_path}/{simulation_dir}/input_param.xml")

            output_data = [
                cls.extract_params_from_agent_csv(
                    f"{all_simulations_path}/{simulation_dir}/{single_simulation_dir}/Stat#Lev0_agentSummary.csv",
                    output_parameters_count)
                for single_simulation_dir in listdir(f"{all_simulations_path}/{simulation_dir}")
                if os.path.isdir(f"{all_simulations_path}/{simulation_dir}/{single_simulation_dir}")
            ]
            output_data = np.array(output_data).mean(0)

            input_data_list.append(input_data)
            output_data_list.append(output_data)

        return np.array(input_data_list), np.array(output_data_list)

    @classmethod
    def extract_simulation_all_data(cls, all_simulations_path: str, output_parameters_count: int):
        input_data_list = []
        output_data_list = []

        for simulation_dir in listdir(all_simulations_path):
            input_data = cls.extract_params_from_xml(f"{all_simulations_path}/{simulation_dir}/input_param.xml")

            for single_simulation_dir in listdir(f"{all_simulations_path}/{simulation_dir}"):
                single_simulation_dir_path = f"{all_simulations_path}/{simulation_dir}/{single_simulation_dir}"

                if os.path.isdir(single_simulation_dir_path):
                    output_data = cls.extract_params_from_agent_csv(
                        f"{single_simulation_dir_path}/Stat#Lev0_agentSummary.csv", output_parameters_count)
                    input_data_list.append(input_data)
                    output_data_list.append(output_data)

        return np.array(input_data_list), np.array(output_data_list)


if __name__ == "__main__":
    data_files_manager = DataFilesManager()
    plt.hist(data_files_manager.extract_simulation_means_data("simulation_output_data", 1)[1], 30)
    plt.show()
    plt.hist(data_files_manager.extract_simulation_all_data("simulation_output_data", 1)[1], 30)
    plt.show()
    # print(data_files_manager.extract_simulation_all_data("simulation_output_data", 1)[1])
    # print(data_files_manager.extract_params_from_xml("simulation_output_data/15.11.2018-19.02.55/input_param.xml"))
    # for i in range(10):
    #     create_random_parameters_file(f"generated_parameters/params_{i}.xml")
