import os
import shutil
from glob import glob

if __name__ == "__main__":
    start_directory = "simulation_output_data"
    start_directory_glob = glob(f"{start_directory}/*")

    for simulation_directory in start_directory_glob:
        simulation_directory_glob = glob(f"{simulation_directory}/*")

        for simulation_item in simulation_directory_glob:
            if os.path.isdir(simulation_item) and len(glob(f"{simulation_item}/*")) == 0:
                os.removedirs(simulation_item)

        if len(simulation_directory_glob) == 1:
            shutil.rmtree(simulation_directory)
            print(f"deleted {simulation_directory}")

    print(f"number of simulations: {len(start_directory_glob)}")
