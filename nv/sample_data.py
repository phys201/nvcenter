import pandas as pd
from pathlib import Path
import xarray as xr


def get_file_path(filename, data_dir='data'):
    # Path(__file__).resolve() returns the absolute path of the current module.
    # __file__ is a special variable path of the current file (here
    # data_io.py). We can use it as base path to construct other paths
    # that should end up correct on other machines or when the package is
    # installed
    module_directory = Path(__file__).resolve().parent
    data_path = Path(module_directory, data_dir, filename)
    return data_path


def load_data(data_file):
    input_file = pd.read_csv(data_file)
    output_data = input_file.to_xarray()
    return output_data