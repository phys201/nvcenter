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
    data_file = get_file_path(data_file, data_dir = 'data')
    if not str(data_file).endswith('.csv'):
        data_file = str(data_file) + '.csv'
        input_file = pd.read_csv(str(data_file))
        output_data = input_file.to_xarray()
    elif str(data_file).endswith('.csv'):
        input_file = pd.read_csv(str(data_file))
        output_data = input_file.to_xarray()
    else:
        raise TypeError('file must be a CSV file')
    return output_data
  