import pandas as pd
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

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
        output_data = input_file.to_numpy(dtype=float)
    elif str(data_file).endswith('.csv'):
        input_file = pd.read_csv(str(data_file))
        output_data = input_file.to_numpy(dtype=gfloat)
    else:
        raise TypeError('file must be a CSV file')
    return output_data

def plot_scan(data, extents):
    
    Ny, Nx = data.shape
    x_min, x_max, y_min, y_max = extents[-4:]
    x_volt = np.linspace(x_min, x_max, Nx)
    y_volt = np.linspace(y_min, y_max, Ny)
    X_volt, Y_volt = np.meshgrid(x_volt, y_volt)
    
    volt_to_um = 0.067
    x_um = x_volt * volt_to_um
    y_um = y_volt * volt_to_um
    X_um = X_volt * volt_to_um
    Y_um = Y_volt * volt_to_um
    Z_um = np.zeros_like(X_um, dtype=float)

    
    plt.figure(figsize=(6, 5))
    plt.imshow(counts_kcps, extent=[x_um.min(), x_um.max(), y_um.min(), y_um.max()])
    plt.colorbar(label="Count rate (kcps)")
    plt.xlabel("x (um)")
    plt.ylabel("y (um)")
    plt.title("Real scan data")
    plt.show()