import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def create_pv_file():
    # 1. Define your file names
    input_filename = 'ev2gym/data/early_pv_scenario/test-pv.csv'  # Replace with your actual file name
    output_filename = 'ev2gym/data/optimal_pv_scenario/test-pv.csv'  # The new file that will be created

    # 2. Load the CSV into a DataFrame
    print("Loading data...")
    df = pd.read_csv(input_filename)

    # 3. Change the last column to 30.0
    # We use the exact column name from your example
    print("Updating prices...")
    df['electricity'] = 0

    # 4. Save the modified DataFrame back to a new CSV file
    # index=False ensures pandas doesn't add an extra column of row numbers
    print("Saving modified file...")
    df.to_csv(output_filename, index=False)

def create_prices_file():
    # 1. Define your file names
    input_filename = 'ev2gym/data/test/test-prices.csv'  # Replace with your actual file name
    output_filename = 'ev2gym/data/optimal_pv_scenario/test-prices-100.csv'  # The new file that will be created

    # 2. Load the CSV into a DataFrame
    print("Loading data...")
    df = pd.read_csv(input_filename)

    # 3. Change the last column to 30.0
    # We use the exact column name from your example
    print("Updating prices...")
    df['Price (EUR/MWhe)'] = 100.0

    # 4. Save the modified DataFrame back to a new CSV file
    # index=False ensures pandas doesn't add an extra column of row numbers
    print("Saving modified file...")
    df.to_csv(output_filename, index=False)


def create_residential_loads():
    # Create an array of 0.001 instead of pure 0.0
    # 672 rows (1 week) x 10 columns (for the sample size)
    tiny_data = np.full((672, 10), 0.001)

    # Convert to pandas DataFrame
    df_tiny = pd.DataFrame(tiny_data)

    # Save to CSV without headers or row numbers
    filename = "ev2gym/data/optimal_pv_scenario/almost_zero_load_10_houses.csv"
    df_tiny.to_csv(filename, header=False, index=False)

    print(f"Created {filename}!")
    print("Max value is now 0.001, avoiding divide-by-zero errors.")

def check_path():
    save_path = ".saved_models/1cs_V2GProfitMax/td3_V2G_profitmaxV2_V2G_profit_max_loads"
    file_name = "best_model.zip"
    full_path = os.path.join(save_path, file_name)

    print(f"Checking path: {os.path.abspath(full_path)}")
    print(f"Does directory exist?: {os.path.exists(save_path)}")
    print(f"Does file exist?: {os.path.exists(full_path)}")

    # List files in that directory to see what is actually there
    if os.path.exists(save_path):
        print("Files in directory:", os.listdir(save_path))
    else:
        print("The directory path is wrong.")

def pv_values():
    # Load the existing file
    input_file = 'ev2gym/data/early_pv_scenario/test-pv.csv'
    output_file = 'ev2gym/data/chaotic_pv_scenario/test-pv.csv'

    # Read the data
    df = pd.read_csv(input_file)

    # Set all electricity values to 1
    df['electricity'] = 0

    # Save the updated CSV
    df.to_csv(output_file, index=False)

    print(f"Successfully created {output_file} with all electricity values set to 1.")

def plot_reward():
    data = np.loadtxt("log.txt")
    #data = data[:500]

    # split columns
    col1 = data[:,0]
    col2 = data[:,1]
    col3 = data[:,2]

    # x-axis (row index)
    x = range(len(col1))

    # plot
    plt.plot(x, col1, label="Profit")
    plt.plot(x, col2, label="Overload")
    plt.plot(x, col3, label="Satisfaction")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward parts")
    plt.legend()

    plt.savefig("my_plots/plot_train_REWARD_without_overload.png")

def plot_transformer_log():
    data = np.loadtxt("transformer.log")
    plt.plot(range(len(data)), data[:,0], label="Max power")
    plt.plot(range(len(data)), data[:,1], label="Inflexible load")
    plt.plot(range(len(data)), data[:,2], label="Solar power")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.title("Transformer power")
    plt.legend()
    plt.savefig("my_plots/plot_transformer_log.png")

def plot_dif_between_forecast_and_actual():
    data = np.loadtxt("transformer-load-vs-forecast.log")
    plt.plot(range(len(data)), data[:,0], label="Forecast")
    plt.plot(range(len(data)), data[:,1], label="Actual")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.title("Difference between forecast and actual")
    plt.legend()
    plt.savefig("my_plots/plot_dif_between_forecast_and_actual.png")

def plot_two_functions():
    def f1(x):
        return 100 * np.exp(-8*x)

    def f2(x):
        return 100 * np.exp(-10*x)
    
    x = np.linspace(0, 1, 100)
    plt.plot(x, f1(x), label="100e^(-8x)")
    plt.plot(x, f2(x), label="100e^(-10x)")
    plt.grid(True, color="lightgray", linestyle="-", linewidth=0.8, alpha=0.8)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("my_plots/plot_two_functions.png")

def mean_and_std():
    data = np.loadtxt("autoencoder/log_N.txt")
    print(f"Data shape: {data.shape}")
    print(f"Mean: {np.mean(data)}")
    print(f"Std: {np.std(data)}")
    


if __name__ == "__main__":
    mean_and_std()