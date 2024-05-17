import json 
import pandas as pd 

def load_data(path):
    f= open(path)   
    data = json.load(f)
    return data

def json_arr_to_file(json_arr, filename_to_write, indent=None):
    with open(filename_to_write, "w") as f:
        json.dump(json_arr, f, indent=indent)
        f.write("\n")


import time

def timer(func):
    """A decorator that prints the execution time of the function it decorates."""
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # End the timer
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def union_list_save_csv(list_jsons_filepaths, out_path ): 
    if out_path.endswith('.csv'):
        out_path = out_path[:-4] + '.csv'
    else:
        out_path = out_path + '.csv'
        
    # Initialize an empty list to hold the combined data from all files
    combined_data = []

    # Iterate over each file, open it, read the JSON content, and append it to the combined_data list
    for file_path in list_jsons_filepaths:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load the JSON data from the file
            combined_data.extend(data)  # Append the data to the combined list

    pd.read_json(json.dumps(combined_data)).to_csv(out_path)