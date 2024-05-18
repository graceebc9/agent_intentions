import sys
import os
from pathlib import Path
import json 
import os
import glob 


# Adding the 'src' and 'src/utils' directories to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
utils_dir = os.path.join(src_dir, 'utils')
sys.path.append(src_dir)
sys.path.append(utils_dir)

from utils import load_data, json_arr_to_file, run_api_call


max_tokens = 300
model = 'gpt-4'
# model = 'gpt-3.5-turbo'
# run_name = 'gpt-4-dataset-V2'
run_name = 'gpt-4-dataset-V2'


# generate prompt to ask gpt 4 tols take a list of 4 options and return the 'adaptive outcome' 
def adapt_prompt(options): 
    gen_adapt_prompt = f""" 
    Please take each of the following options and return a sentance describing the adapative outcome where the user has already taken the option.
    i.e. 'Suggest the user gives their friend space for a while -> ' The user has already given their friend space for a while'. 
    Return the outcomes in the same order as the options.
    {options } 
    Please return responses in a list. 
    """
    prompt = [{'role': 'user', 'content' : gen_adapt_prompt}] 

    return prompt


def process_one_ap_file(file, write_path):
    full_json = load_data(file)
    max_tokens = 300
    model = 'gpt-3.5-turbo'

    for item in full_json:     
        options = item['options']
        prom = adapt_prompt( options)
        adapt_results = run_api_call(prom, model, max_tokens) 
        item['adapt_response'] = adapt_results
    
    print(f'Starting to save file {file}')
    json_arr_to_file(full_json, write_path, indent=4)

def run_ap_gen(model ,run_name): 
        # Make dir to store processed data
    file_dir = os.path.join(script_dir, "data", "dataset_with_adapt", f'd_name--{run_name}')
    os.makedirs(file_dir, exist_ok=True)
    for f in  ['helpful', 'harmless']:
        os.makedirs( os.path.join(file_dir, f) , exist_ok=True)
                
    # Files to process
    topics_files = glob.glob(f'{script_dir}/data/dataset/d_name--{run_name}/*/*.json')
    if len(topics_files) == 0:
        raise Exception("No files found. Please run dataset_generation.py first")
    print(topics_files)

    print(f'Starting to process {len(topics_files)} files \n')
    # Loop over topics files (jsons of topics), process each sub scenario and save new json for each topic file 
    for file in topics_files:
        # check if file exists
        print(file)
       
        file_name = file.split('/')[-1]
        hh  = file.split('/')[-2]
        write_path = os.path.join(file_dir, hh, file_name )
        print(write_path)
        #  check if file exists
        if os.path.exists(write_path):
            print(f'File {write_path} already exists, skipping')
            continue
        process_one_ap_file(file, write_path )

    print('Run complete')


if __name__ == "__main__":
    run_ap_gen(model, run_name)