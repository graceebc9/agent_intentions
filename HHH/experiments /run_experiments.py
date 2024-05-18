# Run experiments 

import subprocess
import os 
import time 

################################ Update the following variables ################################

data_set_name = 'gpt-4-dataset-V2'
type_experiment= 'chainofthought'  #True or False 
max_workers = 12
models = [
    # 'gpt-4-turbo-preview', 
    # # 'gpt-4' , 
    'gpt-3.5-turbo',
    'davinci-002' ,
    "llama-7b-chat", 
    "llama-7b-32k" ,
    "llama-13b-chat"]
    # "llama-70b-chat"]



# models = [
#     # 'mistral-7b' ,
# #  'mistral-7b-instruct',
#  'mixtral-8x7b-instruct'] 



################################ Run the experiments  ########################################

import subprocess
from concurrent.futures import ThreadPoolExecutor

if type_experiment =='fewshot':
    num_shots = [2, 4, 6]
else:
    num_shots = [0 ]



def run_script(script_name, **kwargs):
    command = ['python', script_name]
    for key, value in kwargs.items():
        command.append(f'--{key}={value}')

    max_retries = 3  # Maximum number of retries
    retry_delay = 10  # Seconds to wait between retries

    for attempt in range(max_retries):
        try:
            # Attempt to run the command
            completed_process = subprocess.run(command, check=True)
            break  # If the command succeeds, break out of the loop
        except subprocess.CalledProcessError as e:
            # If the command fails, print an error message and retry after a delay
            print(f"Attempt {attempt + 1} failed, retrying after {retry_delay} seconds...")
            time.sleep(retry_delay)
    else:
        # If all attempts fail, raise an exception or handle failure appropriately
        print(f"All {max_retries} attempts failed. Please check the script or environment.")

def main():
    # Use ThreadPoolExecutor to run scripts concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_script, 'adaptive_prompting.py', model=model, run_name=data_set_name, type_experiment= type_experiment , num_ex = num_ex )
                   for model in models for num_ex in num_shots]

        # Wait for all futures to complete
        for future in futures:
            future.result()  # This will re-raise any exception that occurred in the thread
    # run_script('adaptive_prompting.py', model = models[0],  run_name = data_set_name)

if __name__ == '__main__':
    main()
