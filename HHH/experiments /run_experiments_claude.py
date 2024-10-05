# Run experiments 

import subprocess
import os 

################################ Update the following variables ################################

data_set_name = 'gpt-4-dataset-V2'
fewshot= True  #True or False 
max_workers = 1
models = [
"claude-v1", 
    "claude-instant-v1" ]
    # 'gpt-4-turbo-preview', 
    # # 'gpt-4' , 
    # # 'gpt-3.5-turbo',
    # # 'davinci-002' ]
    # "llama-7b-chat", 
    # "llama-7b-32k" ,
    # "llama-13b-chat",
    # "llama-70b-chat"]
    # 'mistral-7b' ,
    # 'mistral-7b-instruct',
    # 'mixtral-8x7b-instruct']


# models = ['mistral-7b' , 'mistral-7b-instruct', 'mixtral-8x7b-instruct'] 



################################ Run the experiments  ########################################

import subprocess
from concurrent.futures import ThreadPoolExecutor

if fewshot ==True:
    num_shots = [2, 4, 6]
else:
    num_shots = [None ]



def run_script(script_name, **kwargs):
    # Construct the command with script name and kwargs
    command = ['python', script_name]
    for key, value in kwargs.items():
        command.append(f'--{key}={value}')
    
    # Run the command and check if it succeeded
    completed_process = subprocess.run(command, check=True)  # Added check=True for automatic error handling

def main():
    # Use ThreadPoolExecutor to run scripts concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_script, 'adaptive_prompting.py', model=model, run_name=data_set_name, fewshot= fewshot , num_ex = num_ex )
                   for model in models for num_ex in num_shots]

        # Wait for all futures to complete
        for future in futures:
            future.result()  # This will re-raise any exception that occurred in the thread
    # run_script('adaptive_prompting.py', model = models[0],  run_name = data_set_name)

if __name__ == '__main__':
    main()
