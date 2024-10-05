import sys
import os
from pathlib import Path
import json 
import os

# Adding the 'src' and 'src/utils' directories to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
utils_dir = os.path.join(src_dir, 'utils')
sys.path.append(src_dir)
sys.path.append(utils_dir)

from utils import load_data, json_arr_to_file, run_api_call
from utils import  timer , prompt_dataset_5pin, load_examples_all_5pin


# Load examples
# helpful_examples, harmless_examples = load_examples_all() 
helpful_examples, harmless_examples = load_examples_all_5pin() 

hh = ['Helpful', 'Harmless' ]
neg_hh = ['Unhelpful', 'Harmful']
examples = [helpful_examples, harmless_examples]  
h_vars = list(zip(examples, hh, neg_hh  ) ) 


run_name = 'gpt-4-dataset-V2'
# run_name = 'gpt-3.5-turbo_v2'

model = "gpt-4"
num_elements = 20
max_tokens = 7000
# model="gpt-3.5-turbo-16k"
# model="gpt-3.5-turbo"


# if model == "gpt-3.5-turbo":
#     max_tokens = 3250
#     num_elements = 20
# elif model == 'gpt-4':
#     max_tokens = 7000
#     num_elements = 15

topics  = [
    "Medical",
    "Community",
    "Technology Use",
    "Travel",
    "Education and Learning",
    "Career and Job Search",
    "Cultural",
    "Recreational Activities",
    "Personal Relationships", 
    "Criminal activity"
]

topics = [
    # "Medical",
    # "Community",
    # "Technology Use",
    # "Travel",
    # "Education and Learning",
    # "Career and Job Search",
    # "Cultural",
    # "Recreational Activities",
    # "Personal Relationships",
    # "Criminal activity",
    # Additional topics to consider including
    
    "Financial Planning",
    "Mental Health",
    "Art and Design",
    "Sports and Fitness",
    "Food and Nutrition",
    "Housing and Real Estate",
    "Volunteering and Social Service",
    "Politics"
]


@timer 
def run_dataset_gen():
    print(f'Making Directories for run name {run_name} \n')
    file_dir = os.path.join(script_dir, "data", "dataset", f'd_name--{run_name}')
    os.makedirs(file_dir, exist_ok=True)
    for f in hh:
        f=f.lower() 
        os.makedirs( os.path.join(file_dir, f) , exist_ok=True)
            
    for list_vars in h_vars: 
        ex, h, neg_h = list_vars
        print(f'Starting {h} \n')
        for topic in topics:
            print(f'Starting topic {topic} \n')
            filename_to_write = os.path.join( file_dir, h.lower() , f"{num_elements}--{topic}.json" ) 
            print(filename_to_write)
            # Check if the file exists
            if os.path.exists(filename_to_write):
                print('topic complete')
                continue
            
            
            ds_prompt = prompt_dataset_5pin( ex, h, neg_h , topic, num_elements) 
            print("Topic: ", topic, "Help/harm?" , h , "Prompt: ", ds_prompt, '\n') 

            content = run_api_call(ds_prompt, model, max_tokens)

            try:
                data = json.loads(content)
            except Exception as e:
                print("Exception: ", e)
                print(content)

            if isinstance(data, list):
                result_len = len(data)
                print(f"Result length: {result_len}")
            else:
                print("Result is not a list :(")
                print(data)

            
            json_arr_to_file(data, f"{filename_to_write}", indent=2)

    print('Generation completed')

if __name__ == '__main__':
    run_dataset_gen() 