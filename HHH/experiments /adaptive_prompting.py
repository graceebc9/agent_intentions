import sys
import os
from pathlib import Path
import json 
import os
import glob 
import ast 
import re
import argparse
import os
import glob
import time 

# Adding the 'src' and 'src/utils' directories to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
utils_dir = os.path.join(src_dir, 'utils')
sys.path.append(src_dir)
sys.path.append(utils_dir)


from utils import load_data, json_arr_to_file, run_api_call
from utils.model_api.other_api  import   call_llama , call_claude, call_claude3, call_togetherai
from utils import  preprocess_options_and_labels

from utils.prompt_fns import intention_prompt_first, intention_prompt_second
from utils.variable_prompt_fns import intention_prompt_second_fewshotlearning , intention_prompt_second_chainofthought

claude_model_family = [
    "claude-v1",
    "claude-instant-v1"
]

claude3_model_family = [
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307'
]

gpt_models = [
    'gpt-4-turbo-preview',
    'gpt-4',
    'gpt-3.5-turbo']

gpt_base_models = [
    'davinci-002',
    'babbage-002' 
]

llama_model_family = [
    "llama-7b-chat",
    "llama-7b-32k",
    "llama-13b-chat",
    "llama-70b-chat" ]

mixtral_family = [
    'mistral-7b', 
    'mistral-7b-instruct', 
    'mixtral-8x7b-instruct',
    'mixtral-8x22b',
    'mixtral-8x22b-instruct'
    ]

other_models = [
    "NousResearch/Nous-Hermes-Llama2-13b",
    "falcon-7b-instruct",
    "falcon-40b-instruct",
    "alpaca-7b",
    "codellama-7b-instruct",
    "codellama-13b-instruct",
    "codellama-34b-instruct",
    "openassistant-llama2-70b",
    "vicuna-7b",
    "vicuna-13b",
    "vicuna-13b-16k"
]

tog_model_family = [
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1"
]


############################################### Model specific Functions ###############################################

def execute_model_call(model, prompt, max_tokens):
    """Execute the call to the model based on type."""
    if model in gpt_models:
        return run_api_call(prompt, model, max_tokens)
    elif model in gpt_base_models:
        return run_api_call(prompt, model, max_tokens, base=True)
    elif model in llama_model_family + mixtral_family:
        return call_llama(prompt, model)
    elif model in claude_model_family:
        return call_claude(prompt, model)
    elif model in claude3_model_family:
        return call_claude3(prompt, model)
    elif model in tog_model_family:
        return call_togetherai(prompt, model)
    else:
        print(model)
        raise Exception("Other model type called - check how to call API" ) 


def select_prompt_based_on_model(model, scenario, pr_string, adapt_sentence, type_experiment, num_examples=None):
    """Select the appropriate prompt based on the model."""
    # check if type_experiment is 'plain' few'shot or 'chainofhtought'
    if type_experiment == 'plain':
        if model in gpt_models + gpt_base_models + llama_model_family + mixtral_family + claude_model_family:
            return intention_prompt_second(scenario, pr_string, adapt_sentence)
     
    elif type_experiment=='fewshot':
        print('Selecting few shot learn prompt')
        if model in gpt_models + gpt_base_models+ llama_model_family + mixtral_family + claude_model_family:
            return intention_prompt_second_fewshotlearning(scenario, pr_string, adapt_sentence, num_examples)
    elif type_experiment == 'chainofthought':
        if model in gpt_models + gpt_base_models+ llama_model_family + mixtral_family + claude_model_family:
            return intention_prompt_second_chainofthought(scenario, pr_string, adapt_sentence)

    else:
        raise Exception('Invalid type of experiment')
    
def generate_and_process_response(model, scenario, pr_string, max_tokens, mapping, first_call=True, numeric_first_response=None, type_experiment='plain', num_ex=None ):

    """Generate and process the response for the prompt."""
    if first_call:
        prompt = intention_prompt_first(scenario, pr_string)
      
    else:
        adapt_sentence = mapping[numeric_first_response]['adapt_outcome']

        prompt = select_prompt_based_on_model(model, scenario, pr_string, adapt_sentence, type_experiment, num_examples = num_ex)

    response = execute_model_call(model, prompt, max_tokens)
    numeric_response = extract_numeric_response_if_applicable(model, response, cot)
    # print(numeric_response)
    return numeric_response if numeric_response is not None else response
 


############################################### Admin Functions ###############################################

# def extract_numbers_in_range(text, lower=1, upper=5, base=False):
#     # This pattern will match whole numbers in the text
#     pattern = r'\b[1-5]\b'
    
#     # Find all matches in the text
#     matches = re.findall(pattern, text)
    
#     # Convert matched strings to integers
#     numbers = [int(match) for match in matches if lower <= int(match) <= upper]
#     if not base and len(numbers) != 1:
#         # raise ValueError('More than one answer received.')
#         print('More than one answer found')
#         return None 
#     return numbers[0] 


def extract_numbers_in_range(text, lower=1, upper=5, base=False, cot=False):
    # This pattern will match whole numbers in the specified range within the text
    if cot==False:
        pattern = rf'\b[{lower}-{upper}]\b'
        matches = re.findall(pattern, text)
    elif cot==True:
        pattern = rf'Response: \b[{lower}-{upper}]\b'
        matches = [m.split(" ")[1] for m in re.findall(pattern, text)]
    
    # Convert matched strings to integers and filter unique values using set
    numbers = set(int(match) for match in matches if lower <= int(match) <= upper)
    
    # Process based on the number of unique numbers found
    if len(numbers) > 1 or len(numbers) == 0:
        # More than one unique number or none found, invalid input
        print('More than one unique answer or no answer found.')
        return None
    elif len(numbers) == 1:
        # Exactly one unique number found, return it
        return numbers.pop()
    else:
        # Should not reach here, but just in case
        print('Unexpected error.')
        return None

def parse_adapt_outcome(adapt_response):
    """Parse and adjust the adapt response."""
    return ast.literal_eval(adapt_response.replace("'s", "es"))

def validate_adapt_outcomes(adapt_outcome, options):
    """Validate the adapt outcomes."""
    if len(adapt_outcome) != len(options) or len(adapt_outcome) != 5:
        print('Mismatch in number of adapt outcomes or options')
        return False
    return True

def save_processed_data(data, write_path ):
    """Save the processed data to a file."""
    print(f'Starting to save file to {write_path}')
    json_arr_to_file(data, write_path, indent=4)
    print('File saved.')

def extract_numeric_response_if_applicable(model, response, cot=False):
    """Extract numeric response from the model's response if applicable."""
    return extract_numbers_in_range(response, cot=cot)
    # if model not in gpt_base_models and cot == False:
    #     return extract_numbers_in_range(response)
    # else:
    #     return extract_numbers_in_range(response, base=True)

 
def process_one_item(item, model, max_tokens, type_experiment, num_ex = None):
    """Process a single item from the loaded data."""
    # try:
    op, lab, scenario = item['options'], item['labels'], item['scenario']
    adapt_outcome = parse_adapt_outcome(item['adapt_response'])

    if not validate_adapt_outcomes(adapt_outcome, op):
        return  # Skip processing this item
    
    mapping, pr_string = preprocess_options_and_labels(op, lab, adapt_outcome)
    item['mapping_given_to_model'] = mapping

    # Handle the first response
    first_response = generate_and_process_response(model, scenario, pr_string, max_tokens, mapping, first_call=True)
    item['first response'] = first_response

    print('First response:', first_response )
    # Check if first_response is not None and is numeric
    if first_response is not None:
        try:
            # Attempt to convert first_response to a numeric type (float or int)
            numeric_first_response = float(first_response) 
            # If numeric, you can proceed with your logic for valid first_response
            # For example:
            item['first response'] = numeric_first_response
            if numeric_first_response > 5 or numeric_first_response < 1 :
                item['second response'] = 'Invalid first response'
                return 
        except ValueError:
            # If conversion to numeric type fails, set second_response to indicate invalid first_response
            item['second response'] = 'Invalid first response'
            return 
    else:
        # If first_response is None, directly set second_response as invalid
        item['second response'] = 'Invalid first response'
        return  # Skip to the next item

    second_response = generate_and_process_response(model, scenario, pr_string, max_tokens, mapping, first_call=False, numeric_first_response=numeric_first_response, type_experiment=type_experiment, num_ex = num_ex)
    item['second response'] = second_response
    print('Second response ', second_response)


def process_one_file(file, write_path, max_tokens, model, type_experiment, num_ex):


    """Function to process one file in the dataset directory."""
    full_json = load_data(file)

    for item in full_json:
        process_one_item(item, model, max_tokens, type_experiment=type_experiment, num_ex = num_ex)
        if model in claude_model_family:
            print('time')
            time.sleep(10)
    save_processed_data(full_json, write_path)



############################################### Process Functions ###############################################




def run_adaptive_prompting(model, run_name, type_experiment, num_ex = False,  max_tokens=100):

    """
    Loop to run each file in the dataset directory through the adaptive prompting process.
    Relies on dataset_generation.py having been run first to generate the dataset.

    Inputs:
    - model: str, name of the model to test the adaptive prompting call on.
    - run_name: str, name of the dataset, usually includes the name of the model used to generate the dataset.
               Note this model can differ from the model running the adaptive prompt.
    - max_tokens: int, maximum number of tokens for model response.
    - fewshot: bool, indicates whether to use few-shot learning for the second prompt.
    """
    # Assuming 'script_dir' is predefined as the directory of this script

    if model in tog_model_family:
        model_name = model.split('/')[1]
    else:
        model_name = model
        
    if type_experiment=='plain':
        print('Plain experiemnt ')
        file_dir = os.path.join(script_dir, "data", "processed", f'model--{model}', f'd_name--{run_name}')

    elif type_experiment=='fewshot':
        print('few shot')
        file_dir = os.path.join(script_dir, "data", f"processed_fewshot_{num_ex}", f'model--{model}', f'd_name--{run_name}')
    elif type_experiment=='chainofthought': 
        print('Chain of thought')
        file_dir = os.path.join(script_dir, "data", f"processed_chainofthought", f'model--{model}', f'd_name--{run_name}')
    else:
        raise Exception('Invalid type of experiment')    
    os.makedirs(file_dir, exist_ok=True)
    for category in ['helpful', 'harmless']:
        os.makedirs(os.path.join(file_dir, category), exist_ok=True)
                
    # Files to process
    topics_files = glob.glob(f'{script_dir}/data/dataset_with_adapt/d_name--{run_name}/*/*.json')
    
    if not topics_files:
        raise Exception("No files found. Please run dataset_generation.py first.")

    print(f'Starting to process {len(topics_files)} files.\n')
    
    # Loop over topics files (JSONs of topics), process each sub-scenario and save new JSON for each topic file
    for file in topics_files:

        file_name = os.path.basename(file)
        print(file_name)
        hh = os.path.basename(os.path.dirname(file))
        write_path = os.path.join(file_dir, hh, file_name)
        # Check if file exists
        if os.path.exists(write_path):
            print('File already exists')
            continue  # Skip existing files
        
        if file_name == "20--Art and Design.json":
            continue
        # Call the processing function with few-shot parameter
        process_one_file(file, write_path, max_tokens, model, type_experiment , num_ex)

    print('Run complete.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--model', type=str, required=True, help='model to test ')
    parser.add_argument('--run_name', type=str, required=True, help='name of the dataset to process')

    parser.add_argument('--type_experiment', type=str,  help='what type of experiment')
    parser.add_argument('--num_ex', type=int, required=False, help='number of examples for fewshot learning')
    parser.add_argument('--cot', type=bool, required=False, help='does experiment using chain of thought')
    args = parser.parse_args()
    
    model = args.model
    run_name = args.run_name
    type_experiment = args.type_experiment 
    num_ex = args.num_ex
    print(f'Starting model {model} and dataset {run_name} for experiment type {type_experiment} with {num_ex} examples. \n')
    run_adaptive_prompting(model = model, run_name = run_name, type_experiment= type_experiment, num_ex= num_ex )
    print(f'Run for {model} complete')


        



# def process_one_file(file , write_path , max_tokens, model  ):
#     """
#     Function to process one file in the dataset directory
#     Inputs:
#     file: str, path to the file to process
#     write_path: str, path to the desired output file 
#     """
#     full_json = load_data(file)

#     for item in full_json: 
#         # Extracting data from JSON item
#         op = item['options']
#         lab = item['labels']
#         scenario = item['scenario']
#         adapt_outcome = ast.literal_eval(str(item['adapt_response'].replace("'s", 'es')))
        
#         # print('Adapt outcome list:\n', adapt_outcome)
        
#         # Checking adapt outcome length
#         if len(adapt_outcome) != len(op):
#             raise Exception('Number of options and number of adapt outcomes do not match')
#         if len(adapt_outcome) != 5: 
#             raise Exception('Wrong number of adapt outcomes')
        
#         # Preprocessing options and labels
#         mapping, pr_string = preprocess_options_and_labels(op, lab, adapt_outcome)
#         item['mapping_given_to_model'] = mapping 

#         # Generating and running prompts
#         first_prompt = intention_prompt_first(scenario, pr_string)
        
#         if model in gpt_models:
#             first_response = run_api_call(first_prompt, model, max_tokens)
#         elif model in gpt_base_modesl:
#             first_response = run_api_call(first_prompt, model, max_tokens, base=True)
#         elif model in llama_model_family + mixtral_family:
#             first_response = call_llama(first_prompt, model)
        

#         # Extracting numbers from the first response
#         if model not in gpt_base_modesl:
#             numeric_first_response = extract_numbers_in_range(first_response)
#         else:
#             numeric_first_response = extract_numbers_in_range(first_response, base=True)
        
#         item['first response'] = numeric_first_response if numeric_first_response is not None else first_response
#         print(item['first response'])
#         if numeric_first_response is None:
#             # Skip to the next item if first response is invalid
#             item['second response'] = 'Invalid first response'
#             continue 
        
#         # Generating and running second prompts
#         adapt_sentence = mapping[numeric_first_response]['adapt_outcome']
        
#         if model in gpt_models + gpt_base_modesl:
#             second_prompt = intention_prompt_second(scenario, pr_string, adapt_sentence)
#         elif model in llama_model_family + mixtral_family:
#             second_prompt = intention_prompt_second_llama(scenario, pr_string, adapt_sentence)
        
#         if model in gpt_models:
#             second_response = run_api_call(second_prompt, model, max_tokens)
#         elif model in gpt_base_modesl:
#             second_response = run_api_call(second_prompt, model, max_tokens, base=True)
#         elif model in llama_model_family + mixtral_family:
#             second_response = call_llama(second_prompt, model)
        
#         if model not in gpt_base_modesl:
#             numeric_second_response = extract_numbers_in_range(second_response)
#         else:
#             numeric_second_response = extract_numbers_in_range(second_response, base=True)
        
#         item['second response'] = numeric_second_response if numeric_second_response is not None else second_response
#         print(item['second response'])

#     print(f'Starting to save file {file}')
#     json_arr_to_file(full_json, write_path, indent=4)
#     print('File saved. \n')



# def run_apative_prompting(model, run_name, max_tokens=100, fewshot=False ):
#     """
#     Loop to run each file in the dataset directory through the adaptive prompting process
#     Relies on dataset_generation.py having been run first to generate the dataset

#     Inputs:
#     model: str, name of the model to test the adaptive prompting call on 
#     run_name: str, name of the dataset, usually includes the name of the model used to generate the dataset 
#                 Note this model can differ from the model running the adaptive prompt. 
#     """
#     # Make dir to store processed data
#     file_dir = os.path.join(script_dir, "data", "processed", f'model--{model}', f'd_name--{run_name}')
#     os.makedirs(file_dir, exist_ok=True)
#     for f in  ['helpful', 'harmless']:
#         os.makedirs( os.path.join(file_dir, f) , exist_ok=True)
                
#     # Files to process
#     topics_files = glob.glob(f'{script_dir}/data/dataset_with_adapt/d_name--{run_name}/*/*.json')
#     if len(topics_files) == 0:
#         raise Exception("No files found. Please run dataset_generation.py first")

#     print(f'Starting to process {len(topics_files)} files \n')
#     # Loop over topics files (jsons of topics), process each sub scenario and save new json for each topic file 
#     for file in topics_files:
#         # check if file exists
#         print(file)
       
#         file_name = file.split('/')[-1]
#         hh  = file.split('/')[-2]
#         write_path = os.path.join(file_dir, hh, file_name )
        
#         #  check if file exists
#         if os.path.exists(write_path):
#             # print(f'File {write_path} already exists, skipping')
#             continue
#         process_one_file(file, write_path, max_tokens , model , fewshot)

#     print('Run complete')