import os
import json

example_path = '../examples/scenarios'
example_path = os.path.join(os.path.dirname(__file__), '../examples/scenarios')

def load_examples_h(h):
    """
    Load example files from the examples folder
    Inputs:
    h: helpful or harmless 
    """
    file_path = os.path.join(example_path, h + '.json')
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    else:
        print(f"File '{file_path}' does not exist or is not a JSON file.")
    

# def load_examples_all():
#     """
#     Load all examples from the examples folder
    
#     Returns list of helpful and harmless examples 
#     """
#     example_path = os.path.join(os.path.dirname(__file__), '../examples/scenarios')
#     print('Starting 4 pin examples')
#     file_names = os.listdir(example_path)
#     example = []
#     for file in file_names:
#         if file.endswith('.json'):
#             file_path = os.path.join(example_path, file)
#             with open(file_path, 'r') as file:
#                 example.append(json.load(file))
#     return example



def load_examples_all_5pin():
    """
    Load all examples from the examples folder
    
    Returns list of helpful and harmless examples 
    """
    print('Starting 5 pin examples')
    example_path = os.path.join(os.path.dirname(__file__), '../examples/scenarios/5pin')
    file_names = os.listdir(example_path)
    example = []
    for file in file_names:
        if file.endswith('.json'):
            file_path = os.path.join(example_path, file)
            with open(file_path, 'r') as file:
                example.append(json.load(file))
    return example
