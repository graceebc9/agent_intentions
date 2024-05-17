import replicate
from dotenv import load_dotenv
import os

def load_hf_api_vars(): 
    load_dotenv()  # This loads the variables from .env
    REPLICATE_API_TOKEN =  os.getenv('REPLICATE_API_TOKEN')
    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    return client 

def convert_hf_statement(client, messages, max_tokens, model):
    print(f"Calling API with {model}")
    output = client.run(
        model,
        input = {
            "prompt": messages,
            "top_k": 1,
            "top_p": 0.01,
            "temperature": 0.01,
            "max_tokens": max_tokens,
            "prompt_template": "<s>[INST] {prompt} [/INST] ",
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
    )
    return output

def get_response_text(response):
    return response[0]


def run_hf_api_call(prompt, model):
    max_tokens = 1
    client = load_hf_api_vars()
    response = convert_hf_statement(client, prompt, max_tokens, model=model)
    content = get_response_text(response)
    return content 