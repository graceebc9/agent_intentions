import dotenv 
from llamaapi import LlamaAPI
import json
import os
import anthropic
import together

def get_llama_api_vars():
    """ 
    Load llama api token and return llama object
    """
    dotenv.load_dotenv()
    return LlamaAPI(os.getenv('LLAMA_API_TOKEN'))


def call_llama(int_prompt_first, model):
    """ 
    Convert a prompt to a llama api response
    """
    # print(int_prompt_first)
    api_request_json = {
                        "model": model,
                        "messages": [
                        int_prompt_first[0],
                        int_prompt_first[1],
                        ], 
                        "max_tokens" : 300,
                        "temperature": 0.0,
                        "top_p": 0.0,
                        }
    llama = get_llama_api_vars()
    # print(api_request_json)
    response = llama.run(api_request_json)
    # print(json.dumps(response.json(), indent=2))
    output = response.json()['choices'][0]['message']['content']
    # print(output)
    return output 



def get_claude_api_vars():
    """
    Load Claude API token and return Anthropic client object
    """
    dotenv.load_dotenv()
    api_key = os.getenv('CLAUDE_API_KEY')
    return anthropic.Client(api_key=api_key)


def call_claude(int_prompt_first, model, max_tokens = 10):
    """
    Convert a prompt to a Claude API response
    """
    client = get_claude_api_vars()
    formatted_prompt = (
        "Human: " + int_prompt_first[0]['content'] + '\n\n' +
        "Human: " + int_prompt_first[1]['content'] + '\n\n' +
        "Assistant:"
    )
    response = client.completions.create(
        model=model,
        prompt=formatted_prompt,
        max_tokens_to_sample=max_tokens,
        temperature=0,
    )
    output = response.completion
    return output


def call_claude3(int_prompt_first, model, max_tokens = 200):
    """
    Convert a prompt to a Claude API response
    """
    client = get_claude_api_vars()
    # print(int_prompt_first)
    message = client.messages.create(
        model=model,
        max_tokens= max_tokens,
        temperature= 0.0,
        top_k = 1,
        system = int_prompt_first[0]['content'],
        messages=[
            {"role": "user", "content": int_prompt_first[1]['content']}
        ]
    )
    # print(message.content[0].text)
    return message.content[0].text

def get_together_api_vars():
    """ 
    Load together api token and return together object
    """
    dotenv.load_dotenv()
    API_Key = os.getenv('TOGETHER_API_KEY')
    together.api_key = API_Key



def call_togetherai(int_prompt_first, model, max_tokens = 100) :
    get_together_api_vars()
    # print(int_prompt_first)
    response = together.Complete.create(
                                    model = model,
                                    prompt= int_prompt_first[0]['content'] + '\n\n' +  
                                            int_prompt_first[1]['content'] + '\n\n' +
                                            "the correct answer:",
                                    max_tokens = max_tokens,
                                    temperature = 0,
                                    top_p = 0.1,
                                    top_k = 1,
                                    logprobs = 1
                                    )
    print(response)
    return response['choices'][0]['text']