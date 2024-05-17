from openai import OpenAI
# use .env file to store api keys 
from dotenv import load_dotenv
import os


def load_api_vars(): 
    load_dotenv()  # This loads the variables from .env
    OPENAI_API_KEY=  os.getenv('OENAI_API_KEY')
    ORGANIZATION_ID =  os.getenv('ORGANIZATION_ID')
    client = OpenAI(api_key=OPENAI_API_KEY, organization=ORGANIZATION_ID)
    return client 

def convert_statement(client, messages, max_tokens, model, base=False):
    print(f"Calling API with {model}")
    if not base:
        x = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
        )
    else:
        prompt = messages[0]['content'] + "\n" + messages[1]['content'] \
                + "\n" + "The correct answer to the above Question: "
        x = client.completions.create(
            model= model,
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0
        )
    return x

def get_response_text(response, base=False):
    if not base:
        return response.choices[0].message.content
    else:
        return response.choices[0].text


def run_api_call(prompt, model, max_tokens = 150, base=False):
    
    client = load_api_vars()
    response = convert_statement(client, prompt, max_tokens, model=model, base=base)
    content = get_response_text(response, base=base)
    return content 


