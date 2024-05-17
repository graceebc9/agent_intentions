import requests
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import os
import dotenv 


def call_gcd_model(int_prompt_first, model):
    """ 
    Convert a prompt to a llama api response
    """
    dotenv.load_dotenv()
    SCOPES = [os.getenv('SCOPES')]
    SERVICE_ACCOUNT_FILE = os.getenv(f'SERVICE_ACCOUNT_FILE_{model}')

    # Load credentials from the service account file with the specified SCOPES
    cred = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    # Create an authentication request
    auth_req = google.auth.transport.requests.Request()

    # Refresh the credentials
    cred.refresh(auth_req)

    # Obtain the bearer token
    bearer_token = cred.token

    project_id = os.getenv(f"PROJECT_ID_{model}")

    endpoint_id = os.getenv(f"ENDPOINT_ID_{model}")

    location = os.getenv("LOCATION")

    base_url = f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"

    format_prompt =  int_prompt_first[0]['content'] + '\n\n' + int_prompt_first[1]['content'] + '\n\n'  \
                    + "the correct answer :  "
    
    request_body = {
        "instances": [
            {
                "prompt": format_prompt,
                "max_tokens": 10,
            }
        ],
        "parameters" : {
        "temperature": 0.0,
        "maxOutputTokens": 10,
        "topK": 1,
        "topP": 0.0,
        }
    }

    full_url = base_url.format(project_id=project_id, endpoint_id=endpoint_id)

    headers = {
        "Authorization": "Bearer {bearer_token}".format(bearer_token=bearer_token),
        "Content-Type": "application/json"
    }

    # Send a POST request to the model endpoint
    resp = requests.post(full_url, json=request_body, headers=headers)
    
    option = extract_text_after_output(resp.json()['predictions'][0])

    # print(option)
    
    return option

def extract_text_after_output(text):
    """
    Extracts the text after "\nOutput:\n".

    Args:
    text (str): The input text.

    Returns:
    str: The extracted text after "\nOutput:\n", or None if not found.
    """
    separator = "\nOutput:\n"
    parts = text.split(separator)
    if len(parts) > 1:
        return parts[1].strip()
    else:
        raise Exception("No Output from llama")
