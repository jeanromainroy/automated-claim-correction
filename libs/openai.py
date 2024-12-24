"""
This script is used to interact with the OpenAI API.
"""

# Libraries
import requests
import json
import base64

# Config
from config import OPENAI_API_KEY, OPENROUTER_API_KEY, ENGINE

# Constants
OPENAI_ENDPOINT_CHAT = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL_LARGE = "gpt-4o"
OPENAI_MODEL_SMALL = "gpt-4o-mini"

OPENROUTER_ENDPOINT_CHAT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_LARGE = "openai/chatgpt-4o-latest"
OPENROUTER_MODEL_SMALL = "openai/gpt-4o-mini"

LOCAL_ENDPOINT_CHAT = "http://localhost:8080/completion"
LOCAL_MODEL_LARGE = "mistralai/mistral-large-2411"
LOCAL_MODEL_SMALL = "mistralai/mistral-large-2411"

OPENAI_ENDPOINT_EMBEDDINGS = "https://api.openai.com/v1/embeddings"
OPENAI_MODEL_EMBEDDING = "text-embedding-3-small"

REQ_TIMEOUT_SEC = 600

# LLM Engine
if ENGINE == 'openai':
    print(f'LLM Inference: [OpenAI]')
    ENDPOINT_CHAT = OPENAI_ENDPOINT_CHAT
    API_KEY = OPENAI_API_KEY
    MODEL_LARGE = OPENAI_MODEL_LARGE
    MODEL_SMALL = OPENAI_MODEL_SMALL

elif ENGINE == 'openrouter':
    print(f'LLM Inference: [OpenRouter]')
    ENDPOINT_CHAT = OPENROUTER_ENDPOINT_CHAT
    API_KEY = OPENROUTER_API_KEY
    MODEL_LARGE = OPENROUTER_MODEL_LARGE
    MODEL_SMALL = OPENROUTER_MODEL_SMALL

elif ENGINE == 'local':
    print(f'LLM Inference: [Local]')
    ENDPOINT_CHAT = LOCAL_ENDPOINT_CHAT
    API_KEY = None
    MODEL_LARGE = LOCAL_MODEL_LARGE
    MODEL_SMALL = LOCAL_MODEL_SMALL

else:
    print('ERROR: Invalid LLM engine')
    exit(1)


# Set the def complete function based on local or remote
def complete( prompt, max_tokens=8, model_large=False, json_mode=False, temperature=0.0 ):
    if ENGINE == 'local':
        return complete_local( prompt, max_tokens, model_large=model_large, json_mode=json_mode, temperature=temperature )
    else:
        return complete_openai( prompt, max_tokens, model_large=model_large, json_mode=json_mode, temperature=temperature )


# Function to encode the image
def encode_image(image_path):
    """
    Encodes the image as a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def format_prompt_for_model(
        user_message,
        system_instructions="You are an annotation assistant that evaluates content using a structured guide and answers each question with 'Yes' or 'No' only.",
        prefilled_response=None
    ):

    # Build the system prompt section if provided
    system_prompt = f"[SYSTEM_PROMPT] {system_instructions.strip()}[/SYSTEM_PROMPT]" if system_instructions else ""

    # Build the user message prompt
    user_prompt = f"[INST] {user_message.strip().replace(' ', '_')}[/INST]"

    # Concatenate the parts
    full_prompt = f"{system_prompt}{user_prompt}"

    # Append the prefilled response if available
    if prefilled_response:
        full_prompt += f" {prefilled_response.strip()}"

    # Replace whitespace with '_'
    # full_prompt = full_prompt.replace(' ', '_')

    # Return the final formatted prompt
    return full_prompt.strip()



def complete_local( prompt, max_tokens=8, model_large=False, json_mode=False, temperature=0.0 ):
    # ssh -L 8882:localhost:8080 -p 22 -N dev-local

    # Define the payload
    payload = {
        'prompt': format_prompt_for_model(prompt),
        'n_predict': max_tokens,
        'temperature': temperature,
        'top_p': 1.0,
        'top_k': 0.0,
        'cache_prompt': True
    }

    # Headers
    headers = { 'Content-Type': 'application/json' }

    # Send the POST request
    try:
        # Send the POST request
        response = requests.post(ENDPOINT_CHAT, headers=headers, json=payload, timeout=REQ_TIMEOUT_SEC)
        response = response.json()

        # Validate
        if 'content' not in response:
            print('ERROR: Response does not contain content', response)
            return None

        # Extract and sanitize completion
        completion = response['content'].strip()
        completion_clean = completion.strip('"') if len(completion) > 0 else 'N/A'

        return completion_clean

    except KeyboardInterrupt:
        print('\nOperation cancelled by user')
        raise
    except requests.exceptions.RequestException as e:
        print(f'ERROR: Request failed: {str(e)}')
        return None
    except json.JSONDecodeError:
        print('ERROR: Could not parse response')
        return None
    except Exception as e:
        print(f'ERROR: Unexpected error: {str(e)}')
        return None


def complete_openai(prompt, max_tokens=8, model_large=False, json_mode=False, temperature=0.0):

    # Define the payload
    payload = {
        "messages": [ {"role": "user", "content": prompt} ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "model": MODEL_LARGE if model_large else MODEL_SMALL,
        "seed": 42
    }

    # If JSON mode is enabled
    if json_mode:
        payload["response_format"] = { "type": "json_object" }

    # Headers
    headers = { 'Content-Type': 'application/json', 'Authorization': f"Bearer {API_KEY}" }

    try:
        # Send the POST request
        response = requests.post(
            ENDPOINT_CHAT,
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )

        # Raise an error for bad status codes
        response.raise_for_status()

        # Parse the JSON response
        response_data = response.json()

        # Validate response structure
        if 'choices' not in response_data:
            print('ERROR: Response does not contain choices', response_data)
            return None

        if not response_data['choices'] or 'message' not in response_data['choices'][0]:
            print('ERROR: Invalid response structure', response_data)
            return None

        # Extract and clean completion
        completion = response_data['choices'][0]['message']['content'].strip()

        # If JSON mode, parse JSON
        if json_mode:
            completion = json.loads(completion)

        return completion

    except KeyboardInterrupt:
        print('\nOperation cancelled by user')
        raise
    except requests.exceptions.RequestException as e:
        print(f'ERROR: API request failed: {str(e)}')
        return None
    except json.JSONDecodeError as e:
        print(f'ERROR: Could not parse response: {str(e)}')
        return None
    except KeyError as e:
        print(f'ERROR: Unexpected response format: {str(e)}')
        return None
    except Exception as e:
        print(f'ERROR: Unexpected error: {str(e)}')
        return None


# Function to send a text AND image prompt and get a completion from the OpenAI API
def complete_with_image( prompt, image_path, max_tokens=8, json_mode=False ):

    # Encode image to base64 format
    base64_image = encode_image(image_path)

    # Define the payload
    payload = {
        "messages": [
            {
                "role": "user", "content": [
                    { "type": "text", "text": prompt },
                    { "type": "image_url", "image_url": { "detail": "high", "url": f"data:image/jpeg;base64,{base64_image}" } }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "model": OPENAI_MODEL_LARGE,
        "seed": 42
    }

    # If JSON mode is enabled
    if json_mode:
        payload["response_format"] = { "type": "json_object" }

    # Headers
    headers = { 'Content-Type': 'application/json', 'Authorization': f"Bearer {OPENAI_API_KEY}" }

    # Send the POST request
    response = requests.post(OPENAI_ENDPOINT_CHAT, headers=headers, data=json.dumps(payload))
    try:
        response = response.json()
    except:
        print('ERROR: Could not parse response')
        return None

    # Validate
    if 'choices' not in response:
        print('ERROR: Response does not contain choices')
        print(response)
        return None

    # Extract completion
    completion = response['choices'][0]['message']['content']
    completion = completion.strip()

    return completion


# Function to send a text to the OpenAI API and get the embedding
def embed( text ):
    """
    This function sends a text to the OpenAI API and returns the embedding.
    """

    # Define the payload
    payload = {
        "input": text,
        "model": OPENAI_MODEL_EMBEDDING,
    }

    # Headers
    headers = { 'Content-Type': 'application/json', 'Authorization': f"Bearer {OPENAI_API_KEY}" }

    # Send the POST request
    response = requests.post(OPENAI_ENDPOINT_EMBEDDINGS, headers=headers, data=json.dumps(payload))
    try:
        response = response.json()
    except:
        print('ERROR: Could not parse response')
        return None

    # Validate
    if 'data' not in response:
        print('ERROR: Response does not contain data')
        print(response)
        return None

    # Validate
    embedding = response['data'][0]['embedding']

    return embedding
