import base64, os
from transformers import AutoTokenizer
import boto3
import json
from botocore.exceptions import ClientError
from botocore.client import Config

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

def read_data(filename, **kwargs):
    subdir = kwargs.get('subdir', '')
    datatype = kwargs.get('datatype', 'json')
    datadir = os.environ.get('RAGAS_DATA_DIR', None)

    if datadir is None:
        print("Error: RAGAS_DATA_DIR is not set.")
        return None

    if subdir:
        datadir = os.path.join(datadir, subdir)
    filepath = os.path.join(datadir, filename)

    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return None

    try:
        with open(filepath, 'r') as file:
            if datatype == 'json':
                return json.load(file)
            elif datatype == 'text':
                return file.read()
            else:
                print(f"Error: Unsupported datatype '{datatype}'. Use 'json' or 'text'.")
                return None
            pass
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def save_data(filename, string, **kwargs):
    subdir = kwargs.get('subdir', '')
    overwrite = kwargs.get('overwrite', False)
    datadir = os.environ.get('RAGAS_DATA_DIR', None)
    if datadir is None:
        print("Error: RAGAS_DATA_DIR is not set.")
        return False

    if subdir:
        datadir = os.path.join(datadir, subdir)

    #create datadir 
    try:
        os.makedirs(datadir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {datadir}: {e}")
        return False

    filepath = os.path.join(datadir, filename)
    if not os.path.exists(filepath) or overwrite:
        try:
            with open(filepath, 'w') as file:
                file.write(string)
            return True
        except Exception as e:
            print(f"Error writing to file {filepath}: {e}")
    else:
        print(f"File {filepath} already exists and overwrite is disabled.")
    return False


#Encode a string
def encode_string(input_string):
    """
    encode a string. replace / with _jjj_
    """
    string_bytes = input_string.encode('utf-8')
    encoded_bytes = base64.b64encode(string_bytes)
    encoded_string = encoded_bytes.decode('utf-8')
    encoded_string = encoded_string.replace('/','_jjj_')
    return encoded_string

#Decode a string
def decode_string(encoded_string):
    """
    decode given encoded_string. if there is '_jjj_' in encoded_string, first replace to '/'
    """
    encoded_string = encoded_string.replace('_jjj_', '/')
    encoded_bytes = encoded_string.encode('utf-8')
    decoded_bytes = base64.b64decode(encoded_bytes)
    decoded_string = decoded_bytes.decode('utf-8')
    return decoded_string

def Bedrock_Meta_Llama(model_id, prompt, **kwargs):
    read_timeout = kwargs.get('read_timeout', 1800)
    region_name = kwargs.get('region_name', 'us-west-2')
    maxTokens = kwargs.get('maxTokens', 2048)
    temperature = kwargs.get('temperature', 0)
    topP = kwargs.get('topP', 0.9)

    config = Config(read_timeout=read_timeout)
    client = boto3.client("bedrock-runtime", region_name=region_name, config=config)

    conversation = [ { "role": "user", "content": [{"text": prompt}] } ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens":maxTokens, "temperature":temperature, "topP":topP},
            additionalModelRequestFields={}
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        print(f"GOT RESPONSE: {type(response_text)} - #response_text = {len(response_text)}")
        return response_text
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return "```[]```"

def CerebrasLLM(model_id, prompt, **kwargs):
    from langchain_cerebras import ChatCerebras
    max_tokens = kwargs.get('max_tokens', 8192)
    temperature = kwargs.get('temperature', 0)

    messages = [("human", prompt)]
    llm = ChatCerebras(model=model_id, temperature=temperature, max_tokens=max_tokens)
    try:
        ai_msg = llm.invoke(messages)
        print(f"GOT RESPONSE: - #ai_msg.content = {len(ai_msg.content)}")
        return ai_msg.content
    except Exception as exp:
        print(f"JJJJ:  CerebrasLLM ERROR: {exp}")
        return "```[]```"


