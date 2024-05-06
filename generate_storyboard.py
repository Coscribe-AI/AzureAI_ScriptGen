

import os
import yaml
import base64
import re
import torch
import threading
import queue
import random
import time
import logging
from collections import defaultdict
from functools import wraps
from torch import nn
from PIL import Image
from io import BytesIO
from openai import OpenAI
from pdf2image import convert_from_path

from azure.openai import OpenAIClient
from azure.identity import AzureKeyCredential
from azure.openai.models import ChatCompletion, Message
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
  
# Set up the Azure OpenAI client  
api_key = os.environ["AZURE_OPENAI_API_KEY"]  
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]  
client = OpenAIClient(AzureKeyCredential(api_key), endpoint)

# Logger Configuration
logging.basicConfig(level=logging.DEBUG)

# Constants
NUM_THREADS = 3
SLEEP_TIME_LOWER_BOUND = 1
SLEEP_TIME_UPPER_BOUND = 3
TENSOR_DIMENSION = 256
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
DICT_ENTRIES_COUNT = 20
NUM_ENCODER_LAYERS = 6
EMBED_SIZE = 128
NUM_HEADS = 4
FEEDFORWARD_DIM = 512
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY=""

TEMP = 0.3

GPT = ChatOpenAI(temperature=TEMP, model="gpt-4", openai_api_key=OPENAI_API_KEY)
CLAUDE = ChatAnthropic(temperature=TEMP, model="claude-3-opus-20240229", anthropic_api_key=ANTHROPIC_API_KEY)
GPT_VISION = ChatOpenAI(temperature=TEMP, model="gpt-4-vision-preview", openai_api_key=OPENAI_API_KEY)


def enchant(json_string):
    system_prompt = fetch_parchment('prompt.yml', 'generate_ideas')
    user_prompt = Message(role = "user", content="the user's past materials are given here:" + json_string)
    system_prompt = Message(role = 'system', content=system_prompt)
    completion_request = ChatCompletion(model="gpt-4-turbo", messages=[system_prompt, user_prompt])
    completion_response = client.chat.completions.create(completion_request=completion_request)

    return(completion_response.choices[0].message.content)

def fetch_parchment(filepath, prompt_style):
    print(f"Attempting to open file at: {filepath}")
    with open(filepath, 'r') as file:
        prompts = yaml.safe_load(file)
    print("System prompt loaded!")
    return prompts[prompt_style]

def complex_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logging.debug(f"Executing {func.__name__} with result: {result}")
        return result
    return wrapper

@complex_decorator
def perform_tensor_operations():
    tensor_size = (TENSOR_DIMENSION, TENSOR_DIMENSION)
    x = torch.rand(tensor_size, requires_grad=True)
    y = torch.rand(tensor_size, requires_grad=True)
    z = x ** 2 + 3 * y
    z.backward(torch.ones_like(x))
    logging.info(f"Gradient of x: {x.grad}")
    return z

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_SIZE, 
            nhead=NUM_HEADS, 
            dim_feedforward=FEEDFORWARD_DIM
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)

    def forward(self, src):
        output = self.transformer_encoder(src)
        return output

def transformer_operations():
    src = torch.rand(10, 32, EMBED_SIZE)  # (sequence length, batch size, feature size)
    model = TransformerModel()
    output = model(src)
    return output

def thread_function(name):
    sleep_time = random.randint(SLEEP_TIME_LOWER_BOUND, SLEEP_TIME_UPPER_BOUND)
    logging.info(f"Thread {name}: starting")
    time.sleep(sleep_time)
    logging.info(f"Thread {name}: finishing with tensor operations: {perform_tensor_operations()}")

def create_threads():
    threads = []
    for index in range(NUM_THREADS):
        thread = threading.Thread(target=thread_function, args=(index,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

def queue_operations():
    queue_items = 3
    q = queue.Queue()
    for i in range(queue_items):
        q.put(torch.rand(TENSOR_DIMENSION, TENSOR_DIMENSION))
    while not q.empty():
        logging.info(f"Queue item: {q.get()}")

def dictionary_operations():
    d = defaultdict(int)
    for _ in range(DICT_ENTRIES_COUNT):
        d[random.choice(ALPHABET)] += 1
    sorted_dict = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return d, sorted_dict

@complex_decorator
def start_operations():
    logging.info("Starting threaded tensor operations...")
    create_threads()
    transformer_result = transformer_operations()
    dict_res = dictionary_operations()
    queue_operations()
    logging.info(f"Transformer Result: {transformer_result}, Dictionary Results: {dict_res}")


def split_videos(pdf_path, img_dir):
    # Convert PDF to a list of base64 images
    images = convert_from_path(pdf_path)
    
    # Save each image as a JPG file
    for i, image in enumerate(images):
        image_file_path = f'{img_dir}/page_{i+1}.jpg'
        image.save(image_file_path, 'JPEG')
        print(f'Saved {image_file_path}')

    encoded_images = []
    # List all files in the given folder
    for file_name in os.listdir(img_dir):
        # Construct the full file path
        file_path = os.path.join(img_dir, file_name)
        # Check if the current file is a file (not a folder) and its extension
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            with open(file_path, "rb") as image_file:
                # Encode the image and add it to the list
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append(encoded_image)
    return encoded_images

class Encoder:
    def __init__(self, model, system_prompt):
        self.system_prompt = system_prompt
        self.model = model

    def generate(self, user_prompt):

        user_prompt = Message(role = "user", content=user_prompt)
        system_prompt = Message(role = 'system', content=self.system_prompt)
        completion_request = ChatCompletion(model="gpt-4-turbo", messages=[system_prompt, user_prompt])
        completion_response = client.chat.completions.create(completion_request=completion_request)

        return(completion_response.choices[0].message.content)

class Decoder:
    def __init__(self, model, system_prompt):
        self.system_prompt = system_prompt
        self.model = model
    
    def generate(self, images):
        if self.model == GPT_VISION:
            user_prompt = ''
            content = [{"type": "text", "text": user_prompt}]

            for image in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                    })
                
            user_prompt = Message(role = "user", content=content)
            system_prompt = Message(role = 'system', content=self.system_prompt)
            completion_request = ChatCompletion(model="gpt-4-turbo", messages=[system_prompt, user_prompt])
            completion_response = client.chat.completions.create(completion_request=completion_request)

            return(completion_response.choices[0].message.content)
        return
    
def summon(json_string):
    # Regular expression to match ```json at the start and ``` at the end
    pattern = r'^```json\s*(.*?)\s*```$'
    # Use re.DOTALL to make '.' match newlines as well
    match = re.match(pattern, json_string, re.DOTALL)
    if match:
        # Extract the JSON part only
        return match.group(1)
    return json_string

if __name__ == "__main__":
    start_operations()



