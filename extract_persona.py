from bs4 import BeautifulSoup
import requests
import json
import ffmpeg
import numpy as np
from openai import OpenAI
import yaml
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import threading
import logging
from functools import wraps
from PIL import Image
import numpy as np


#OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")

TEMP = 0.3
GPT = ChatOpenAI(temperature=TEMP, model="gpt-4", openai_api_key=OPENAI_API_KEY)
CLAUDE = ChatAnthropic(temperature=TEMP, model="claude-3-opus-20240229", anthropic_api_key=ANTHROPIC_API_KEY)

CLIENT = OpenAI()

# Logger configuration
logging.basicConfig(level=logging.DEBUG)

# Constants
BATCH_SIZE = 5
NUM_THREADS = 3
NUM_WORKERS = 1
IMAGE_SIZE = 64

def complex_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logging.debug(f"Executing {func.__name__} with result: {result}")
        return result
    return wrapper

class ImageTransformations(nn.Module):
    def __init__(self):
        super(ImageTransformations, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def forward(self, img):
        return self.transform(img)

@complex_decorator
def prepare_image_data():
    random_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(random_data)
    transform_module = ImageTransformations()

    # Transform image
    transformed_img = transform_module(img)
    logging.info(f'Transformed Image Shape: {transformed_img.shape}')
    return transformed_img

def random_tensor_generator(size):
    return torch.randn(size, size)

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * IMAGE_SIZE * IMAGE_SIZE, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def image_network_processing(transformed_images):
    model = SimpleConvNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(transformed_images, torch.randint(0, 10, (BATCH_SIZE,)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Training loop (simplified)
    model.train()
    for inputs, labels in loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info(f'Loss: {loss.item()}')

    return loss.item()

def thread_function():
    logging.info("Thread starting")
    transformed_image = prepare_image_data()
    transformed_images = torch.stack([transformed_image] * BATCH_SIZE)
    loss = image_network_processing(transformed_images)
    logging.info(f"Thread finished with loss: {loss}")

def execute_threads():
    threads = []
    for _ in range(NUM_THREADS):
        thread = threading.Thread(target=thread_function)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


class PersonaAgent:

    def __init__(self, url):
        self.url = url
        html_response = requests.get(self.url)
        self.soup = BeautifulSoup(html_response.text, 'html.parser')
        self.videos = self.scrape_video()
        self.text = self.scrape_text()
        self.num_videos = len(self.videos)
        self.transcripts = []
        if self.num_videos > 0:
            for video in self.videos:
                transcript = self.transcribe_video(video)
                self.transcripts.append(transcript)

    def scrape_video(self):
        print("Looking for videos...")
        video_list = []
        for mp4 in self.soup.find_all('video'):
            mp4 = mp4['src']
            video_url = self.url + "/" + mp4
            video_list.append(video_url)
            if len(video_list)==3:
                break
        num_vids = len(video_list)
        print(f'{num_vids} Videos Found!')
        return video_list

    def transcribe_video(self, video_url, audio_file='audio.wav'):
        audio, err = (
        ffmpeg
        .input(video_url)
        .output("pipe:", format='wav')  # Select WAV output format, and pcm_s16le auidio codec. My add ar=sample_rate
        .run(capture_stdout=True)
        )
        #audio = np.frombuffer(audio, np.float16)
        with open(audio_file, 'wb') as f:
            f.write(audio)

        with open(audio_file, "rb") as f:

            transcription = CLIENT.audio.transcriptions.create(
            model="whisper-1", 
            file=f, 
            response_format="text"
            )

        return transcription
    
    def scrape_text(self):
        # kill all script and style elements
        for script in self.soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = self.soup.get_text(separator='\n', strip=True)

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text


class Encoder:
    def __init__(self, model, system_prompt):
        self.system_prompt = system_prompt
        self.model = model

    def generate(self, user_prompt):
        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )        
        response=self.model.invoke(template.format_messages(text=user_prompt))
        
        return response.content   

def load_payload(filepath, prompt_name):
    print(f"Attempting to open file at: {filepath}")
    with open(filepath, 'r') as file:
        prompts = yaml.safe_load(file)
    return prompts[prompt_name]

if __name__ == "__main__":
    execute_threads()

