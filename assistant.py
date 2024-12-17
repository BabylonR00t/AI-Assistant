import os
import time
import re
import threading
import cv2
import pyaudio
import speech_recognition as sr
from PIL import ImageGrab, Image
from faster_whisper import WhisperModel
from groq import Groq
from openai import OpenAI
import google.generativeai as genai

# set a wake up word
wake_word = 'AI-listen'

# API keys
groq_client = Groq(api_key="<YOUR_GROQ_API_KEY>")
genai.configure(api_key='<YOUR_GOOGLE_GENAI_API_KEY>')
openai_client = OpenAI(api_key='<YOUR_OPENAI_API_KEY>')

# system Message for the Assistant
 system_msg = (
        'You are an AI function calling model. You will determine whether taking a screenshot, '
        'capturing the webcam, or calling no functions is best for a voice assistant to respond '
        'to the user’s prompt. The webcam can be assumed to be a normal laptop webcam facing the user. '
        'You will respond with only one selection from this list: ["take screenshot", "capture webcam", "None"]\n'
        'Do not respond with anything but the most logical selection from that list with no explanations. '
        'Format the function call name exactly as I listed.'
    )


convo = [{'role': 'system', 'content': system_message}]

# Generation Configuration
generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

# Safety Settings
safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
]


model = genai.GenerativeModel(
    'gemini-1.5-flash-latest',
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Whisper Model Configuration
num_cores = os.cpu_count()
whisper_size = 'base'  # Change to 'large' for better accuracy
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)


r = sr.Recognizer()
source = sr.Microphone()

# Functions
def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\nIMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append({'role': 'assistant', 'content': response.content})
    return response.content

def function_call(prompt):
    system_msg = (
        'You are an AI function calling model. You will determine whether taking a screenshot, '
        'capturing the webcam, or calling no functions is best for a voice assistant to respond '
        'to the user’s prompt. The webcam can be assumed to be a normal laptop webcam facing the user. '
        'You will respond with only one selection from this list: ["take screenshot", "capture webcam", "None"]\n'
        'Do not respond with anything but the most logical selection from that list with no explanations. '
        'Format the function call name exactly as I listed.'
    )
    function_convo = [{'role': 'system', 'content': system_msg},
                      {'role': 'user', 'content': prompt}]
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content.strip()

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)
    return path

def web_cam_capture():
    try:
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            raise ValueError('Error: Camera did not open!!')
        path = 'webcam.jpg'
        ret, frame = webcam.read()
        if ret:
            cv2.imwrite(path, frame)
        else:
            raise ValueError('Error: Failed to capture image from webcam.')
        return path
    except Exception as e:
        print(e)
        return None
    finally:
        webcam.release()

def vision_prompt(prompt, photo_path=None):
    if photo_path:
        img = Image.open(photo_path)
        vision_prompt_msg = (
            'You are the vision analysis AI that provides semantic meaning from images to provide context '
            'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
            'to the user. Instead, take the user prompt input and try to extract all meaning from the photo '
            'relevant to the user prompt. Then generate as much objective data about the image for the AI '
            f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
        )
        response = model.generate_content([vision_prompt_msg, img])
        return response.text
    else:
        return None

def speak(text):
    player = pyaudio.PyAudio()
    stream = player.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    with openai_client.audio.speech.with_streaming_response.create(
            model='tts-1',
            voice='fable',
            response_format='pcm',
            input=text,
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            stream.write(chunk)
    stream.stop_stream()
    stream.close()
    player.terminate()

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,?!.]*(.*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    if match:
        prompt = match.group(1).strip()
        return prompt if prompt else None
    return None

def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        print(f'USER: {clean_prompt}')
        call = function_call(clean_prompt)
        visual_context = None

        if 'take screenshot' in call:
            print('Taking screenshot.')
            photo_path = take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path=photo_path)
        elif 'capture webcam' in call:
            print('Capturing webcam.')
            photo_path = web_cam_capture()
            if photo_path:
                visual_context = vision_prompt(prompt=clean_prompt, photo_path=photo_path)

        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        print(f'ASSISTANT: {response}')
        speak(response)

def start_listening():
    def listen_in_background():
        with source as s:
            r.adjust_for_ambient_noise(s, duration=1)
            print('\nSay', wake_word, 'followed with your prompt. \n')
            while True:
                audio = r.listen(s)
                callback(r, audio)

    listen_thread = threading.Thread(target=listen_in_background, daemon=True)
    listen_thread.start()

    while True:
        time.sleep(0.5)

if __name__ == "__main__":
    start_listening()
