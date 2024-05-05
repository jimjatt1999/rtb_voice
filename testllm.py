import sounddevice as sd
import numpy as np
import wavio
import whisper
import requests
from rtb_rest import GoalDelta
from json import loads

# Function to list all available devices
def list_devices():
    print(sd.query_devices())

# Function to record audio with stop capability
def record_audio(duration=10, sample_rate=44100, channels=1):
    try:
        print("Press 'Enter' at any time to stop recording.")
        # Non-blocking recording
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            blocking=False,
        )
        input()  # Wait for Enter key to stop recording early
        sd.stop()
        print("Recording stopped.")
        return recording
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to save the recording
def save_audio(recording, filename="output.wav", sample_rate=44100):
    wavio.write(filename, recording, sample_rate, sampwidth=2)
    print(f"Audio saved as {filename}")

# Function to transcribe audio using OpenAI's Whisper model
def transcribe_audio(file_path):
    print("Transcribing...")
    model = whisper.load_model("base")  # Choose model size as needed: "tiny", "small", "medium", "large"
    result = model.transcribe(file_path)
    transcription = result["text"]
    print("Transcription completed.")
    return transcription

# Function to send a request to the language model
def send_llm_request(prompt):
    system_prompt = """
For example, if the user says 'move forward 5cm', the response should be (0.05, 0, 0). If the user says 'move up 3cm', the response should be (0, 0.03, 0).
Understand simple directions like forward, back, up, and down as movements along the x-axis and y-axis respectively. Just return the tuple nothing else. return just the tuple no text nothing else just values (x, y, z) no text!!!!! """
    data = {
        "model": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",  # Model identifier is required
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 20,
        "stream": False,
    }

    response = requests.post(url="http://localhost:1234/v1/chat/completions", json=data)
    return response.text

def update(x, y, z):
    requests.post( url = "http://0.0.0.0:8000/update_goal", json = GoalDelta(x=-x,y=y,z=z).model_dump())
    
# Main function to handle the workflow
def main():
    requests.post( url = "http://0.0.0.0:8000/run" )
    list_devices()  # List all devices to find the correct one and its capabilities
    sample_rate = 44100  # Sample rate of the audio
    duration = 10  # Duration of the recording in seconds

    # Record and save audio
    recording = record_audio(duration, sample_rate)
    if recording is not None:
        audio_file = "output.wav"
        save_audio(recording, audio_file, sample_rate)

        # Transcribe the audio
        transcription = transcribe_audio(audio_file)
        print("Transcribed Text:", transcription)

        # Send the transcribed text as a prompt to the language model
        response = loads(send_llm_request(transcription))
        print("Model Response:", response)
        response_content = response["choices"][0]["message"]["content"]
        print(response_content)
        numbers = [float(num) for num in response_content.replace(' ', '').replace('(', '').replace(')', '').split(',')]
        update(*numbers)



if __name__ == "__main__":
    main()



