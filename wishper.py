import sounddevice as sd
import soundfile as sf
import simpleaudio as sa
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from TTS.api import TTS
import openai
from openai import OpenAI

# Set up Groq-compatible client
client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1",
)

def query_groq(user_prompt,tokens=200):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or llama3/gemma
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=tokens
    )

    return response.choices[0].message.content



device = "cuda" if torch.cuda.is_available() else "cpu"

def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, recording, samplerate)
    print(f"Audio saved to {filename}")




def transcribe_audio(filename="input.wav"):
    model = whisper.load_model("tiny",device=device)  # Or "tiny" for faster inference or base for better accuracy
    print("Model loaded")
    result = model.transcribe(filename)
    return result["text"]



def speak(text):
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    tts.tts_to_file(text=text, file_path="response.wav")
    wave_obj = sa.WaveObject.from_wave_file("response.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()
    print("Response spoken and saved to response.wav")

# workes fine 
# need to make it run untill my user speaks not for fixed time
# need to speak the output according to length or question and it should stop when i intrrupt it. and listen me
# how big ans should an assitent give ?
def voice_assistant():
    record_audio()
    user_input = transcribe_audio()
    print("User said:", user_input)
    max_num = len(user_input.split())
    reply=query_groq(user_input)
    print(reply)
    speak(reply)




if __name__ == "__main__":
    voice_assistant()
