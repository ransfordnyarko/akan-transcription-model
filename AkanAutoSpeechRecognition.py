import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa

def record_audio(filename, duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait() 
    print("Recording finished.")
    write(filename, sample_rate, audio)

def play_audio(filename):
    data, samplerate = sf.read(filename)
    sd.play(data, samplerate)
    sd.wait() 


def transcribe_audio(processor, model, filename):
    # Load audio file
    print('Loading file')
    audio, rate = librosa.load(filename, sr=16000)

    # Preprocess and transcribe
    print("Transcribing")
    input_values = processor(audio, sampling_rate=rate, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription



if __name__ == "__main__":
    repo_name = 'ransfordnyarko/wav2vec2-large-xls-r-1b-asante-twi'
    tokenizer = 'ransfordnyarko/wav2vec2-large-xls-r-300m-asante-twi'
    processor = Wav2Vec2Processor.from_pretrained(tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(repo_name)
    # record_audio("my_audio.wav", duration=5)
    # play_audio("my_audio.wav")
    
    transcription = transcribe_audio(processor=processor, model=model, filename="my_audio.wav")
    print("Transcription:", transcription)