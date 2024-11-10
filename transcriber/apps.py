from django.apps import AppConfig
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

class TranscriberConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'transcriber'

# Load Wav2Vec2 model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Function to process audio and generate transcription
def transcribe_audio(audio_file):
    try:
        # Load and resample the audio file
        audio, sr = librosa.load(audio_file, sr=16000)
        
        if sr != 16000:
            print("Warning: Audio is not at 16kHz. Resampling might have failed.")
        
        print(f"Audio Sample Rate: {sr}")
        print(f"Audio Duration: {len(audio) / sr:.2f} seconds")

        # Trim silence
        audio, _ = librosa.effects.trim(audio)

        # Tokenize and make predictions
        input_values = tokenizer(audio, return_tensors="pt", padding="longest").input_values
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        # Decode predicted IDs to text
        transcription = tokenizer.decode(predicted_ids[0])
        print("Transcription result:", transcription)
        return transcription

    except Exception as e:
        print("Error during transcription:", str(e))
        return "Error in audio processing"
