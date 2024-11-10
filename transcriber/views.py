from django.shortcuts import render
from django.http import JsonResponse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa
import os
import traceback

# Load the Wav2Vec2 model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

def index(request):
    return render(request, 'transcriber/index.html')

def transcribe(request):
    if request.method == 'POST':
        try:
            audio_file = request.FILES.get('audio')
            if not audio_file:
                return JsonResponse({'error': 'No audio file provided'}, status=400)

            audio_path = f'temp_{audio_file.name}'
            with open(audio_path, 'wb') as f:
                f.write(audio_file.read())

            audio, sr = librosa.load(audio_path, sr=16000)
            input_values = tokenizer(audio, return_tensors="pt", padding="longest").input_values
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.decode(predicted_ids[0])

            os.remove(audio_path)

            return JsonResponse({'transcription': transcription})
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            traceback.print_exc()
            return JsonResponse({'error': 'An error occurred during transcription'}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)
