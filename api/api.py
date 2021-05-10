from flask import Flask
from flask import request
from flask import send_file, send_from_directory, safe_join, abort
from transformers import pipeline, set_seed, Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import torch
import json
import base64
import numpy as np
from scipy.io.wavfile import write

generator = pipeline('text-generation', model='gpt2')
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

print("Imported gpt and wav2vec2")

tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2', pretrained=False)
checkpoint = torch.load("tacotron2_statedict.pt", map_location='cpu')
tacotron2.load_state_dict(checkpoint['state_dict'])
tacotron2.eval()

print("Imported Tacotron")

waveglow = torch.load("waveglow_256channels_universal_v5.pt", map_location='cpu')['model']
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.eval()

print("Import everything!")

#audio_input, rate = librosa.load("download.wav", sr=16000)

# transcribe
#input_values = processor(audio_input, return_tensors="pt").input_values
#logits = model(input_values).logits


app = Flask(__name__)

@app.route('/api/image/', methods=['GET'])
def get_img():
    filename = request.args.get('filename')
    return send_file("images/" + filename + ".png", mimetype='image/png')


@app.route('/api/asr', methods=['POST'])
def get_asr():
    content = request.get_json()
    ans = base64.b64decode(bytes(content["message"], 'utf-8'))

    with open("temp.wav", "wb") as fh:
        fh.write(ans)

    audio_input, rate = librosa.load("temp.wav", sr=16000)
    # transcribe
    input_values = processor(audio_input, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].capitalize()

    print("The transcription was:", transcription)

    return {'string': transcription}


@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('sentence')

    filename = './audio/tts_output.wav'

    # preprocessing
    sequence = np.array(tacotron2.text_to_sequence(
        text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence)

    # run the models
    with torch.no_grad():
        _, mel, _, _ = tacotron2.infer(sequence)
        audio = waveglow.infer(mel)
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050

    write(filename, rate, audio_numpy)

    return send_file(filename, mimetype="audio/mpeg")


@app.route('/api/transformer', methods=['POST'])
def get_current_response():
    data = request.get_json()
    print(data)
    data = data['sentence']
    response = generator(data, max_length=1000, num_return_sequences=1)
    print(response)
    return {'string': response[0]['generated_text']}


print("Server Started!")
