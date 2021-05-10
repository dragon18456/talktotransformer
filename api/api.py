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
import subprocess

import os
import time

from TTS.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesis import synthesis

generator = pipeline('text-generation', model='gpt2')
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

print("Imported gpt and wav2vec2")

def tts(model, text, CONFIG, use_cuda, ap, use_gl, figures=True):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id, style_wav=None,
                                                                             truncated=False, enable_eos_bos_chars=CONFIG.enable_eos_bos_chars)
    # mel_postnet_spec = ap._denormalize(mel_postnet_spec.T)
    if not use_gl:
        waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
        waveform = waveform.flatten()
    if use_cuda:
        waveform = waveform.cpu()
    waveform = waveform.numpy()
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(waveform.shape)
    print(" > Run-time: {}".format(time.time() - t_1))
    print(" > Real-time factor: {}".format(rtf))
    print(" > Time per step: {}".format(tps)) 
    return alignment, mel_postnet_spec, stop_tokens, waveform

SAMPLE_RATE = 22050

use_cuda = False
TTS_MODEL = "tts_model.pth.tar"
TTS_CONFIG = "config.json"
VOCODER_MODEL = "vocoder_model.pth.tar"
VOCODER_CONFIG = "config_vocoder.json"
# load configs
TTS_CONFIG = load_config(TTS_CONFIG)
VOCODER_CONFIG = load_config(VOCODER_CONFIG)

# load the audio processor
ap = AudioProcessor(**TTS_CONFIG.audio)         

# LOAD TTS MODEL
# multi speaker 
speaker_id = None
speakers = []

# load the model
num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, len(speakers), TTS_CONFIG)

# load model state
cp =  torch.load(TTS_MODEL, map_location=torch.device('cpu'))

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()

# set model stepsize
if 'r' in cp:
    model.decoder.set_r(cp['r'])

model.decoder.max_decoder_steps = None

from TTS.vocoder.utils.generic_utils import setup_generator

# LOAD VOCODER MODEL
vocoder_model = setup_generator(VOCODER_CONFIG)
vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location="cpu")["model"])
vocoder_model.remove_weight_norm()
vocoder_model.inference_padding = 0


ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])    
if use_cuda:
    vocoder_model.cuda()
vocoder_model.eval()


print("Import everything!")

app = Flask(__name__)


# NOTE: This route is needed for the default EB health check route
@app.route('/')
def home():
    return "ok"


@app.route('/api/image/', methods=['GET'])
def get_img():
    filename = request.args.get('filename')
    return send_file("images/" + filename + ".png", mimetype='image/png')


@app.route('/api/tts', methods=['GET'])
def get_tts():
    sentence = request.args.get('sentence')

    filename = './audio/tts_output.wav'

    # preprocessing
    align, spec, stop_tokens, wav = tts(model, sentence, TTS_CONFIG, use_cuda, ap, use_gl=False, figures=True)

    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    write(filename, SAMPLE_RATE, wav_norm.astype(np.int16))

    return send_file(filename, mimetype="audio/mpeg")


@app.route('/api/transformer', methods=['POST'])
def get_current_response():
    data = request.get_json()
    print(data)
    sentence = data['sentence']
    max_length = data['max_length']
    if not max_length:
        max_length = 100
    if max_length.isnumeric():
        max_length = int(max_length)
    if max_length > 1000:
        max_length = 1000
    response = generator(sentence, max_length=max_length, num_return_sequences=1)
    print(response)
    return {'string': response[0]['generated_text']}

@app.route('/api/asr', methods=['POST'])
def get_asr():
    content = request.get_json()
    ans = base64.b64decode(bytes(content["message"], 'utf-8'))

    with open("temp.wav", "wb") as fh:
        fh.write(ans)

    audio_input, rate = librosa.load("temp.wav", sr=16000)
    # transcribe
    input_values = processor(audio_input, return_tensors="pt").input_values
    logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].capitalize()

    print("The transcription was:", transcription)

    return {'string': transcription}


print("Server Started!")

if __name__ == '__main__':
    app.run(debug=True)
