# talktotransformer

Final Project for Berkeley CS 194-080: Full Stack Deep Learning

This React.js + Flask website allows users to talk to a transformer and have it speak the response back to them. The backend uses serveral different machine learning architectures:
- Wav2Vec2.0 from huggingface: https://huggingface.co/transformers/model_doc/wav2vec2.html for the ASR
- GPT-2 from huggingface: https://huggingface.co/transformers/model_doc/gpt2.html for the Language Model
- Tacotron2 + Waveglow from Nvidia: https://github.com/NVIDIA/tacotron2 for the Text-to-Speech

Download pretrained models into api for the Text-to-Speech:
- Waveglow: https://drive.google.com/u/0/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF&export=download 
- Tacotron2: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view

