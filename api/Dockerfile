FROM python:3.8-buster
WORKDIR /api
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN gdown --id 1dntzjWFg7ufWaTaFy80nRz-Tu02xWZos -O tts_model.pth.tar
RUN gdown --id 18CQ6G6tBEOfvCHlPqP8EBI4xWbrr9dBc -O config.json
RUN gdown --id 1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K -O vocoder_model.pth.tar
RUN gdown --id 1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu -O config_vocoder.json
RUN gdown --id 11oY3Tv0kQtxK_JPgxrfesa99maVXHNxU -O scale_stats.npy


RUN apt-get update
RUN apt-get -y install ffmpeg 
RUN apt-get -y install espeak
RUN git clone https://github.com/coqui-ai/TTS

RUN cd TTS && git checkout b1935c97 && pip install -r requirements.txt && python setup.py install && cd ..


EXPOSE 5000
COPY . .
CMD ["gunicorn", "wsgi:app", "-t 500"]
