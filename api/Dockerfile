FROM python:3.8
COPY api.py /home/api.py
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1 
ENTRYPOINT FLASK_APP=/home/api.py flask run
