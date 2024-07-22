from tensorflow/tensorflow:latest-gpu
WORKDIR /app
COPY . ./digital-auto

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r ./digital-auto/requirements.txt
