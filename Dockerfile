FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get -qy update

RUN apt-get install -qy --fix-missing sudo

RUN apt-get install -y --fix-missing \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev
RUN apt-get clean && rm -rf /tmp/* /var/tmp/*

ENV TRAINING_DIR=/training_model
WORKDIR /training_model

COPY . /training_model


RUN pip --no-cache-dir install -r /training_model/requirements.txt
RUN pip --no-cache-dir install tqdm

#ENTRYPOINT [ "train_cnn.sh" ]
ENTRYPOINT [ "/bin/bash", "train_cnn.sh" ]

