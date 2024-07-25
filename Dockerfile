FROM python:3.8

ENV TENSORFLOW_VERSION 2.13.1
ENV PROTOC_VERSION 3.15.6

COPY . /root

WORKDIR /root

RUN pip install --upgrade pip

RUN apt-get update && apt-get -y install git vim curl lsb-release unzip gnupg gcc python3-dev

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install tensorflow==${TENSORFLOW_VERSION} pandas==2.0.3 image kaggle==1.6.14

# Install gcloud and gsutil commands
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

# Install protoc
RUN curl -OL "https://github.com/google/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip" && \
    unzip protoc-${PROTOC_VERSION}-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-${PROTOC_VERSION}-linux-x86_64.zip

# Install Object Detection API with TensorFlow 2
RUN cd ./Tensorflow/models/research && \
  protoc object_detection/protos/*.proto --python_out=.  && \
  cp object_detection/packages/tf2/setup.py .  && \
  python -m pip install .

VOLUME /data

# CMD ["python", "script_main.py"]