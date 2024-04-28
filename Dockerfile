FROM ubuntu:20.04

WORKDIR /workspace

COPY . /workspace/

RUN cd /workspace

RUN pip install --no-cache-dir -r requirements.txt