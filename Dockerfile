FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN pip install pytorch-lightning einops monai[all]
RUN pip install x-unet==0.3.0

WORKDIR /code