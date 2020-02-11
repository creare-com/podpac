# Use the continuumio/anaconda3 image as a base
# https://hub.docker.com/r/continuumio/anaconda3
FROM continuumio/anaconda3:latest

# Creare maintained
MAINTAINER Creare podpac@creare.com

# Install general tools
RUN apt-get update --yes --quiet && apt-get install --yes --quiet --no-install-recommends \
    build-essential \ 
    curl \ 
    unzip \
    tar \
    wget

# Create a podpac anaconda environment and activate
RUN conda init bash && . ~/.bashrc \
    && conda create --yes --name podpac python=3 anaconda \
    && conda activate podpac

# Install PODPAC with all dependencies
RUN pip install podpac[all]
