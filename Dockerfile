# Use the continuumio/anaconda3 image as a base
# https://hub.docker.com/r/continuumio/anaconda3
FROM continuumio/anaconda3:2024.10-1

# Creare maintained
LABEL org.opencontainers.image.authors="podpac@creare.com"

# Install general tools
RUN apt-get update --yes --quiet && apt-get install --yes --quiet --no-install-recommends \
    build-essential \ 
    curl \ 
    unzip \
    tar \
    wget && \
    apt-get clean && \
    # Create a podpac anaconda environment and activate
    conda init bash && . ~/.bashrc \
    && conda create --yes --name podpac python=3 anaconda \
    && conda activate podpac && \
    # Install PODPAC with all dependencies
    pip install podpac[all]
