FROM python:3.12

# Creare maintained
LABEL org.opencontainers.image.authors="podpac@creare.com"

# Install general tools
RUN apt-get update --yes --quiet && apt-get install --yes --quiet --no-install-recommends \
    build-essential \ 
    curl \ 
    unzip \
    tar \
    wget && \
    apt-get clean
COPY . /opt/podpac
WORKDIR /opt/podpac
# Install PODPAC with all dependencies
RUN pip install .[all]
