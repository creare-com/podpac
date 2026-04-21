FROM python:3.8 
# FROM python:3.12

USER root
RUN pip3 install pytest==8.3.5 pytest-cov==5.0.0 s3fs==2024.10.0 boto3==1.37.3

# Set up user to match the host OS (https://stackoverflow.com/a/78621662/415551)
# Ubuntu uses addgroup and adduser, RHEL uses groupadd and useradd
ARG HOST_USER
ARG HOST_UID
ARG HOST_GID
RUN <<EOF
    addgroup --gid ${HOST_GID} ${HOST_USER}
    adduser --uid ${HOST_UID} --gid ${HOST_GID} ${HOST_USER}
EOF

ENV HOME /home/${HOST_USER}
ENV TMPDIR=/tmp
WORKDIR /home/${HOST_USER}

USER ${HOST_USER}
