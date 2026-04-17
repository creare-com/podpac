FROM python:3.8 
# FROM python:3.12

USER root
RUN pip3 install pytest pytest-cov

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
