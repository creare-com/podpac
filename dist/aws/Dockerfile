FROM amazonlinux:latest

ARG REF="master"

# development tools
RUN yum update -y && yum -y install \ 
    groupinstall \
    development \
    gcc \
    gcc-c++ \
    git \
    zip \
    freetype-devel \
    yum-utils \
    findutils \
    openssl-devel \
    && yum clean all

# Mock current AWS Lambda docker image
# NOTE: this is still Py3.7, need to be careful about version management
RUN yum -y install \
    python3 \
    python3-pip \
    python3-devel \
    && yum clean all

# clone the podpac repository and checkout the requested tag
# for developers looking to create a custom deployment package or dependencies,
# comment this block and un-comment the next block
RUN git clone https://github.com/creare-com/podpac.git /podpac/ &&\
    pushd /podpac/ && \
    git fetch --all && \
    git checkout $REF && \
    popd

# # uncomment this block to create a custom deployment package or dependencies archive
# # based on your local copy of the PODPAC repository
# # this command assumes you are building the Dockerfile using `build_lambda.sh` (which runs from the root of the PODPAC repository )
# ADD . /podpac/

# Install core, datatype and aws optional dependencies
RUN mkdir /tmp/vendored/ && \
    cd /podpac/ && rm -rf .git/ doc/ .github/ && \
    pip3 install . -t /tmp/vendored/ --upgrade && \
    pip3 install .[datatype] -t /tmp/vendored/ --upgrade && \
    pip3 install .[aws] -t /tmp/vendored/ --upgrade

# need to add some __init__ files
RUN cd /tmp/vendored/ && touch pydap/__init__.py && \
    touch pydap/responses/__init__.py && \
    touch pydap/handlers/__init__.py && \
    touch pydap/parsers/__init__.py

# copy handler and _mk_dist:
RUN cp /podpac/dist/aws/handler.py /tmp/vendored/handler.py && \
    cp /podpac/dist/aws/_mk_dist.py /tmp/vendored/_mk_dist.py

RUN cd /tmp/vendored && \
    find * -maxdepth 0 -type f | grep ".zip" -v | grep -v ".pyc" | xargs zip -9 -rqy podpac_dist.zip
RUN cd /tmp/vendored && \
    find * -maxdepth 0 -type d -exec zip -9 -rqy {}.zip {} \;
RUN cd /tmp/vendored && du -s *.zip > zip_package_sizes.txt
RUN cd /tmp/vendored && du -s * | grep .zip -v > package_sizes.txt
RUN cd /tmp/vendored && python3 _mk_dist.py
