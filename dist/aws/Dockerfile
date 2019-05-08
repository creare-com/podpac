FROM amazonlinux:latest

ARG COMMIT_SHA=""
ARG TAG=""
RUN echo $COMMIT_SHA

RUN yum update -y

# Install apt dependencies
RUN yum install -y gcc gcc-c++ freetype-devel yum-utils findutils openssl-devel

RUN yum -y groupinstall development

# Mock current AWS Lambda docker image
# Find complete list of package https://gist.github.com/vincentsarago/acb33eb9f0502fcd38e0feadfe098eb7
RUN  yum install -y libjpeg-devel libpng-devel libcurl-devel ImageMagick-devel.x86_64 python3-devel.x86_64 which

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

ADD . /podpac/

RUN mkdir /tmp/vendored/ && \
    cp /podpac/settings.json /tmp/vendored/settings.json && \
    cd /podpac/ && git clean -xdf && \
    pip3 install -r dist/aws/aws_requirements.txt -t /tmp/vendored/ --upgrade

RUN cd /tmp/vendored/ && touch pydap/__init__.py && \
    touch pydap/responses/__init__.py && \
    touch pydap/handlers/__init__.py && \
    touch pydap/parsers/__init__.py

RUN cp -r /podpac/ /tmp/vendored/ && \
    mv /tmp/vendored/podpac/dist/aws/handler.py /tmp/vendored/handler.py && \
    cp -r /tmp/vendored/podpac/podpac/* /tmp/vendored/podpac/ && \
    rm -rf /tmp/vendored/podpac/podpac/*

RUN pip3 install pyproj==2.1.3 -t /tmp/vendored/ --upgrade

RUN cd /tmp/vendored && \
    find * -maxdepth 0 -type f | grep ".zip" -v | grep -v ".pyc" | xargs zip -9 -rqy podpac_dist_latest.zip
RUN cd /tmp/vendored && \
    find * -maxdepth 0 -type d -exec zip -9 -rqy {}.zip {} \;
RUN cd /tmp/vendored && du -s *.zip > zip_package_sizes.txt
RUN cd /tmp/vendored && du -s * | grep .zip -v > package_sizes.txt
RUN cd /tmp/vendored && cp podpac/dist/aws/mk_dist.py . && python3 mk_dist.py
