
FROM ubuntu:16.04
LABEL Movidius-ncsdkv2 francesco.argentieri89@gmail.com

ARG http_proxy

ARG https_proxy

ENV http_proxy ${http_proxy}

ENV https_proxy ${https_proxy}

RUN echo $https_proxy

RUN echo $http_proxy

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
      build-essential \
      git \
      lsb-release \
      sudo \
      udev \
      usbutils \
      wget \
      cmake \
    && apt-get clean all

RUN useradd -c "Movidius User" -m movidius

COPY installer /etc/sudoers.d/

RUN mkdir -p /etc/udev/rules.d/

USER movidius

WORKDIR /home/movidius

RUN git clone -b ncsdk2 https://github.com/movidius/ncsdk.git

WORKDIR /home/movidius/ncsdk

RUN make install

ENV SCRIPT_DIR /script
WORKDIR ${SCRIPT_DIR}
COPY requirements.txt ${SCRIPT_DIR}
RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]

