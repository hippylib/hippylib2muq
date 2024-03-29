FROM quay.io/fenicsproject/stable:2019.1.0.r3
MAINTAINER Ki-Tae Kim

USER root

# Install MUQ
RUN mkdir -p ./lib/muq && \
    git clone -b v0.4.0 https://bitbucket.org/mituq/muq2.git && \
    cd muq2/; mkdir build; cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/home/fenics/lib/muq -DMUQ_USE_PYTHON=ON -DNANOFLANN_EXTERNAL_SOURCE=https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.4.3.tar.gz ../ && \
    make -j4 install && \
    cd /home/fenics/ && \
    rm -rf muq2

RUN apt-get -qq -y update && \
    pip3 install --upgrade pip && \
    pip3 install jupyter && \
    pip3 install matplotlib && \
    pip3 install h5py && \
    pip3 install pyyaml && \
    pip3 install seaborn==0.10.0 && \
    pip3 install statsmodels && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER fenics

WORKDIR /home/fenics/

# Install the latest hippylib
RUN git clone https://github.com/hippylib/hippylib.git

# Install hippylib-muq interface
RUN git clone https://github.com/hippylib/hippylib2muq.git


# Set environmental variables
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/fenics/lib/muq/lib" \
    PYTHONPATH="/home/fenics/lib/muq/python:/home/fenics/hippylib:/home/fenics/hippylib2muq"

USER root

CMD ["bash"]
