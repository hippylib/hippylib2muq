#FROM ceciledc/fenics_mixed_dimensional:latest
FROM quay.io/fenicsproject/stable:latest
MAINTAINER Ki-Tae Kim

USER root

RUN apt-get -qq -y update && \
    pip3 install --upgrade pip && \
    pip3 install hippylib && \
    pip3 install h5py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER fenics

WORKDIR /home/fenics/

# Install MUQ
RUN mkdir -p ./lib/muq && \
    git clone --depth 1 https://bitbucket.org/mituq/muq2.git && \
    cd muq2/; mkdir build; cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/home/fenics/lib/muq -DMUQ_USE_MPI=OFF -DMUQ_USE_PYTHON=ON ../ && \
    make -j2 install && \
    cd /home/fenics/ && \
    rm -rf muq2

# Install hippylib-muq interface
RUN git clone https://github.com/hippylib/hippylib2muq.git

USER root

# Set environmental variables
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/fenics/lib/muq/lib" \
    PYTHONPATH="/home/fenics/lib/muq/lib"

CMD ["bash"]
