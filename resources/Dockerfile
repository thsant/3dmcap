# 3-Demeter Capture image
FROM thsant/opencvpy2
MAINTAINER Thiago T. Santos "thiago.santos@embrapa.br"
ENV INSTALL_PREFIX /usr/local
#ENV http_proxy=http://proxy.cnptia.embrapa.br:3128
#ENV HTTP_PROXY=http://proxy.cnptia.embrapa.br:3128
#ENV https_proxy=https://proxy.cnptia.embrapa.br:3128
#ENV HTTPS_PROXY=https://proxy.cnptia.embrapa.br:3128

# Layer 1: Pangolin 
RUN apt-get update \
    && apt-get install -y libglew-dev \
    && cd /usr/local/src \
    && git clone https://github.com/stevenlovegrove/Pangolin.git \
    && cd Pangolin \
    && INSTALL_PREFIX=/usr/local \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. \
    && make -j4 \
    && make install \
    && cd ../../ \
    && rm -rf Pangolin

# Layer 2: ORB-SLAM2 for 3-Demeter Capture
RUN apt-get install -y python-pip \
    && pip install cython \
    && cd /usr/local/src \
    && git clone https://github.com/thsant/ORB_SLAM2.git \
    && cd ORB_SLAM2 \
    && ./build.sh \
    && cp lib/libORB_SLAM2.so /usr/local/lib \
    && cp Thirdparty/DBoW2/lib/libDBoW2.so /usr/local/lib \
    && cp Thirdparty/g2o/lib/libg2o.so /usr/local/lib \
    && cp python/slam.so /usr/local/lib/python2.7/dist-packages/ \
    && mkdir /usr/local/share/3dmcap \
    && cp Vocabulary/ORBvoc.txt /usr/local/share/3dmcap/ \
    && cd ../ \
    && rm -rf ORB_SLAM2

# Layer 3: PMVS
RUN cd /usr/local/src \
    && git clone https://github.com/pmoulon/CMVS-PMVS.git \
    && cd CMVS-PMVS/program \
    && mkdir build \
    && cd build/ \
    && INSTALL_PREFIX=/usr/local \
    && cmake -D CMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. \
    && make -j4 \
    && make install \
    && cd ../../../ \
    && rm -rf CMVS-PMVS/

# Layer 4: Adding other 3-Demeter dependencies
RUN apt-get install -y python-wxgtk3.0 python-vtk python-tk v4l-utils 

# Layer 5: Finally, 3-Demeter Capture code
RUN cd /usr/local/src \
    && git clone https://github.com/thsant/3dmcap.git \
    && cd 3dmcap \
    && pip install -r requirements.txt

# Layer 6: Installation and configuration
RUN cd /usr/local/src/3dmcap \
    && cp -r dmcap/ /usr/local/lib/python2.7/dist-packages \
    && cp -r ./resources/* /usr/local/share/3dmcap \
    && cp ./dmcap/camcal.py ./dmcap/capture.py /usr/local/bin \ 
    && adduser --system demeter \
    && adduser demeter video

USER demeter
WORKDIR /home/demeter

