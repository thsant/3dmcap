# 3dmcap

![3-Demeter Logo](https://github.com/thsant/3dmcap/blob/master/resources/3dmcap_banner.jpg "3dmcapr Logo")

**Author:** [Thiago T. Santos](https://www.embrapa.br/en/web/portal/team/-/empregado/351534/thiago-teixeira-santos), [Embrapa Agricultural Informatics](https://www.embrapa.br/en/informatica-agropecuaria)

**3dmcap** is an application for three-dimensional
  reconstruction of objects from digital images. It allows an ordinary
  camera and a computer to operate as **a simple 3-D scanner**. The
  application assists the user on imaging and computing a cloud of
  3-D points, sampling the objects surfaces in three dimensions. Its
  purpose within Embrapa is the 3-D reconstruction of plants for
  purposes of automatic measurement in phenotyping and precision
  agriculture.  

## Related Publications

Santos, T. T., Bassoi, L. H., Oldoni, H., & Martins,
R. L. (2017). **Automatic grape bunch detection in vineyards based on affordable 3D phenotyping using a consumer webcam**. In
J. G. A. Barbedo, M. F. Moura, L. A. S. Romani, T. T. Santos, &
D. P. Drucker (Eds.), Anais do XI Congresso Brasileiro de
Agroinformática (SBIAgro 2017) (pp. 89–98). Campinas:
Unicamp. [PDF](http://ainfo.cnptia.embrapa.br/digital/bitstream/item/169609/1/Automatic-grape-SBIAgro.pdf)


## License

3dmcap is released under a [GPLv3 license](https://github.com/thsant/3dmcap/blob/master/LICENSE). This software relies on
other open-source components as [PMVS](https://www.di.ens.fr/pmvs/), [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2), [g2o](https://github.com/RainerKuemmerle/g2o) and  [OpenCV](https://opencv.org/).

If you need 3dmcap for commercial purposes, please contact Embrapa Agricultural Informatics
[Technology Transfer Office](https://www.embrapa.br/en/informatica-agropecuaria/transferencia-de-tecnologia).

If you use 3dmcap in an academic work, please cite:

```
@inproceedings{Santos2017,
	address = {Campinas},
	author = {Santos, Thiago Teixeira and Bassoi, Luis Henrique and Oldoni, Henrique and Martins, Roberto Luvisutto},
	title = {{Automatic grape bunch detection in vineyards based on affordable 3D phenotyping using a consumer webcam}},
	booktitle = {Anais do XI Congresso Brasileiro de Agroinform{\'{a}}tica (SBIAgro 2017)},
	editor = {Barbedo, Jayme Garcia Arnal and Moura, Maria Fernanda and Romani, Luciana Alvim Santos and Santos, Thiago Teixeira and Drucker, D{\'{e}}bora Pignatari},
	number = {October},
	pages = {89--98},
	publisher = {Unicamp},	
	year = {2017}
}
```

# Running using Docker

The easiest way to install and run 3dmcap is using our **Docker image**, [thsant/3dmcap](https://cloud.docker.com/swarm/thsant/repository/docker/thsant/3dmcap/general). If you
have Docker running in a Ubuntu Linux host and a camera connected as `/dev/video0`,  the following command should install and run 3dmcap for you:

```
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 -v $HOME/.3dmcap:/home/demeter/.3dmcap -v $HOME/workdir:/home/demeter/workdir thsant/3dmcap capture.py
```

Note the command above assumes you have, in your home directory, a `.3dmcap` directory, containing the configuration file, and a `workdir` directory for saving the results produced by
the application running in the Docker container.

If you have problems loading the graphical interface, try to execute

```
xhost +local:root
```

before the docker run. After the execution, run

```
xhost -local:root
```

to return the access controls.

## Errors regarding NVIDIA drivers and OpenGL

If you are facing crashes presenting the following error message:

```
libGL error: No matching fbConfigs or visuals found
libGL error: failed to load driver: swrast
```

then you are having issues regarding NVIDIA drivers and OpenGL from Docker. In this case, consider
[NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) as a workaround.

# Dependencies

The software depends on:

* Eigen 3;
* OpenCV with Python 2.7 support;
* Our [modified version of ORB_SLAM2](https://github.com/thsant/ORB_SLAM2). It includes [DBoW2](https://github.com/dorian3d/DBoW2) and [g2o](https://github.com/RainerKuemmerle/g2o)
as the original ORB_SLAM2.

The Python components depends on:

* numpy
* scipy
* Pillow
* wxPython
* ObjectListView 
* traits
* traitsui
* mayavi
* PyYAML

3dmcap have beend tested in **Ubuntu 16.04** and **16.10**.

## Building 3dmcap

For users that prefer build the entire system in their own hosts, this section describes the detailed building process. We
assume `/usr/local` as the install directory.

```
INSTALL_PREFIX=/usr/local
```

Add Eigen (needed by OpenCV and g2o), build-essential and cmake:

```
apt-get install -y build-essential libeigen3-dev cmake
```

Add OpenCV 3.4 dependencies:

```
apt-get install -y git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev 
apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```

Get the [OpenCV 3.4.1 sources](https://opencv.org/releases.html) and extract the files to `/usr/local/src`. Then build OpenCV:

```
cd /usr/local/src/opencv-3.4.1 
mkdir build 
cd build 
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. 
make -j4 
make install 
```

Check if the Python module is OK:

```
python -c 'import cv2; print cv2.__version__'
```

Install [Pangolin](https://github.com/stevenlovegrove/Pangolin.git):

```
apt-get install -y libglew-dev 
cd /usr/local/src 
git clone https://github.com/stevenlovegrove/Pangolin.git 
cd Pangolin 
mkdir build 
cd build 
cmake -D CMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. 
make -j4 
make install 
```

Install our modified ORB-SLAM2 version:

```
apt-get install -y python-pip 
pip install cython 
cd /usr/local/src 
git clone https://github.com/thsant/ORB_SLAM2.git 
cd ORB_SLAM2 
./build.sh 
cp lib/libORB_SLAM2.so /usr/local/lib 
cp Thirdparty/DBoW2/lib/libDBoW2.so /usr/local/lib 
cp Thirdparty/g2o/lib/libg2o.so /usr/local/lib 
cp python/slam.so /usr/local/lib/python2.7/dist-packages/ 
mkdir /usr/local/share/3dmcap 
cp Vocabulary/ORBvoc.txt /usr/local/share/3dmcap/ 
```

Install PMVS. We recommend [pmoulon's version at GitHub](https://github.com/pmoulon/CMVS-PMVS.git):

```
cd /usr/local/src 
git clone https://github.com/pmoulon/CMVS-PMVS.git 
cd CMVS-PMVS/program 
mkdir build 
cd build/ 
cmake -D CMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. 
make -j4 
make install 
```

Add other 3-Demeter dependencies:

```
apt-get install -y python-wxgtk3.0 python-vtk python-tk v4l-utils 
```

Finally, get 3dmcap code:

```
cd /usr/local/src 
git clone https://github.com/thsant/3dmcap.git 
cd 3dmcap 
pip install -r requirements.txt
```

Configure the environment:

```
cd /usr/local/src/3dmcap 
cp -r dmcap/ /usr/local/lib/python2.7/dist-packages 
cp -r ./resources/* /usr/local/share/3dmcap 
cp ./dmcap/camcal.py ./dmcap/capture.py /usr/local/bin  
```

Edit the **3dmcap.cfg** file and save it to your `$HOME/.3dmcap` directory. You can run **capture.py** to start 3dmcap.  

## Manual

For a tutorial about how to use the application, including details about configuration, see [the PDF manual](https://github.com/thsant/3dmcap/blob/master/resources/3dmcap.pdf).