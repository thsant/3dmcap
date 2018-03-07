.. 3-Demeter Capture documentation master file, created by
   sphinx-quickstart on Wed Mar  7 14:18:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

3-Demeter Capture Tutorial
==========================

Before you start
----------------

Camera selection and calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Good cameras
* v4l2-ctl

::
   
 v4l2-ctl -d /dev/video1 -c focus_auto=0
 v4l2-ctl -d /dev/video1 -c focus_absolute=0

* camcal.py (imagem)

Printing the pattern for scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration file
~~~~~~~~~~~~~~~~~~~~~~

::
   
   [camera]
   width=1920
   height=1080
   
   [general]
   resources_path=/usr/local/share/3dmcap
   ref_distance_mm=51.5
   
   [orbslam]
   config_fpath=/usr/local/share/3dmcap/Logitech-C920.yaml

End of indented.

* Places for the configuration file.


The image acquisition step
--------------------------

* Start and select camera
* Start acquisition
* Left to right translation for stereo initialization
* Don't forget good frames for the pattern
* Pausing, restarting and relocalization
* Stoping and inspection
* Saving acquistion results

Multiple view stereo with PMVS
------------------------------

* Exporting files to PMVS
* Running PMVS inside 3-Demeter Capture
* Memory issues

Scaling
-------

* Seeing results

Exploring your point cloud using Meshlab
----------------------------------------

* Other tools to explore point clouds








  
