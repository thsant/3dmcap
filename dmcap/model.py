import cv2
import numpy as np
import os
import time

from threading import Thread
from wx.lib.pubsub import pub

import slam
from dmcap import util


class DMCModel:

    def __init__(self, slam_cfg_fpath, voc_db_fpath,
                 frame_width, frame_height):
        self.grabber = DMCFrameGrabber(frame_width, frame_height)
        self.slam_cfg_fpath = slam_cfg_fpath
        self.voc_db_fpath = voc_db_fpath
        self.vo = None
        self.is_capturing = False
        self.working_dir = None
        self.cap_device = None

        params = util.load_params(self.slam_cfg_fpath)
        cx = params['Camera.cx']
        cy = params['Camera.cy']
        fx = params['Camera.fx']
        fy = params['Camera.fy']
        k1 = params['Camera.k1']
        k2 = params['Camera.k2']
        k3 = params['Camera.k3']
        p1 = params['Camera.p1']
        p2 = params['Camera.p2']
        self.K = np.array([[fx, 0., cx],
                           [0., fy, cy],
                           [0., 0., 1.]])
        self.dist_coef = np.array([k1, k2, p1, p2, k3])

        self.T = {}
        self._kf_count = 0
        pub.subscribe(self.add_frame, "new frame")

    def start_capture(self, path):
        self.working_dir = path
        self.vo = slam.VOSystem(self.voc_db_fpath,
                                self.slam_cfg_fpath,
                                0,
                                False)
        self.vo.init_numpy_integration()
        self.grabber.start()
        self.is_capturing = True

    def resume_capture(self):
        self.is_capturing = True
        self.grabber.resume()

    def pause_capture(self):
        self.is_capturing = False
        self.grabber.pause()

    def stop_capture_save_odometry(self):

        # Stop the frame grabber
        self.grabber.finish = True
        while self.grabber.is_alive():
            time.sleep(0.05)

        # Shutdown ORB-SLAM
        self.vo.Shutdown()

        # Save the calibration matrix K
        cal_mat_fname = self.working_dir + '/3dmc_calibration_matrix.txt'
        np.savetxt(cal_mat_fname, self.K, header='Calibration matrix K')

        # Save odometry
        vo_file = self.working_dir + '/3dmc_odometry.txt'
        self.vo.SaveKeyFrameTrajectoryTUM(vo_file)

    def set_cap_device(self, device):
        self.cap_device = device
        self.grabber.set_video_device(self.cap_device)

    def add_frame(self, data):
        if self.is_capturing:
            frame, i = data['frame'], data['frame_num']
            self.T[i] = self.vo.TrackMonocular(frame, i)
            if self.T[i] is not None:
                pub.sendMessage("new odometry",
                                data={'frame_num': i,
                                      'Pi': self.T[i]})
                if self.vo.isCurFrameAKeyFrame():
                    self._kf_count += 1
                    frame_undistort = cv2.undistort(frame, self.K,
                                                    self.dist_coef)
                    frame_file_path = self.working_dir + os.sep
                    frame_file_path += 'frame-%.5d.jpg' % i
                    # Update map just after 4 keyframes
                    if self._kf_count % 4 == 0:
                        X = self.vo.GetGoodMapPoints()
                    else:
                        X = None
                    pub.sendMessage("new keyframe",
                                    data={'frame_num': i,
                                          'frame_path': frame_file_path,
                                          'map': X})
                    frame_writer = DMCFrameWriter(frame_file_path,
                                                  frame_undistort)
                    frame_writer.start()


class DMCVideoDeviceError(Exception):

    def __init__(self, device):
        print 'Impossible get frames from device %s.' % device


class DMCFrameWriter(Thread):

    def __init__(self, file_path, frame):
        Thread.__init__(self)
        self.path = file_path
        self.frame = frame

    def run(self):
        cv2.imwrite(self.path, self.frame)


class DMCFrameGrabber(Thread):

    _RUNNING = 0
    _PAUSED = 1

    def __init__(self, frame_width, frame_height):
        Thread.__init__(self)

        self.vid_device = None
        self.vid_device_number = -1
        self.capture = None
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.cur_frame = None
        self.finish = False
        self.state = DMCFrameGrabber._RUNNING

    def set_video_device(self, device):
        self.vid_device = device
        self.capture = cv2.VideoCapture(self.vid_device)

        if self.capture is None or not self.capture.isOpened():
                raise DMCVideoDeviceError(device)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Turn auto-focus OFF
        # PROBLEM: It raises a VIDEOIO ERROR: V4L: Property
        #                <unknown property string>(39) not supported by device
        # self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # Set focus to infinity
        # self.capture.set(cv2.CAP_PROP_FOCUS, 0)

        # other option? v4l2-ctl
        # v4l2-ctl -d /dev/video1 -c focus_auto=0
        # v4l2-ctl -d /dev/video1 -c focus_absolute=0

    def pause(self):
        self.state = DMCFrameGrabber._PAUSED

    def resume(self):
        self.state = DMCFrameGrabber._RUNNING

    def run(self):

        if self.capture is None or not self.capture.isOpened():
            # Try open the device again:
            self.set_video_device(self.vid_device)

        frame_count = 0
        success, frame = self.capture.read()
        if success:
            self.cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pub.sendMessage("new frame",
                            data={'frame_num': frame_count,
                                  'frame': self.cur_frame.copy()})

            print frame.shape

        while not self.finish:
            if self.state == DMCFrameGrabber._RUNNING:
                success, frame = self.capture.read()

                if success:
                    frame_count += 1
                    self.cur_frame = frame
                    pub.sendMessage("new frame",
                                    data={'frame_num': frame_count,
                                          'frame': self.cur_frame.copy()})

            time.sleep(0.1)

        self.capture.release()
