import tempfile
import os
import wx

from distutils import dir_util
from threading import Thread
from wx.lib.pubsub import pub

import pmvs
import scale as pcs

from dmcap.gui import PCapMainFrame
from dmcap.model import DMCModel, DMCVideoDeviceError


class DMCDenseStereoBuilder(Thread):

    def __init__(self, control):
        Thread.__init__(self)
        self.control = control
        self.ply_path = None

    def run(self):
        self.ply_path = pmvs.run_pmvs(self.control.working_dir)
        self.control.finish_mvs(self.ply_path)


class DMCController:

    def __init__(self, conf):
        self.conf = conf
        resources_path = self.conf.get('general', 'resources_path')

        config_fpath = self.conf.get('orbslam', 'config_fpath')
        voc_db_fpath = resources_path + '/ORBvoc.txt'

        frame_width = self.conf.getint('camera', 'width')
        frame_height = self.conf.getint('camera', 'height')
        self.model = DMCModel(config_fpath,
                              voc_db_fpath,
                              frame_width,
                              frame_height)

        self.view = PCapMainFrame(self, resources_path)

        self.mvs_builder = DMCDenseStereoBuilder(self)
        self.working_dir = None

        pub.subscribe(self.update_frame, "new frame")
        pub.subscribe(self.update_odometry, "new odometry")
        pub.subscribe(self.update_keyframe, "new keyframe")

        self.view.Show()

    def update_frame(self, data):
        """Update the bitmap to the frame from new frame event."""
        frame = data['frame'][:, :, ::-1]
        fnum = data['frame_num']
        wx.CallAfter(self.view.updateFrame, frame, fnum)

    def update_odometry(self, data):
        """Update the camera position in the viewer."""
        trans_cwi = data['Pi']

        if trans_cwi is None:
            pass
        else:
            wx.CallAfter(self.view.updateOdometry, trans_cwi)

    def update_keyframe(self, data):
        """Update keyframe"""
        fnum = data['frame_num']
        fpath = data['frame_path']
        map_points = data['map']
        wx.CallAfter(self.view.appendKeyFrame, fnum, fpath, map_points)

    def set_cap_device(self, device):
        try:
            self.model.set_cap_device(device)
        except DMCVideoDeviceError:
            self.view.OnError('Impossible opening camera %s.' % device)

    def new_acquisition(self):
        self.working_dir = tempfile.mkdtemp(prefix='3dmc-')
        self.model.start_capture(self.working_dir)

    def resume_acquisition(self):
        self.model.resume_capture()

    def pause_acquisition(self):
        self.model.pause_capture()

    def finish_acquisition(self):
        self.model.stop_capture_save_odometry()
        self.view.loadOdometry(self.working_dir + '/3dmc_odometry.txt')

    def export_to_mvs(self):
        vo_file = self.working_dir + '/3dmc_odometry.txt'
        if os.access(vo_file, os.F_OK):
            pmvs.export2pmvs(self.model.K, self.working_dir)

    def start_mvs(self):
        self.mvs_builder.start()

    def save_capture(self, dir_full_path):
        dir_util.copy_tree(self.working_dir, dir_full_path)
        self.working_dir = dir_full_path
        self.view.SetStatus('Working directory is now "%s"' % self.working_dir)

    def open_cap_dir(self, dir_full_path):
        self.working_dir = dir_full_path
        self.view.SetStatus('Working directory is now "%s"' % self.working_dir)

    def finish_mvs(self, ply_path):
        if ply_path is not None:
            self.view.loadPLY(ply_path)
            self.view.MVSFinished()

    def normalize(self):
        # List keyframes
        ref_dist_mm = self.conf.getfloat('general',
                                         'ref_distance_mm')
        success, msg = pcs.normalize(self.model.K,
                                     self.working_dir,
                                     ref_dist_mm)
        if success:
            ply_norm_path = msg
            self.view.loadPLY(ply_norm_path, clear_previous=True)
        else:
            self.view.notify(msg)
