#!/usr/bin/env python
import cv2
import glob
import numpy as np
import os
import re
import subprocess
import time
import wx
import wx.lib.agw.pyprogress as pp

from mayavi.core.ui.api import SceneEditor, MlabSceneModel
from ObjectListView import ObjectListView, ColumnDefn
from scipy import linalg as la
from threading import Thread, RLock
from traits.api import HasTraits, Instance
from traitsui.api import View, Item

from dmcap import scale


class MayaviView(HasTraits):

    scene = Instance(MlabSceneModel, ())
    item = Item('scene', editor=SceneEditor(), resizable=True,
                show_label=False)
    view = View(item, resizable=True)

    def __init__(self):
        HasTraits.__init__(self)

        X = np.array([[0., 0., 0.]])
        s = np.array([0])
        v = np.array([[0.15, 0.15, 0.15]])
        dist = 0.25
        self.cam_path_plot = self.scene.mlab.plot3d(X[:, 0], X[:, 1], X[:, 2],
                                                    s,
                                                    tube_radius=None,
                                                    colormap='jet')

        self.cam_pos_plot = self.scene.mlab.quiver3d(X[-1:, 0],
                                                     X[-1:, 1],
                                                     X[-1:, 2],
                                                     v[-1:, 0],
                                                     v[-1:, 1],
                                                     v[-1:, 2],
                                                     opacity=0.5,
                                                     scale_factor=dist/2,
                                                     color=(1, 0, 0),
                                                     mode='2dthick_arrow')

        self.map_plot = self.scene.mlab.points3d(X[:, 0], X[:, 1], X[:, 2],
                                                 color=(0, 1, 0.25),
                                                 mode='point')
        self.scene.reset_zoom()


class MayaviUpdater(Thread):

    def __init__(self, gui_frame):
        Thread.__init__(self)
        self.gui = gui_frame
        self.updating = False

    def run(self):
        while self.updating:
            wx.CallAfter(self.gui.update3DViewer)
            time.sleep(0.02)


class PCapMainFrame(wx.Frame):

    FRAME_VIEW_WIDTH = 480
    FRAME_VIEW_HEIGHT = 270

    def __init__(self, controller, resources_path):
        wx.Frame.__init__(self, None)

        self.control = controller
        self.resources_path = resources_path
        self.cam_pos = None
        self.cam_ray = None
        self.cam_pos_lock = RLock()
        self.pointer = 0
        self.SetTitle("3-Demeter PlantCapture")
        self.status_bar = self.CreateStatusBar()
        self.is_mvs_on = False
        self.map_points = None
        self._update_counter = 0

        # Creating menus
        filemenu = wx.Menu()
        menuAbout = filemenu.Append(wx.ID_ABOUT, "&About",
                                    " Information about this program")
        menuExit = filemenu.Append(wx.ID_EXIT, "E&xit",
                                   " Terminate the program")

        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "&File")

        setmenu = wx.Menu()
        device_menu_item = setmenu.Append(0, "&Camera device", " Choose an \
        available camera for capturing")
        scale_menu_item = setmenu.Append(1, "&Scale correction", " Look for a\
        scale pattern")

        menuBar.Append(setmenu, "S&ettings")

        self.SetMenuBar(menuBar)

        # Tools
        actions = {
            'cap_new': (
                'New capture',
                'image_add.svg',
                'Start new image acquisition',
                self.OnNewAcq
            ),
            'cap_continue': (
                'Continue',
                'image_run.svg',
                'Resume acquisition',
                self.OnResumeAcq
            ),
            'cap_pause': (
                'Pause',
                'image_pause.svg',
                'Pause acquisition',
                self.OnPauseAcq
            ),
            'cap_finish': (
                'Finish capture',
                'image_stop.svg',
                'Finish acquisition',
                self.OnFinishAcq
            ),
            'dir_save_to': (
                'Save as...',
                'folder_downloads.svg',
                'Save capture files to a new directory',
                self.OnSave2Dir
            ),
            'dir_open': (
                'Open...',
                'folder_image.svg',
                'Open a directory with previously captured frames',
                self.OnOpenDir
            ),
            'mvs_export': (
                'Export to MVS',
                'download.svg',
                'Export files to the MVS subsystem',
                self.OnExportMVS
            ),
            'mvs_run': (
                'Multiple View Stereo',
                'tree.svg',
                'Start 3-D reconstruction',
                self.OnMVS
            ),
            'norm': (
                'Normalize',
                'size_height_accept.svg',
                'Normalize scale and orientation',
                self.OnNormalize
            ),
            'exit': (
                'Quit',
                'logout.svg',
                'Exit application',
                self.OnExit
            )
        }

        # Creating the toolbar
        self._toolbar = self.CreateToolBar()
        self._tools = {}

        tools_list = ['cap_new', 'cap_continue', 'cap_pause',
                      'cap_finish', '|',
                      'dir_save_to', 'dir_open', 'mvs_export',
                      'mvs_run', 'norm', '|',
                      'exit']
        for t in tools_list:
            if t is '|':
                self._toolbar.AddSeparator()
            else:
                label, icon, help, command = actions[t]
                bitmap_path = self.resources_path + '/picol/' + icon
                bitmap = wx.Bitmap(bitmap_path)
                self._tools[t] = self._toolbar.AddLabelTool(wx.ID_ANY,
                                                            label,
                                                            bitmap,
                                                            shortHelp=help)
                self.Bind(wx.EVT_TOOL, command, self._tools[t])

        self._toolbar.Realize()

        # Creating the main window
        self.main_window = wx.SplitterWindow(self, wx.ID_ANY)
        self.left_pane = wx.Panel(self.main_window, wx.ID_ANY)
        self.right_pane = wx.Panel(self.main_window, wx.ID_ANY)
        self.main_window.SplitVertically(self.left_pane,
                                         self.right_pane,
                                         sashPosition=256)

        self.main_vsizer = wx.BoxSizer(wx.VERTICAL)
        self.main_vsizer.Add(self.main_window, 1, wx.EXPAND)

        # Adding frame list
        self.keyframes = []
        self.frame_list = ObjectListView(self, wx.ID_ANY, style=wx.LC_REPORT)
        self.frame_list.SetColumns([ColumnDefn("#", "left", 30, "num"),
                                    ColumnDefn("Frame", "left", 100, "frame"),
                                    ColumnDefn("Path", "left", 130, "path")])

        self.updateKeyFrames()

        lsizer = wx.BoxSizer(wx.VERTICAL)
        lsizer.Add(self.frame_list, 1, wx.EXPAND)
        self.left_pane.SetSizer(lsizer)
        lsizer.Fit(self.left_pane)

        rsizer = wx.BoxSizer(wx.VERTICAL)
        # Adding frame display
        self.frame_image = None
        self.bitmap = wx.EmptyBitmap(PCapMainFrame.FRAME_VIEW_WIDTH,
                                     PCapMainFrame.FRAME_VIEW_HEIGHT)
        pnl_f = wx.Panel(self.right_pane)
        sb = wx.StaticBox(pnl_f, label='Camera frame')
        sbs = wx.StaticBoxSizer(sb, orient=wx.HORIZONTAL)
        sbs.AddSpacer(11)

        self.frame_view = wx.StaticBitmap(pnl_f, bitmap=self.bitmap)
        sbs.Add(self.frame_view, 0,
                wx.ALIGN_CENTER_HORIZONTAL | wx.ADJUST_MINSIZE, 10)
        sbs.AddSpacer(11)
        pnl_f.SetSizer(sbs)

        # Adding Mayavi view
        pnl_m = wx.Panel(self.right_pane)
        sb = wx.StaticBox(pnl_m, label='3-D Viewer')
        sbs = wx.StaticBoxSizer(sb, orient=wx.HORIZONTAL)
        sbs.AddSpacer(11)
        self.mayavi_view = MayaviView()
        traits_ui = self.mayavi_view.edit_traits(parent=pnl_m,
                                                 kind='subpanel')
        self.mayavi_control = traits_ui.control
        sbs.Add(self.mayavi_control, 1, wx.CENTER | wx.EXPAND)
        sbs.AddSpacer(11)
        pnl_m.SetSizer(sbs)

        rsizer.Add(pnl_f, 0, wx.LEFT)
        rsizer.Add(pnl_m, 1, wx.CENTER | wx.EXPAND)
        self.right_pane.SetSizer(rsizer)
        rsizer.Fit(self.right_pane)

        # Set events:
        self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)
        self.Bind(wx.EVT_MENU, self.OnSetDevice, device_menu_item)
        self.Bind(wx.EVT_MENU, self.OnSetScaleCor, scale_menu_item)
        self.frame_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnFrameSelected)

        self.SetSizer(self.main_vsizer)
        self.SetAutoLayout(1)
        self.main_vsizer.Fit(self)
        self.SetSize((900, 800))

        logo_path = self.resources_path + '/splash-logo.png'
        logo = cv2.imread(logo_path)[:, :, (2, 1, 0)]
        self.updateFrame(logo, 0)

        self.viewer_updater = MayaviUpdater(self)

    def updateFrame(self, frame, frame_num):
        """Update the bitmap to the frame from new frame event."""
        w, h = PCapMainFrame.FRAME_VIEW_WIDTH, PCapMainFrame.FRAME_VIEW_HEIGHT
        thumbnail = cv2.resize(frame, (w, h))
        self.frame_image = wx.ImageFromData(w, h, thumbnail.tostring())
        self.bitmap = wx.BitmapFromImage(self.frame_image)
        self.frame_view.SetBitmap(self.bitmap)
        if frame_num > 0:
            h, w, _ = frame.shape
            msg = 'Frame %d (%d x %d)' % (frame_num, w, h)
            self.status_bar.SetStatusText(msg)

    def updateOdometry(self, Tcw):
        """Add the new odometry data and refresh the 3-D viewer."""
        with self.cam_pos_lock:
            cam_center = Tcw[0:3, 3]
            cam_center.shape = 1, 3
            cam_p_ray = np.dot(Tcw[0:3, 0:3], np.array([0, 0, 1.]))
            if self.cam_pos is None:
                self.cam_pos = cam_center.copy()
                self.cam_ray = cam_p_ray.copy()
            else:
                self.cam_pos = np.vstack((self.cam_pos, cam_center))
                self.cam_ray = np.vstack((self.cam_ray, cam_p_ray))

    def appendKeyFrame(self, frame_num, frame_path, map_points=None):
        n = len(self.keyframes) + 1
        self.keyframes.append({'num': n,
                               'frame': 'Frame %.5d' % frame_num,
                               'path': frame_path})
        self.updateKeyFrames()

        if map_points is not None:
            self.map_points = map_points

    def updateKeyFrames(self):
        self.frame_list.SetObjects(self.keyframes)

    def update3DViewer(self):
        with self.cam_pos_lock:
            self._update_counter += 1
            if self.cam_pos is not None and self.cam_pos.shape[0] > 3:
                self.mayavi_view.scene.disable_render = True
                # Update path plot
                ms = self.mayavi_view.cam_path_plot.mlab_source
                ms.reset(x=self.cam_pos[:, 0],
                         y=self.cam_pos[:, 1],
                         z=self.cam_pos[:, 2],
                         scalars=np.arange(self.cam_pos.shape[0]))

                # Update quiver plot (current camera position)
                ms = self.mayavi_view.cam_pos_plot.mlab_source
                ms.reset(x=self.cam_pos[-1, 0],
                         y=self.cam_pos[-1, 1],
                         z=self.cam_pos[-1, 2],
                         u=self.cam_ray[-1, 0],
                         v=self.cam_ray[-1, 1],
                         w=self.cam_ray[-1, 2])

                # Update the map once each 30 updates
                if self._update_counter % 30 == 0 \
                        and self.map_points is not None:
                    ms = self.mayavi_view.map_plot.mlab_source
                    ms.reset(x=self.map_points[:, 0],
                             y=self.map_points[:, 1],
                             z=self.map_points[:, 2])

                    if self._update_counter == 30:
                        self.mayavi_view.scene.reset_zoom()

                mlab = self.mayavi_view.scene.mlab
                dist = la.norm(self.cam_pos[-1] - self.cam_pos[-2])
                self.mayavi_view.scene.disable_render = False

    def loadPLY(self, ply_path, clear_previous=False):
        self.mayavi_view.scene.disable_render = True
        with open(ply_path, 'r') as fp:
            line_num = 0
            line = 'X'
            while len(line) > 0:
                line = fp.readline()
                line_num += 1
                if line.startswith('end_header'):
                    break
        data = np.loadtxt(ply_path, skiprows=line_num)
        X = data[:, 0:3]
        color = data[:, 6:]
        s = np.max(color, axis=1)
        mlab = self.mayavi_view.scene.mlab
        if clear_previous:
            mlab.clf()
        mlab.points3d(X[:, 0], X[:, 1], X[:, 2], s,
                      colormap='gray', mode='point')
        self.mayavi_view.scene.reset_zoom()
        self.mayavi_view.scene.disable_render = False

    def loadOdometry(self, filename):
        with open(filename, 'r') as f:
            params = [line.split() for line in f.readlines()]

        kf_ts = [int(float(p[0])) for p in params]

        # Translation vectors
        t = [np.array([float(v) for v in p[1:4]]) for p in params]
        for ti in t:
            ti.shape = 3, 1

        # q for quaternions
        q = [np.array([float(v) for v in p[4:]]) for p in params]

        # Get Tcw
        Tcw = []
        pr = []
        for (qx, qy, qz, qw), t_ts in zip(q, t):
            R = np.array([[1 - 2*qy**2 - 2*qz**2,
                           2*qx*qy - 2*qz*qw,
                           2*qx*qz + 2*qy*qw],
                          [2*qx*qy + 2*qz*qw,
                           1 - 2*qx**2 - 2*qz**2,
                           2*qy*qz - 2*qx*qw],
                          [2*qx*qz - 2*qy*qw,
                           2*qy*qz + 2*qx*qw,
                           1 - 2*qx**2 - 2*qy**2]])

            T_ts = np.hstack((R, t_ts))
            pr_ts = np.dot(R, np.array([0, 0, 1.]))
            Tcw.append(T_ts)
            pr.append(pr_ts)

        self.mayavi_view.scene.mlab.clf()
        self.mayavi_view.scene.mlab.quiver3d([T[0, 3] for T in Tcw],
                                             [T[1, 3] for T in Tcw],
                                             [T[2, 3] for T in Tcw],
                                             [p[0] for p in pr],
                                             [p[1] for p in pr],
                                             [p[2] for p in pr],
                                             opacity=0.5,
                                             color=(1, 0, 0),
                                             mode='2darrow')
        if self.map_points is not None:
            mlab = self.mayavi_view.scene.mlab
            mlab.points3d(self.map_points[:, 0],
                          self.map_points[:, 1],
                          self.map_points[:, 2],
                          color=(0, 1, 0.2),
                          mode='point')

        work_dir = os.path.dirname(filename)
        fpaths = glob.glob(work_dir + os.sep + '*.jpg')
        fpaths.sort()
        fnames = [os.path.basename(fpath) for fpath in fpaths]
        fnums = [re.match('frame\-0*(\d+)\.jpg', fn).group(1) for fn in fnames]
        fnames = ['Frame %s' % n for n in fnums]

        self.keyframes = [{'num': i+1, 'frame': fn, 'path': fp}
                          for i, (fn, fp) in enumerate(zip(fnames, fpaths))]
        self.frame_list.DeleteAllItems()
        self.updateKeyFrames()

    def OnAbout(self, e):
        dlg = wx.MessageDialog(self,
                               "Create 3-D models for plants.",
                               "About 3-Demeter PlantCapture",
                               wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def OnSetDevice(self, e):

        args = ['v4l2-ctl', '--list-devices']
        l_cams = subprocess.check_output(args).split('\n')
        l_cams = [u.strip() for u in l_cams if len(u) > 0]
        cams = {l_cams[i]: l_cams[i+1] for i in range(0, len(l_cams), 2)}
        dialog = wx.SingleChoiceDialog(self, "Select camera device:", "Camera",
                                       cams.keys())
        if dialog.ShowModal() == wx.ID_OK:
            sel_device = dialog.GetStringSelection()
            self.control.set_cap_device(cams[sel_device])

        dialog.Destroy()

    def OnSetScaleCor(self, e):

        options = {'Look for scale correction pattern': True,
                   'Do not perform scale correction': False}
        dialog = wx.SingleChoiceDialog(self, "Scale correction:", "Options",
                                       options.keys())
        if dialog.ShowModal() == wx.ID_OK:
            sel_opt = dialog.GetStringSelection()
            self.control.setScaleCor(options[sel_opt])

        dialog.Destroy()

    def OnError(self, message):
        dialog = wx.MessageDialog(self, message, "Error", wx.OK)
        dialog.ShowModal()
        dialog.Destroy()

    def OnNewAcq(self, e):
        self.viewer_updater.updating = True
        self.viewer_updater.start()
        self.control.new_acquisition()

    def OnResumeAcq(self, e):
        self.viewer_updater = MayaviUpdater(self)
        self.viewer_updater.updating = True
        self.viewer_updater.start()
        self.control.resume_acquisition()

    def OnPauseAcq(self, e):
        self.viewer_updater.updating = False
        self.control.pause_acquisition()

    def OnFinishAcq(self, e):

        for t in ['cap_new', 'cap_continue', 'cap_pause', 'cap_finish']:
            self._tools[t].Enable(False)
        self._toolbar.Realize()

        self.control.finish_acquisition()
        self.viewer_updater.updating = False

    def OnSave2Dir(self, e):
        dlg = wx.DirDialog(self, message="Choose a directory")
        if dlg.ShowModal() == wx.ID_OK:
            full_path = dlg.GetPath()
            self.control.save_capture(full_path)

        dlg.Destroy()

    def OnOpenDir(self, e):
        dlg = wx.DirDialog(self,
                           message="Choose an existing directory",
                           style=wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            dir_path = dlg.GetPath()
            self.control.open_cap_dir(dir_path)

        dlg.Destroy()

    def OnExportMVS(self, e):
        self.control.export_to_mvs()

    def OnMVS(self, e):
        self.is_mvs_on = True
        dialog = pp.PyProgress(self, -1, "3-D Reconstruction",
                               "Running PMVS...",
                               agwStyle=wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME)

        dialog.SetGaugeProportion(0.2)
        dialog.SetGaugeSteps(50)
        dialog.SetGaugeBackground(wx.WHITE)
        dialog.SetFirstGradientColour(wx.WHITE)
        dialog.SetSecondGradientColour(wx.BLACK)

        self.control.start_mvs()

        while self.is_mvs_on:
            wx.MilliSleep(100)
            dialog.UpdatePulse()

        dialog.Destroy()

    def MVSFinished(self):
        self.is_mvs_on = False

    def notify(message, msg_type):
        if msg_type == 'error':
            wx.MessageBox(message, 'Error', wx.OK | wx.ICON_ERROR)
        elif msg_type == 'info':
            wx.MessageBox(message, 'Info', wx.OK | wx.ICON_INFORMATION)

    def OnNormalize(self, e):
        self.control.normalize()

    def OnExit(self, e):
        self.Close(True)  # Close the frame.

    def OnFrameSelected(self, event):
        i = event.GetIndex()
        kf_data = self.keyframes[i]

        frame = cv2.imread(kf_data['path'])

        squares = scale.preprocess_and_find_squares(frame)
        mapping = scale.detect_markers(frame, squares)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pts = [s.reshape((-1, 1, 2)) for s, _ in mapping.values()]
        frame_rgb = cv2.polylines(frame_rgb, pts, True, (255, 0, 0),
                                  thickness=3, lineType=cv2.LINE_AA)

        for k, (pt, _) in mapping.iteritems():
            x, y = pt.mean(axis=0)
            pos = (int(x), int(y))
            cv2.putText(frame_rgb, '%d' % (k), pos, cv2.FONT_ITALIC,
                        2, (255, 0, 0), 2, cv2.LINE_AA)

        w, h = PCapMainFrame.FRAME_VIEW_WIDTH, PCapMainFrame.FRAME_VIEW_HEIGHT
        thumbnail = cv2.resize(frame_rgb, (w, h))
        self.frame_image = wx.ImageFromData(w, h, thumbnail.tostring())
        self.bitmap = wx.BitmapFromImage(self.frame_image)
        self.frame_view.SetBitmap(self.bitmap)
        self.status_bar.SetStatusText('Frame %d' % kf_data['num'])

    def SetStatus(self, message):
        self.status_bar.SetStatusText(message)
