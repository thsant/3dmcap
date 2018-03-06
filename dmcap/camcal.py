#!/usr/bin/env python

from Tkinter import *
import ttk
import tkMessageBox
import tkFileDialog

from PIL import ImageTk, Image

import cv2
import numpy as np
import threading
from threading import Lock
import Queue
import argparse


def mat2tk(img_mat, size=None):
    # PIL thumbnails function is slow and will produce
    # delays for large images. Let's use a really fast
    # method to avoid any lag in the live frame view.
    if size is None:
        pil_img = Image.fromarray(img_mat)
    else:
        src_width = img_mat.shape[1]
        dst_width = size[0]
        assert isinstance(dst_width, int)
        step = src_width / dst_width
        pil_img = Image.fromarray(img_mat[::step, ::step])

    return ImageTk.PhotoImage(pil_img)


def find_chess_board(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(img_gray, (9, 6))
    img_chess = None
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img_gray, corners, (5, 5), (-1, -1), term)
        img_chess = img.copy()
        cv2.drawChessboardCorners(img_chess, (9, 6), corners, found)

    return found, img_chess


class CalibrationApp:
    def __init__(self, device, width=1920, height=1080, square_size=1.):
        """

        :type square_size: int
        """
        self.device = device  # camera device (ex.: /dev/video0 on Linux)
        self.width = width
        self.height = height
        self.square_size = square_size

        self.capture_lock = Lock()
        self.capture = None
        self.input_frame = None
        self.frames = []
        self.frames_thumb = []
        self.frames_tiny = []
        self.cur_frame_idx = 0
        self.K = None
        self.dist_coefs = None
        self.rms = 0
        self.size_msg = 'Unknown size.'
        self.start_capture()

        self.tk_root = Tk()
        self.tk_root.title('Camera calibration')

        self.main_frame = ttk.Frame(self.tk_root, padding="3 3 12 12")
        self.main_frame.grid(column=0, row=0, sticky=(N, W, E, S))
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        thumb_w = 320
        thumb_h = int(float(thumb_w) / self.width * self.height)

        self.thumb_size = thumb_w, thumb_h

        pil_img = Image.new('RGB', self.thumb_size)
        self.blank = ImageTk.PhotoImage(pil_img)
        live_frame = ttk.Labelframe(self.main_frame, text='Live camera')
        live_frame.grid(column=0, row=0, sticky='NWES', padx=(3, 3))

        self.in_frame_label = ttk.Label(live_frame, image=self.blank)
        self.in_frame_label.grid(column=0, row=0, padx=11, pady=11)

        self.size_label = ttk.Label(live_frame, text=self.size_msg)
        self.size_label.grid(column=0, row=1, padx=11, sticky=W)

        b = ttk.Button(live_frame, text="Add", command=self.add_frame)
        b.grid(column=0, row=2, padx=3, pady=5)

        # Start: 'Selected frames' labeled frame
        browser_frame = ttk.Labelframe(self.main_frame, text='Selected frames')
        browser_frame.grid(column=1, row=0, rowspan=5, sticky='NWES', padx=(3, 3))

        pil_img = Image.new('RGB', (thumb_w / 2, thumb_h / 2))
        self.blank_sthumb = ImageTk.PhotoImage(pil_img)
        self.left_flabel = ttk.Label(browser_frame,
                                     image=self.blank_sthumb,
                                     compound=TOP)
        self.left_flabel.pack(side='left')
        self.left_flabel.grid(column=0, row=0, padx=11, pady=11)

        self.center_flabel = ttk.Label(browser_frame,
                                       image=self.blank,
                                       compound=TOP)
        self.center_flabel.grid(column=1, row=0, columnspan=3, padx=11, pady=11)

        self.right_flabel = ttk.Label(browser_frame,
                                      image=self.blank_sthumb,
                                      compound=TOP)
        self.right_flabel.grid(column=4, row=0, padx=11, pady=11)

        b = ttk.Button(browser_frame, text="< Previous", command=self.prev_frame)
        b.grid(column=1, row=1, padx=3, pady=5)

        b = ttk.Button(browser_frame, text="Remove", command=self.remove_frame)
        b.grid(column=2, row=1, padx=3, pady=5)

        b = ttk.Button(browser_frame, text="Next >", command=self.next_frame)
        b.grid(column=3, row=1, padx=3, pady=5)
        # End: 'Selected frames'

        # Start: 'Calibration' labeled frame
        cal_frame = ttk.Labelframe(self.main_frame, text='Calibration')
        cal_frame.grid(column=1, row=5, sticky='NWES', padx=(3, 3), pady=(11, 3))

        mat_frame = ttk.Labelframe(cal_frame, text='Internal Matrix')
        mat_frame.grid(column=0, row=0, columnspan=2, sticky='NWES', padx=7, pady=7)
        self.params_labels = [[], [], []]
        for i in range(3):
            self.params_labels[i].extend([ttk.Label(mat_frame, text='?', anchor=E)
                                          for _ in range(3)])
        for i in range(3):
            for j in range(3):
                self.params_labels[i][j].grid(column=j, row=i, sticky=E,
                                              padx=3, pady=3)

        dist_frame = ttk.Labelframe(cal_frame, text='Lens distortion')
        dist_frame.grid(column=3, row=0, sticky='NWES', padx=7, pady=7)
        self._dist_coef_names = ['k1', 'k2', 'p1', 'p2', 'k3']
        self.dist_coef_labels = {k: ttk.Label(dist_frame, text='%s: ?' % k)
                                 for k in self._dist_coef_names}
        self.dist_dict = {k: '?' for k in self._dist_coef_names}
        for row_i, k in enumerate(self._dist_coef_names):
            self.dist_coef_labels[k].grid(column=0, row=row_i, sticky='W', padx=3, pady=3)

        b = ttk.Button(cal_frame, text="Calibrate", command=self.calibrate)
        b.grid(column=0, row=1, padx=7, pady=5)
        b = ttk.Button(cal_frame, text="Save", command=self.save_calibration)
        b.grid(column=1, row=1, padx=3, pady=5)
        # Start: 'Calibration' labeled frame

        # Start: Status bar
        f = ttk.Frame(self.main_frame, relief=RIDGE, borderwidth=1)
        f.grid(column=0, row=6, columnspan=2, sticky='NWES', padx=(3, 3), pady=(3, 0))
        self.status = ttk.Label(f, anchor=E)
        self.status.grid(column=0, row=0, sticky='NWES')
        # End: Status bar

        # Start: 'Settings' labeled frame
        set_frame = ttk.Labelframe(self.main_frame, text='Settings')
        set_frame.grid(column=0, row=5, sticky='NWES', padx=(3, 3), pady=(11, 3))

        sqr_size_str = StringVar()
        sqr_size_str.set(str(self.square_size))

        l = ttk.Label(set_frame, text='Square size (mm):')
        l.grid(column=0, row=0, padx=(7, 3), pady=(3, 3), sticky='E')
        self.sqrsize_entry = ttk.Entry(set_frame, width=5, textvariable=sqr_size_str)
        self.sqrsize_entry.grid(column=1, row=0, columnspan=3,
                                padx=(3, 7), pady=(3, 3), sticky='W')

        device_str = StringVar()
        device_str.set(str(self.device))

        l = ttk.Label(set_frame, text='Camera device:')
        l.grid(column=0, row=1, padx=(7, 3), pady=(3, 3), sticky='E')
        self.device_entry = ttk.Entry(set_frame, width=10, textvariable=device_str)
        self.device_entry.grid(column=1, row=1, columnspan=3,
                               padx=(3, 7), pady=(3, 3), sticky='W')

        width_str = StringVar()
        width_str.set(str(self.width))
        height_str = StringVar()
        height_str.set(str(self.height))

        l = ttk.Label(set_frame, text='Resolution:')
        l.grid(column=0, row=2, padx=(7, 3), pady=(3, 3), sticky='E')
        self.width_entry = ttk.Entry(set_frame, width=4, textvariable=width_str)
        self.width_entry.grid(column=1, row=2, padx=(3, 7), pady=(3, 3), sticky='W')
        l = ttk.Label(set_frame, text='x')
        l.grid(column=2, row=2)
        self.height_entry = ttk.Entry(set_frame, width=4, textvariable=height_str)
        self.height_entry.grid(column=3, row=2, padx=(3, 7), pady=(3, 3), sticky='W')

        b = ttk.Button(set_frame, text="Update settings", command=self.update_settings)
        b.grid(column=0, row=3, columnspan=4, padx=7, pady=5)
        # End: 'Settings' labeled frame

        # Starting frame capture thread
        self.thread_queue = Queue.Queue(maxsize=10)
        target_args = {'thread_queue': self.thread_queue}
        self.capture_thread = threading.Thread(target=self.capture_loop,
                                               kwargs=target_args)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        self.tk_root.after(100, self.listen_for_frame)

        self.tk_root.protocol("WM_DELETE_WINDOW", self._quit)
        self.tk_root.mainloop()

    def _quit(self):
        self.tk_root.quit()
        print 'Bye'
        sys.exit(0)

    def update_settings(self):

        restart_capture = False

        self.square_size = int(self.sqrsize_entry.get())

        new_val = int(self.width_entry.get())
        if self.width != new_val:
            self.width = new_val
            restart_capture = True

        new_val = int(self.height_entry.get())
        if self.height != new_val:
            self.height = new_val
            restart_capture = True

        new_val = self.device_entry.get()
        if self.device != new_val:
            self.device = new_val
            restart_capture = True

        if restart_capture:
            self.start_capture()
            self.status.configure(text='Capture restarted.')
            self.size_label.configure(text=self.size_msg)

    def start_capture(self):

        self.capture_lock.acquire()
        # Drop all previously acquired frames
        self.frames = []
        self.frames_thumb = []
        self.frames_tiny = []

        self.cur_frame_idx = 0
        self.K = None
        if self.capture is not None:
            self.capture.release()

        self.capture = cv2.VideoCapture(self.device)

        if self.capture is None:
            print "Error capturing from device '%d'" % self.device
            sys.exit(1)

        # Try to set resolution to the settings values
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        success, frame = self.capture.read()
        if success:
            height, width, _ = frame.shape
            if height != self.height or width != self.width:
                print "Warning: captured frame does not present the desired\
                 resolution. Gotten %d x %d, expecting %d x %d" \
                      % (width, height, self.width, self.height)

                self.height = height
                self.width = width

            self.size_msg = "Original size: %d x %d pixels" % (self.width, self.height)
        else:
            print "Error capturing from device " + str(self.device)
            sys.exit(1)

        self.capture_lock.release()

    def capture_loop(self, thread_queue=None):

        self.capture_lock.acquire()
        success, frame = self.capture.read()
        self.capture_lock.release()

        while True:
            frame = frame[:, :, (2, 1, 0)]
            thread_queue.put(frame, True, None)
            self.capture_lock.acquire()
            success, frame = self.capture.read()
            self.capture_lock.release()

    def listen_for_frame(self):
        """
        Check if there is a new frame queue
        """
        try:
            self.input_frame = self.thread_queue.get()
            tk_img = mat2tk(self.input_frame, self.thumb_size)
            self.in_frame_label.configure(image=tk_img)
            self.in_frame_label.image = tk_img
        except Queue.Empty:
            pass

        self.tk_root.after(10, self.listen_for_frame)

    def add_frame(self):
        f = self.input_frame.copy()
        self.frames.append(f)

        # Storing display images for GUI use
        thumb_size = self.thumb_size
        tiny_size = tuple(v / 2 for v in self.thumb_size)
        f_thumb = cv2.resize(f, thumb_size, cv2.INTER_CUBIC)
        f_tiny = cv2.resize(f, tiny_size, cv2.INTER_CUBIC)

        found, img_chess = find_chess_board(f_thumb)
        if found:
            self.frames_thumb.append(mat2tk(img_chess))
        else:
            self.frames_thumb.append(mat2tk(f_thumb))

        self.frames_tiny.append(mat2tk(f_tiny))

        self.cur_frame_idx = len(self.frames) - 1
        self.update_browser()

    def prev_frame(self):
        self.cur_frame_idx = max((self.cur_frame_idx - 1, 1))
        self.update_browser()

    def next_frame(self):
        self.cur_frame_idx = min((self.cur_frame_idx + 1, len(self.frames) - 1))
        self.update_browser()

    def remove_frame(self):
        try:
            self.frames.pop(self.cur_frame_idx)
            self.frames_thumb.pop(self.cur_frame_idx)
            self.frames_tiny.pop(self.cur_frame_idx)
        except IndexError:
            tkMessageBox.showwarning('No frames',
                                     'Impossible to remove a frame.')

        self.cur_frame_idx = max(self.cur_frame_idx - 1, 0)
        self.update_browser()

    def update_browser(self):
        n_frames = len(self.frames)

        if n_frames == 0:
            f_l = f_r = self.blank_sthumb
            f_c = self.blank
            l_l = l_c = l_r = ''
        elif n_frames == 1:
            f_l = f_r = self.blank_sthumb
            f_c = self.frames_thumb[0]
            l_l, l_c, l_r = '', 0, ''
        else:
            ci = self.cur_frame_idx
            li = (ci - 1) % n_frames
            ri = (ci + 1) % n_frames

            f_l, f_r = self.frames_tiny[li], self.frames_tiny[ri]
            f_c = self.frames_thumb[ci]
            l_l = '%d/%d' % (li + 1, n_frames)
            l_c = '%d/%d' % (ci + 1, n_frames)
            l_r = '%d/%d' % (ri + 1, n_frames)

        self.left_flabel.configure(image=f_l)
        self.left_flabel.image = f_l
        self.left_flabel.configure(text=l_l)

        self.center_flabel.configure(image=f_c)
        self.center_flabel.image = f_c
        self.center_flabel.configure(text=l_c)

        self.right_flabel.configure(image=f_r)
        self.right_flabel.image = f_r
        self.right_flabel.configure(text=l_r)

    def calibrate(self):

        n_frames = len(self.frames)
        if n_frames < 2:
            tkMessageBox.showwarning('Insufficient images',
                                     'Please, grab more images before calibration.')
            return

        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= self.square_size

        obj_points = []
        img_points = []

        h, w, _ = self.frames[0].shape
        for img in self.frames:
            h, w = img.shape[:2]
            found, corners = cv2.findChessboardCorners(img, pattern_size)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                                 corners, (5, 5), (-1, -1), term)

                img_points.append(corners.reshape(-1, 2))
                obj_points.append(pattern_points)

        self.rms, self.K, self.dist_coefs, _, _ = cv2.calibrateCamera(obj_points,
                                                                      img_points,
                                                                      (w, h),
                                                                      None,
                                                                      None)

        # Updating GUI ('Calibration' labeled frame and status bar)
        for i in range(3):
            for j in range(3):
                self.params_labels[i][j].configure(text='%.4f' % self.K[i, j])

        for k, v in zip(self._dist_coef_names, self.dist_coefs[0]):
            self.dist_coef_labels[k].configure(text='%s: %.4f' % (k, v))

        self.status.configure(text='Re-projection error: %.4f' % self.rms)

    def save_calibration(self):

        if self.K is None:
            tkMessageBox.showwarning('No calibration',
                                     'Get a set of images and calibrate before saving.')
            return

        filename = tkFileDialog.asksaveasfilename(parent=self.tk_root,
                                                  title='Save camera calibration to...',
                                                  defaultextension='.yaml')
        self.status.configure(text='Saving calibration to %s.' % filename)

        with open(filename, 'w') as fp:
            fp.write('%YAML:1.0\n\n')
            fp.write('# Camera parameters\n\n')
            fp.write('Camera.fx: %.10f\n' % self.K[0, 0])
            fp.write('Camera.fy: %.10f\n' % self.K[1, 1])
            fp.write('Camera.cx: %.10f\n' % self.K[0, 2])
            fp.write('Camera.cy: %.10f\n\n' % self.K[1, 2])
            fp.write('Camera.k1: %.10f\n' % self.dist_coefs[0, 0])
            fp.write('Camera.k2: %.10f\n' % self.dist_coefs[0, 1])
            fp.write('Camera.p1: %.10f\n' % self.dist_coefs[0, 2])
            fp.write('Camera.p2: %.10f\n' % self.dist_coefs[0, 3])
            fp.write('Camera.k3: %.10f\n' % self.dist_coefs[0, 4])


if __name__ == "__main__":
    desc = 'Calibrate a fixed focus camera using a chessboard.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--squaresize', dest='square_size', type=int,
                        default=1., help='Chessboard square size (prefer mm).')
    parser.add_argument('--device', dest='device',
                        default=0, help='Camera device identifier (ex.: "/dev/video0".')

    parser.add_argument('--fwidth', dest='fwidth', type=int,
                        default=1920, help='Frame width (prefered resolution).')
    parser.add_argument('--fheight', dest='fheight', type=int,
                        default=1080, help='Frame height (prefered resolution).')

    args = parser.parse_args()

    call_app = CalibrationApp(device=args.device,
                              square_size=args.square_size,
                              width=args.fwidth,
                              height=args.fheight)
