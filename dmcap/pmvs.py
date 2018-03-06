import cv2
import numpy as np
import os
import shutil
import subprocess
import sys

from os.path import isfile
from scipy import linalg as la


def quaternion2rot(x, y, z, w):
    R = np.array([[1. - 2*y**2 - 2*z**2, 2*x*y + 2*w*z, 2*x*z - 2*w*y],
                  [2*x*y - 2*w*z, 1. - 2*x**2 - 2*z**2, 2*y*z + 2*w*x],
                  [2*x*z + 2*w*y, 2*y*z - 2*w*x, 1. - 2*x**2 - 2*y**2]])
    return R


def null(A, eps=1e-15):
    """
    http://mail.scipy.org/pipermail/scipy-user/2005-June/004650.html
    """
    u, s, vh = la.svd(A)
    n = A.shape[1]   # the number of columns of A
    if len(s) < n:
        expanded_s = np.zeros(n, dtype=s.dtype)
        expanded_s[:len(s)] = s
        s = expanded_s
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, vh, axis=0)

    return np.transpose(null_space)


def export2pmvs(K, input_path):

    odo_file = open(input_path + '/3dmc_odometry.txt')
    lines = odo_file.readlines()
    odo_file.close()

    odo_array = np.array([[float(f) for f in l.split()] for l in lines])

    R = {}
    P = {}
    C = {}

    for ts, tx, ty, tz, qx, qy, qz, qw in odo_array:
        # ts is the timestamp
        R[ts] = quaternion2rot(qx, qy, qz, qw)
        Cn = np.array([tx, ty, tz]).reshape(3, 1)
        RC = np.dot(-R[ts], Cn)
        P[ts] = np.dot(K, np.hstack((R[ts], RC)))
        C[ts] = np.array([tx, ty, tz, 1.]).reshape(4, 1)

    T = odo_array[:, 0]

    frame_file_names = {ti: input_path + os.sep + 'frame-%.5d.jpg' % int(ti)
                        for ti in T}
    available = {k: isfile(v) for k, v in frame_file_names.iteritems()}
    frame = {k: cv2.imread(v.encode(sys.getfilesystemencoding()))
             for k, v in frame_file_names.iteritems() if available[k]}

    t_avail = frame.keys()
    t_avail.sort()

    pmvs_dir = input_path + os.sep + 'pmvs'

    if os.access(pmvs_dir, os.F_OK):
        shutil.rmtree(pmvs_dir)
    os.mkdir(pmvs_dir)
    os.mkdir(pmvs_dir + os.sep + 'models')
    os.mkdir(pmvs_dir + os.sep + 'visualize')
    for i, ti in enumerate(t_avail):
        target_path = pmvs_dir + '/visualize/%08d.jpg' % i
        cv2.imwrite(target_path.encode(sys.getfilesystemencoding()), frame[ti])

    os.mkdir(pmvs_dir + os.sep + 'txt')
    for i, ti in enumerate(t_avail):
        with open('%s/txt/%08d.txt' % (pmvs_dir, i), 'w') as f:
            f.write('CONTOUR\n')
            np.savetxt(f, P[ti], '%.6f %.6f %.6f %.6f')

    opt_basename = '3dmc-3dmodel.cfg'
    opt = {'level': 1,
           'csize': 2,
           'threshold': 0.7,
           'wsize': 7,
           'minImageNum': 3,
           'CPU': 4,
           'setEdge': 0,
           'useBound': 0,
           'useVisData': 0,
           'sequence': -1,
           'maxAngle': 5,
           'quad': 2.0,
           'oimages': 0}

    opt_file = open(pmvs_dir + '/' + opt_basename, 'w')

    for k, v in opt.iteritems():
        opt_file.write(k + " " + str(v) + "\n")

    n = len(frame)
    opt_file.write("timages -1 0 " + str(n) + "\n")
    opt_file.close()


def run_pmvs(path):
    pmvs_path = path + os.sep + 'pmvs' + os.sep
    print 'Running pmvs2 on %s...' % pmvs_path
    args = ['pmvs2', pmvs_path, '3dmc-3dmodel.cfg']
    subprocess.check_call(args)
    print 'Done!'
    return pmvs_path + 'models/3dmc-3dmodel.cfg.ply'
