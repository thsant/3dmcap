import numpy as np
from scipy import linalg as la
import os


def load_params(yaml_file_path):
    with open(yaml_file_path, 'r') as fs:
        yaml_input = fs.read()

    import yaml
    # yaml package is not very robust, so we are cleaning the input
    yaml_clean = ''
    for l in yaml_input.split('\n'):
        if not l.startswith('#') and not l.startswith('%YAML'):
            yaml_clean += l + '\n'
    return yaml.load(yaml_clean)


def quaternion2rot(x, y, z, w):
    R = np.array([[1. - 2*y**2 - 2*z**2, 2*x*y + 2*w*z, 2*x*z - 2*w*y],
                  [2*x*y - 2*w*z, 1. - 2*x**2 - 2*z**2, 2*y*z + 2*w*x],
                  [2*x*z + 2*w*y, 2*y*z - 2*w*x, 1. - 2*x**2 - 2*y**2]])
    return R


def load_acquisition_data(K, input_path):
    odo_file = open(input_path + '/3dmc_odometry.txt')
    lines = odo_file.readlines()
    odo_file.close()

    odo_array = np.array([[float(f) for f in l.split()] for l in lines])
    R = {}
    P = {}
    for ts, tx, ty, tz, qx, qy, qz, qw in odo_array:
        # ts is the timestamp
        R[ts] = quaternion2rot(qx, qy, qz, qw)
        Cn = np.array([tx, ty, tz]).reshape(3, 1)
        RC = np.dot(-R[ts], Cn)
        P[int(ts)] = np.dot(K, np.hstack((R[ts], RC)))

    T = odo_array[:, 0]

    frame_path = {int(ti): input_path + os.sep + 'frame-%.5d.jpg' % int(ti)
                  for ti in T}

    return P, frame_path


def load_ply(ply_path):
    fp = open(ply_path, 'r')
    line_num = 0
    line = 'X'
    ply_header = ''
    while len(line) > 0:
        line = fp.readline()
        ply_header += line
        line_num += 1
        if line.startswith('end_header'):
            break
    fp.close()
    data = np.loadtxt(ply_path, skiprows=line_num)

    return data, ply_header


def dump_ply(ply_path, ply_header, data):
    with open(ply_path, 'w') as fp:
        fp.write(ply_header)

        line_format = '%.4f %.4f %.4f %.4f %.4f %.4f %d %d %d\n'
        for (x, y, z, nx, ny, nz, r, g, b) in data:
            line = line_format % (x, y, z, nx, ny, nz, r, g, b)
            fp.write(line)


def dlt_triangulation(ui, Pi, uj, Pj):
    """Hartley & Zisserman, 12.2."""
    ui /= ui[2]
    xi, yi = ui[0], ui[1]

    uj /= uj[2]
    xj, yj = uj[0], uj[1]

    a0 = xi * Pi[2, :] - Pi[0, :]
    a1 = yi * Pi[2, :] - Pi[1, :]
    a2 = xj * Pj[2, :] - Pj[0, :]
    a3 = yj * Pj[2, :] - Pj[1, :]

    A = np.vstack((a0, a1, a2, a3))
    U, s, VT = la.svd(A)
    V = VT.T

    X3d = V[:, -1]

    return X3d/X3d[3]


def compute_marker_3d_pos(M, P, m, i, j):
    x_i = M[i][m][0].mean(axis=0)
    x_j = M[j][m][0].mean(axis=0)

    homog_x_i = np.array([x_i[0], x_i[1], 1])
    homog_x_j = np.array([x_j[0], x_j[1], 1])
    return dlt_triangulation(homog_x_i, P[i], homog_x_j, P[j])
