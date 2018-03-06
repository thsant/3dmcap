import cv2
import numpy as np
from scipy import linalg as la
import os
from os.path import isfile
from dmcap import util


m_base = [np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1, 0],
                    [0, 1, 1, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]])]

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 1, 0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0],
                        [0, 1, 0, 1, 0, 1, 0],
                        [0, 1, 0, 1, 0, 1, 0],
                        [0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0],
                        [0, 1, 1, 0, 1, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 1, 0],
                        [0, 1, 0, 1, 1, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0],
                        [0, 1, 0, 1, 1, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 1, 0],
                        [0, 1, 1, 1, 1, 0, 0],
                        [0, 1, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0],
                        [0, 1, 1, 0, 0, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0],
                        [0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 1, 0],
                        [0, 1, 1, 0, 1, 1, 0],
                        [0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0],
                        [0, 1, 0, 1, 0, 1, 0],
                        [0, 1, 0, 1, 0, 1, 0],
                        [0, 1, 1, 0, 0, 1, 0],
                        [0, 1, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 1, 0],
                        [0, 1, 1, 1, 0, 1, 0],
                        [0, 1, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 1, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 1, 1, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0],
                        [0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0, 1, 0],
                        [0, 0, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 1, 0, 0, 1, 1, 0],
                        [0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

# Plate Z
m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 1, 1, 0],
                        [0, 0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0],
                        [0, 1, 0, 0, 1, 1, 0],
                        [0, 1, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 1, 0],
                        [0, 0, 0, 0, 1, 1, 0],
                        [0, 1, 0, 1, 0, 0, 0],
                        [0, 1, 1, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 1, 0],
                        [0, 1, 1, 0, 0, 1, 0],
                        [0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 1, 0, 1, 0],
                        [0, 1, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 1, 0, 1, 0, 1, 0],
                        [0, 1, 0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0],
                        [0, 1, 0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 0, 1, 1, 0],
                        [0, 0, 1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 1, 0],
                        [0, 0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))

m_base.append(np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 1, 1, 0, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]]))


def find_squares(I_edges):
    squares = []
    _, contours, _ = cv2.findContours(I_edges, mode=cv2.RETR_CCOMP,
                                      method=cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.03*cnt_len, True)

        if len(cnt) == 4 and cv2.isContourConvex(cnt):
            if cv2.contourArea(cnt) > 2500:
                squares.append(cnt.reshape(-1, 2))

    return squares, contours


def rot_to_upleft(S):
    return np.roll(S, -np.argmin(S[:, 0] + S[:, 1]), axis=0)


def rotate_marker(M):
    M_90 = M.T[:, ::-1]
    M_180 = M_90.T[:, ::-1]
    M_270 = M_180.T[:, ::-1]
    return [M, M_90, M_180, M_270]


def compare_markers(M_i, M_j):
    for ri, M_r in enumerate(rotate_marker(M_j)):
        if (M_i == M_r).min() or (M_i[:, ::-1] == M_r).min():
            return True, (0, 90, 180, 270)[ri]

    return False, None


def preprocess_and_find_squares(I):
    I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I_he = cv2.equalizeHist(I_gray)
    smooth = cv2.medianBlur(I_he, 5)
    canny = cv2.Canny(smooth, 64, 128, apertureSize=3)
    squares, contours = find_squares(canny)
    return [rot_to_upleft(s) for s in squares]


def detect_markers(I, squares):
    """Find markers"""
    I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I_he = cv2.equalizeHist(I_gray)

    marker_map = []

    ideal_sqr = np.array([[0,  0],
                          [350, 0],
                          [350, 350],
                          [0, 350]], dtype=np.float32)

    for square in squares:
        H = cv2.getPerspectiveTransform(np.array(square, dtype=np.float32),
                                        ideal_sqr)
        S = cv2.warpPerspective(I_he, H, (350, 350))
        M = np.zeros_like(S)
        M[S >= 190] = 255
        M[S < 190] = 0
        marker = cv2.resize(M, (7, 7), interpolation=cv2.INTER_CUBIC)
        marker_map.append((M, marker/255, square))

    mapping = {}
    for i, (M_i, marker_i, sq_i) in enumerate(marker_map):
        matched = False
        for r, marker_ref in enumerate(m_base):
            matched, rot = compare_markers(marker_i, marker_ref)
            if matched:
                mapping[r] = (sq_i, rot)
    return mapping


def plot_mapping(I, mapping):
    from matplotlib import pyplot as plt
    for k, (sq_i, rot) in mapping.iteritems():
        mu_i = sq_i.mean(axis=0)
        plt.text(mu_i[0], mu_i[1], '%d' % k, color='y')
        plt.plot(sq_i[[0, 1, 2, 3, 0], 0], sq_i[[0, 1, 2, 3, 0], 1], 'y-')
        plt.plot(sq_i[[0, 1, 2, 3, 0], 0], sq_i[[0, 1, 2, 3, 0], 1], 'yo')
    plt.imshow(I, cmap=plt.cm.gray)


def normalize(K, path, marker_ref_dist):
    P, frame_path = util.load_acquisition_data(K, path)

    # Frame set F
    F = [k for k, v in frame_path.iteritems() if isfile(v)]

    M = {}
    visible = {m: [] for m in range(9)}
    for k in F:
        frame_k = cv2.imread(frame_path[k])
        squares = preprocess_and_find_squares(frame_k)
        M[k] = detect_markers(frame_k, squares)
        for m in M[k].keys():
            visible[m].append(k)

    pairs = {k: [v[0], v[-1]] for k, v in visible.iteritems() if len(v) >= 2}
    X = {m: util.compute_marker_3d_pos(M, P, m, i, j)
         for m, (i, j) in pairs.iteritems()}

    x_pairs = [(3, 0), (6, 3), (4, 1), (7, 4), (5, 2), (8, 5)]
    x_pairs = [(m_a, m_b)
               for m_a, m_b in x_pairs if m_a in X.keys() and m_b in X.keys()]
    y_pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]
    y_pairs = [(m_a, m_b)
               for m_a, m_b in y_pairs if m_a in X.keys() and m_b in X.keys()]

    if len(x_pairs) == 0 or len(y_pairs) == 0:
        msg = 'Unable to normalize - no enough markers'
        return False, msg

    ply_path = path + '/pmvs/models/3dmc-3dmodel.cfg.ply'
    data, ply_header = util.load_ply(ply_path)

    # Points
    cloud = data[:, 0:3]
    # Normal vectors
    nvec = data[:, 3:6]
    # Color (R,G,B) tuples
    color = data[:, 6:]

    dy = np.array([la.norm(X[v] - X[u]) for u, v in y_pairs])

    cloud_n = cloud * (marker_ref_dist/dy.mean())

    Xn = {k: v * (marker_ref_dist/dy.mean()) for k, v in X.iteritems()}
    origin = Xn[4][0:3]
    # New X-axis vector
    u, v = x_pairs[0]
    vx = Xn[v] - Xn[u]
    vx = vx[0:3]
    vx = vx/la.norm(vx)
    # New Y-axis vector
    u, v = y_pairs[0]
    vy = Xn[v] - Xn[u]
    vy = vy[0:3]
    vy = vy/la.norm(vy)
    # New Z-axis vector
    vz = -1 * np.cross(vx, vy)
    # Ensure orthogonal basis (avoiding model distortion)
    vy = -1 * np.cross(vx, vz)

    R_o = la.inv(np.vstack((vx, vy, vz)).T)
    O_o = np.dot(R_o, origin.T).T
    cloud_o = np.dot(R_o, cloud_n.T).T - O_o
    nvec_o = np.dot(R_o, nvec.T).T
    data_o = np.hstack((cloud_o, nvec_o, color))

    ply_norm_path = path + '/pmvs/models/3dmc-3dmodel.norm.ply'
    util.dump_ply(ply_norm_path, ply_header, data_o)

    return True, ply_norm_path
