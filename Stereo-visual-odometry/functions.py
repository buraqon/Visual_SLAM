import cv2
import numpy as np
from math import cos, sin
from helperFunctions import *


def get_keypoints(im_l1, tile, thresh):

    fast = cv2.FastFeatureDetector_create(thresh);
    h, w = im_l1.shape
    kp = []
    idx = 0
    for y in range(0, h, tile[0]):
        for x in range(0, w, tile[1]):
            imBox = im_l1[y:y+tile[0], x:x+tile[1]]
            keypoints = fast.detect(imBox)
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            if (len(keypoints) > 10):
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                for kpt in keypoints[0:10]:
                    kp.append(kpt)
            else:
                for kpt in keypoints:
                    kp.append(kpt)
    kp = cv2.KeyPoint_convert(kp)

    return kp

def feature_tracking(im_l1, im_l2, kp):

    H, W = im_l1.shape
    #convert keypoints to x,y positions
    points_tracked_1 = cv2.KeyPoint_convert(kp)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    points_tracked_2, is_tracked, error = cv2.calcOpticalFlowPyrLK(im_l1, im_l2, points_tracked_1, None, flags=cv2.MOTION_AFFINE, **lk_params)
    # is_tracked is a boolean to show if the point is tracked or not

    points_tracked_1 = points_tracked_1[is_tracked.T[0]==1]
    points_tracked_2 = points_tracked_2[is_tracked.T[0]==1]
    error = error[is_tracked.T[0]==1]

    error_thresh = 4
    points_tracked_1 = points_tracked_1[error.T[0]<error_thresh]
    points_tracked_2 = np.around(points_tracked_2[error.T[0]<error_thresh])

    hPts = np.where(points_tracked_2[:,1] >= 375)
    wPts = np.where(points_tracked_2[:,0] >= W)
    outTrackPts = hPts[0].tolist() + wPts[0].tolist()
    outDeletePts = list(set(outTrackPts))

    if len(outDeletePts) > 0:
        points_tracked_1 = np.delete(points_tracked_1, outDeletePts, axis=0)
        points_tracked_2 = np.delete(points_tracked_2, outDeletePts, axis=0)     

    # for i in range(points_tracked_1.shape[0]):
    #     p1 = points_tracked_1[i]
    #     p2 = points_tracked_2[i]
    #     im_l1_debug= cv2.arrowedLine(im_l1, (int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])), color=(255, 0, 0), thickness=1)

    return points_tracked_1, points_tracked_2

def extract_wc(F, disparity, Proj_0, Proj_1):
    disparityMinThres = 0
    disparityMaxThres = 999

    Fr = np.copy(F)
    selected = np.zeros(F.shape[0])

    for i in range(F.shape[0]):
        T1Disparity = disparity[int(F[i,1]), int(F[i,0])]
        if (T1Disparity > disparityMinThres and T1Disparity < disparityMaxThres):
            Fr[i, 0] = F[i, 0] - T1Disparity
            selected[i] = 1

    selected = selected.astype(bool)
    F_3d = F[selected, ...]
    Fr_3d = Fr[selected, ...]

    # 3d point cloud triagulation
    numPoints = F_3d.shape[0]
    point_3d = generate3DPoints(F_3d, Fr_3d, Proj_0, Proj_1)
    #print(point_3d.shape)
    return point_3d, F_3d

def extract_projM(directory_calib):
    found = 0
    calib_file = open(directory_calib, "r")
    while not found:
        line = calib_file.readline()
        phrases = line.split(" ")
        if phrases[0] == "P_rect_00:":
            cam_proj_0 = phrases

        if phrases[0] == "P_rect_01:":
            cam_proj_1 = phrases
            found = 1

    Proj_0 = np.zeros((3,4))
    Proj_1 = np.zeros((3,4))
    for row in range(3):
        for column in range(4):
            Proj_0[row, column] = cam_proj_0[row*4+column+1]
            Proj_1[row, column] = cam_proj_1[row*4+column+1]
    return Proj_0, Proj_1

def extract_desc(img, F):
    descriptors = []
    size = 11
    for feat in F:
        d = []
        for i in range(size):
            for j in range(size):
                if(i!=5 or j!=5):
                    d.append(img[int(feat[1])-5+i, int(feat[0])-5+j])
        descriptors.append(d)
    descriptors = np.array(descriptors)
    return  descriptors

def score_matrix(descriptor1, descriptor2):
    S = np.zeros((descriptor1.shape[0], descriptor2.shape[0]))
    #print(S.shape)
    for i in range(descriptor1.shape[0]):
        for j in range(descriptor2.shape[0]):
            diff = descriptor1[i] - descriptor2[j]
            adiff = np.absolute(diff)
            s = np.sum(adiff)
            S[i,j] = s
    return S

def match_features(S):
    match_array = []
    for i in range(S.shape[0]):
        row = S[i,:]
        fbb = np.min(row)
        ind = np.array(np.where(row == fbb))[0,:]
        for j in ind:
            col = S[:, j].T
            fab = np.min(col)
            if fab == fbb and fab< 4000:
                match_array.append([i,j])
                break
    match_array = np.array(match_array)
    return match_array