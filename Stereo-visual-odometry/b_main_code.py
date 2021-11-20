import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import least_squares
from functions import *
from helperFunctions import *

directory_left = '/home/hippo/notebook/Stereo-visual-odometry/sync_image2/image_00/data/'
directory_right = '/home/hippo/notebook/Stereo-visual-odometry/sync_image2/image_01/data/'
directory_calib = '/home/hippo/notebook/Stereo-visual-odometry/calib.txt'
imdir_l = []
imdir_r = []

imdir_list = []
for filename in os.listdir(directory_left):
    imdir_list.append(filename)
imdir_list.sort()

tile = [10, 20]
thresh = 30

iter = 0

# Those parameters are not mine...
translation = None
rotation = None
canvasH = 300
canvasW = 300

traj = []
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
Proj_0, Proj_1 = extract_projM(directory_calib)

# The steps are explained in the loop in which similar ones are repeated.
im_l1 = cv2.imread(directory_left + imdir_list[0], 0)
im_r1 = cv2.imread(directory_right + imdir_list[0], 0)
disparity1 = stereo.compute(im_l1,im_r1)
F1 = get_keypoints(im_l1, tile, thresh)
F1_3d, F1 = extract_wc(F1, disparity1, Proj_0, Proj_1)
descriptor1 = extract_desc(im_l1, F1)

for filename in imdir_list[1:]:

    print("iteration is:" + str(iter)+ " of " + str(len(imdir_list)))
    iter = iter + 1
    
    # New frame loaded
    im_l2 = cv2.imread(directory_left + filename, 0)
    im_r2 = cv2.imread(directory_right + filename, 0)

    # Disparity map from the stereo image
    disparity2 = stereo.compute(im_l2,im_r2)

    # Feature Extracting
    F2 = get_keypoints(im_l2, tile, thresh)

    # Extract world coordinates
    F2_3d, F2 = extract_wc(F2,disparity2, Proj_0, Proj_1)

    # Calculate Descriptors
    descriptor2 = extract_desc(im_l2, F2)

    # Next Iteration
    im_l1 = im_l2
    im_r1 = im_r2
    disparity1 = disparity2
    F1 = F2
    F1_3d = F2_3d
    descriptor1 = descriptor2

    if iter == 1:
        break




cv2.imshow('stereo', stereo)
cv2.waitKey(0)
cv2.closeAllWindows()
np.savetxt('debug.txt', np.round(S,2), delimiter=',')
# Feature Tracking
#points_tracked_l1, points_tracked_l2 = feature_tracking(im_l1, im_l2, kp)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(F2_3d[:,0], F1_3d[:,1], F1_3d[:,2], c=F1_3d[:,2], cmap='Greens');
# plt.show()


# traj = np.array(traj)print
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
# plt.plot(traj[:,0], traj[:,2])
# plt.savefig("output.png")
# plt.show()
  