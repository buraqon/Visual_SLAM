 Finding points in 3d
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
    disparity1 = stereo.compute(im_l1,im_r1)
    disparity2 = stereo.compute(im_l2,im_r2)

    disparityMinThres = 0
    disparityMaxThres = 999

    points_tracked_r1 = np.copy(points_tracked_l1)
    points_tracked_r2 = np.copy(points_tracked_l2)
    selectedPointMap = np.zeros(points_tracked_l1.shape[0])

    for i in range(points_tracked_l1.shape[0]):
        #point_disp_1 = disparity1[int(points_tracked_l1[i,0]), int(points_tracked_l2[i,1])]
        T1Disparity = disparity1[int(points_tracked_l1[i,1]), int(points_tracked_l1[i,0])]
        T2Disparity = disparity2[int(points_tracked_l2[i,1]), int(points_tracked_l2[i,0])]

        if (T1Disparity > disparityMinThres and T1Disparity < disparityMaxThres
            and T2Disparity > disparityMinThres and T2Disparity < disparityMaxThres):
            points_tracked_r1[i, 0] = points_tracked_l1[i, 0] - T1Disparity
            points_tracked_r2[i, 0] = points_tracked_l1[i, 0] - T2Disparity
            selectedPointMap[i] = 1

    selectedPointMap = selectedPointMap.astype(bool)
    trackPoints1_KLT_L_3d = points_tracked_l1[selectedPointMap, ...]
    points_tracked_r1_3d = points_tracked_r1[selectedPointMap, ...]
    trackPoints2_KLT_L_3d = points_tracked_l2[selectedPointMap, ...]
    points_tracked_r2_3d = points_tracked_r2[selectedPointMap, ...]

    # 3d point cloud triagulation
    numPoints = trackPoints1_KLT_L_3d.shape[0]
    d3dPointsT1 = generate3DPoints(trackPoints1_KLT_L_3d, points_tracked_r1_3d, Proj_0, Proj_1)
    d3dPointsT2 = generate3DPoints(trackPoints2_KLT_L_3d, points_tracked_r2_3d, Proj_0, Proj_1)



    ransacError = float('inf')
    dOut = None
    # RANSAC
    ransacSize = 6
    for ransacItr in range(250):
        sampledPoints = np.random.randint(0, numPoints, ransacSize)
        rD2dPoints1_L = trackPoints1_KLT_L_3d[sampledPoints]
        rD2dPoints2_L = trackPoints2_KLT_L_3d[sampledPoints]
        rD3dPointsT1 = d3dPointsT1[sampledPoints]
        rD3dPointsT2 = d3dPointsT2[sampledPoints]

        dSeed = np.zeros(6)
        #minimizeReprojection(d, trackedPoints1_KLT_L, trackedPoints2_KLT_L, cliqued3dPointT1, cliqued3dPointT2, Proj1)
        optRes = least_squares(minimizeReprojection, dSeed, method='lm', max_nfev=200,
                            args=(rD2dPoints1_L, rD2dPoints2_L, rD3dPointsT1, rD3dPointsT2, Proj_0))

        #error = optRes.fun
        error = minimizeReprojection(optRes.x, trackPoints1_KLT_L_3d, trackPoints2_KLT_L_3d,
                                        d3dPointsT1, d3dPointsT2, Proj_0)

        eCoords = error.reshape((d3dPointsT1.shape[0]*2, 3))
        totalError = np.sum(np.linalg.norm(eCoords, axis=1))

        if (totalError < ransacError):
            ransacError = totalError
            dOut = optRes.x

        #clique size check
        # reproj error check
        # r, t generation
    Rmat = genEulerZXZMatrix(dOut[0], dOut[1], dOut[2])
    translationArray = np.array([[dOut[3]], [dOut[4]], [dOut[5]]])

    if (isinstance(translation, np.ndarray)):
        translation = translation + np.matmul(rotation, translationArray)
    else:
        translation = translationArray

    if (isinstance(rotation, np.ndarray)):
        rotation = np.matmul(Rmat, rotation)
    else:
        rotation = Rmat

    # Prepare the new iteration
    im_l1 = im_l2
    im_r1 = im_r2

    outMat = np.hstack((rotation, translation))
    #np.savetxt(fpPoseOut, outMat, fmt='%.6e', footer='\n')
    matList = outMat.tolist()
    #outtxt = ''
    for val in matList:
        for v in val:
            outtxt = outtxt + '{0:06e}'.format(v) + ' '

    outtxt = outtxt.rstrip()
    outtxt = outtxt + '\n'
    traj.append(translation)

    if iter > 150: # fix the points that are outside the frame
        break


## Projection matrix stuff



