#coding:utf-8
import cv2
import numpy as np
import glob


def augmentedReality():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w = 11
    h = 8

    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    objpoints = []
    imgpoints = []

    #images = glob.glob('images/CameraCalibration/*.bmp')
    image_path = 'images/CameraCalibration/'
    for i in range(1,6):
        img = cv2.imread(image_path + str(i)+'.bmp')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
        
        if ret == True:
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            '''
            cv2.drawChessboardCorners(img, (w,h), corners, ret)
            cv2.namedWindow("Corners",0);
            cv2.resizeWindow("Corners", 1000, 1000);
            cv2.imshow('Corners',img)
            cv2.waitKey(500)
            '''
    #cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    point_size = 2
    point_color = (0, 0, 255) # BGR
    thickness = 8
    #for i,fname in enumerate(images):
    for i in range(0,5):
        img_ = cv2.imread(image_path + str(i+1)+'.bmp')
        obj_test = np.array([[0,0,-2],[1,1,0],[1,-1,0],[-1,-1,0],[-1,1,0]],np.float32)
        imgpoint, _ = cv2.projectPoints(obj_test, rvecs[i], tvecs[i], mtx, dist)
        cv2.line(img_, tuple(imgpoint[0][0]), tuple(imgpoint[1][0]), point_color, thickness)
        cv2.line(img_, tuple(imgpoint[0][0]), tuple(imgpoint[2][0]), point_color, thickness)
        cv2.line(img_, tuple(imgpoint[0][0]), tuple(imgpoint[3][0]), point_color, thickness)
        cv2.line(img_, tuple(imgpoint[0][0]), tuple(imgpoint[4][0]), point_color, thickness)
        cv2.line(img_, tuple(imgpoint[1][0]), tuple(imgpoint[2][0]), point_color, thickness)
        cv2.line(img_, tuple(imgpoint[2][0]), tuple(imgpoint[3][0]), point_color, thickness)
        cv2.line(img_, tuple(imgpoint[3][0]), tuple(imgpoint[4][0]), point_color, thickness)
        cv2.line(img_, tuple(imgpoint[4][0]), tuple(imgpoint[1][0]), point_color, thickness)
        cv2.circle(img_,tuple(imgpoint[1][0]), point_size, point_color, thickness)
        cv2.circle(img_,tuple(imgpoint[2][0]), point_size, point_color, thickness)
        cv2.circle(img_,tuple(imgpoint[3][0]), point_size, point_color, thickness)
        cv2.circle(img_,tuple(imgpoint[4][0]), point_size, point_color, thickness)

        cv2.namedWindow("AR",0);
        cv2.resizeWindow("AR", 1000, 1000);
        cv2.imshow('AR',img_)
        cv2.waitKey(500)
    cv2.destroyAllWindows()






    ### Intrinsic matrix
    #print(mtx)

    ###Distortion matrix
    #print(dist)

    ### Extrinsic matrix
    '''
    Rs = []
    for i in range(len(rvecs)):
        R,jacobian= cv2.Rodrigues(rvecs[i])
        R_concat = np.concatenate((R,tvecs[i]),axis = 1)
        Rs.append(R_concat)
    print(Rs[0])

    img2 = cv2.imread('images\\CameraCalibration\\1.bmp')
    h,  w = img2.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
    #x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]
    cv2.imwrite('cameraResult.png',dst)
    '''

    ###Error
    '''
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    print("total error: ", total_error/len(objpoints))
    '''








