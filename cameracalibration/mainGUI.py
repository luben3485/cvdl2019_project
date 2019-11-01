
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from mainWindow import Ui_MainWindow
import cv2
import numpy as np
import glob
import camera

class mainGUI(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self,parent=None):
        super(mainGUI, self).__init__(parent=parent)
        self.setupUi(self)

        # push button
        self.corners.clicked.connect(self.findCorners)
        self.intrinsic.clicked.connect(self.findIntrinsic)
        self.distortion.clicked.connect(self.findDistortion)
        self.extrinsic.clicked.connect(self.findExtrinsic)
        self.augmentedreality.clicked.connect(camera.augmentedReality)
        self.findcontour.clicked.connect(self.contour)
        self.perspective.clicked.connect(self.perspectiveTransform)
        self.rotation.clicked.connect(self.rotation_scaling_translation)
        # variable
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.w = 11
        self.h = 8
        self.objpoints =  []
        self.imgpoints = []
        self.shape = (0,0)
    def perspectiveTransform(self):
        pass

    def rotation_scaling_translation(self):
        angle = self.angle.toPlainText()
        scale = self.scale.toPlainText()
        tx = self.textEdit_3.toPlainText()
        ty = self.textEdit_4.toPlainText()
        
        img = cv2.imread('images/OriginalTransform.png')

        center = (130,125)
        M = cv2.getRotationMatrix2D(center, int(angle), scale=float(scale))
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        M = np.float32([[1, 0, int(tx)], [0, 1, int(ty)]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        cv2.imshow('Orinfinal Transform',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def findCorners(self):
        objp = np.zeros((self.w*self.h,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.w,0:self.h].T.reshape(-1,2)
        objpoints = []
        imgpoints = []
        images = glob.glob('images/CameraCalibration/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            self.shape = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (self.w,self.h),None)
            if ret == True:
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
            
                cv2.drawChessboardCorners(img, (self.w,self.h), corners, ret)
                cv2.namedWindow("Corners",0);
                cv2.resizeWindow("Corners", 1000, 1000);
                cv2.imshow('Corners',img)
                cv2.waitKey(300)
        cv2.destroyAllWindows()

    def findIntrinsic(self):
        if len(self.objpoints)==0 or len(self.imgpoints)==0:
            self.showDialog('Please press \"Find Corners\" button first')
        else:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.shape, None, None)
            print(mtx)

    def findDistortion(self):
        if len(self.objpoints)==0 or len(self.imgpoints)==0:
            self.showDialog('Please press \"Find Corners\" button first')
        else:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.shape, None, None)
            print(dist)
        
    def findExtrinsic(self):
        if len(self.objpoints)==0 or len(self.imgpoints)==0:
            self.showDialog('Please press \"Find Corners\" button first')
        else:
            index = self.comboBox.currentIndex()
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.shape, None, None)
            Rs = []
            for i in range(len(rvecs)):
                R,jacobian= cv2.Rodrigues(rvecs[i])
                R_concat = np.concatenate((R,tvecs[i]),axis = 1)
                Rs.append(R_concat)
            print(Rs[index])
    def contour(self):
        img = cv2.imread('images/Contour.png')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
        ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
        binary, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        cv2.drawContours(img,contours,-1,(0,0,255),3)  

        cv2.imshow('Shapes',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def showDialog(self,msg):
        dlg = QtWidgets.QDialog(self)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(16)
        
        label = QtWidgets.QLabel(dlg)
        label.setFont(font)
        label.setGeometry(QtCore.QRect(0, 0, 250,250))
        label.setObjectName("label")
        label.setText(msg)
        dlg.setWindowTitle("Messenge!")
        dlg.resize(250,250)
        dlg.exec_()


if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    main_gui = mainGUI()
    main_gui.show()
    sys.exit(app.exec_()) 