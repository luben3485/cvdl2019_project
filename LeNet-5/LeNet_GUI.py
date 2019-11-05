from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from LeNet_UIDesign import Ui_MainWindow
import LeNet_utils

class LeNet_GUI(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self,parent=None):
        super(LeNet_GUI, self).__init__(parent=parent)
        self.setupUi(self)

        # push button
        self.pushButton.clicked.connect(LeNet_utils.randomShowPics)
        self.pushButton_2.clicked.connect(LeNet_utils.printHyperparameter)
        self.pushButton_3.clicked.connect(LeNet_utils.train_one_epoch_gui)
        self.pushButton_4.clicked.connect(LeNet_utils.showTrainingResult)
        self.pushButton_5.clicked.connect(self.inference)
    
    def inference(self):
        index = int(self.lineEdit.text())
        LeNet_utils.predict(index)

if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    lenet_gui = LeNet_GUI()
    lenet_gui.show()
    sys.exit(app.exec_()) 