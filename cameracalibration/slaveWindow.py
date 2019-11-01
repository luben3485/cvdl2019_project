import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel
from PyQt5.QtGui import QPalette, QBrush, QPixmap
import time
 
 
class imageWindow(QWidget):
    
    def __init__(self,img_path):
        super().__init__()
        self.img_path = img_path
        self.initUI() 
    def initUI(self):
        pix = QPixmap(self.img_path)
        lb1 = QLabel(self)
        lb1.setGeometry(0,0,600,600)
        #lb1.setStyleSheet("border: 2px solid red")
        lb1.setPixmap(pix)
        #設定視窗的位置和大小
        self.setGeometry(300, 300, 600, 600)  
        #設定視窗的標題
        self.setWindowTitle('View')
        
if __name__ == '__main__':
    #建立應用程式和物件
    app = QApplication(sys.argv)
    image_window = imageWindow('view.jpg')
    image_window.show()
    sys.exit(app.exec_()) 