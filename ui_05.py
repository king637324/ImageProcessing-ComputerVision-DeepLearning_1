# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw1_05_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(382, 453)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 341, 381))
        self.groupBox.setObjectName("groupBox")
        self.ButtonQ5_1 = QtWidgets.QPushButton(self.groupBox)
        self.ButtonQ5_1.setGeometry(QtCore.QRect(20, 30, 291, 41))
        self.ButtonQ5_1.setObjectName("ButtonQ5_1")
        self.ButtonQ5_2 = QtWidgets.QPushButton(self.groupBox)
        self.ButtonQ5_2.setGeometry(QtCore.QRect(20, 80, 291, 41))
        self.ButtonQ5_2.setObjectName("ButtonQ5_2")
        self.ButtonQ5_3 = QtWidgets.QPushButton(self.groupBox)
        self.ButtonQ5_3.setGeometry(QtCore.QRect(20, 130, 291, 41))
        self.ButtonQ5_3.setObjectName("ButtonQ5_3")
        self.ButtonQ5_4 = QtWidgets.QPushButton(self.groupBox)
        self.ButtonQ5_4.setGeometry(QtCore.QRect(20, 180, 291, 41))
        self.ButtonQ5_4.setObjectName("ButtonQ5_4")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 240, 301, 141))
        self.groupBox_2.setObjectName("groupBox_2")
        self.ButtonQ5_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.ButtonQ5_5.setGeometry(QtCore.QRect(10, 80, 281, 41))
        self.ButtonQ5_5.setObjectName("ButtonQ5_5")
        self.spinBox = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox.setGeometry(QtCore.QRect(10, 40, 281, 31))
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(999)
        self.spinBox.setObjectName("spinBox")
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 382, 25))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("mainWindow", "VGG16 TEST"))
        self.ButtonQ5_1.setText(_translate("mainWindow", "1. Show Train Images"))
        self.ButtonQ5_2.setText(_translate("mainWindow", "2. ShowHyperParameter"))
        self.ButtonQ5_3.setText(_translate("mainWindow", "3. Show Model Structure"))
        self.ButtonQ5_4.setText(_translate("mainWindow", "4. Show Accuracy and Loss"))
        self.groupBox_2.setTitle(_translate("mainWindow", "5. Test"))
        self.ButtonQ5_5.setText(_translate("mainWindow", "5. Test"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())