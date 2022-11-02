from sys import int_info
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import hw1_ui as ui

import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication
import numpy as np

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        ####################
        #####    Q1    #####
        #################### 
        self.ButtonQ1_1.clicked.connect(self.Q1_1)
        self.ButtonQ1_2.clicked.connect(self.Q1_2)
        self.ButtonQ1_3.clicked.connect(self.Q1_3)
        self.ButtonQ1_4.clicked.connect(self.Q1_4)

        ####################
        #####    Q2    #####
        #################### 
        self.ButtonQ2_1.clicked.connect(self.Q2_1)
        self.ButtonQ2_2.clicked.connect(self.Q2_2)
        self.ButtonQ2_3.clicked.connect(self.Q2_3)

        ####################
        #####    Q3    #####
        #################### 
        self.ButtonQ3_1.clicked.connect(self.Q3_1)
        self.ButtonQ3_2.clicked.connect(self.Q3_2)
        self.ButtonQ3_3.clicked.connect(self.Q3_3)
        self.ButtonQ3_4.clicked.connect(self.Q3_4)
        self.picture_gaussian = None


        ####################
        #####    Q4    #####
        #################### 
        self.ButtonQ4_1.clicked.connect(self.Q4_1)
        self.ButtonQ4_2.clicked.connect(self.Q4_2)
        self.ButtonQ4_3.clicked.connect(self.Q4_3)
        self.ButtonQ4_4.clicked.connect(self.Q4_4)


    def Q1_1(self):
        print("-------------Q1_1-------------")
        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')

        if picture is None:
            print('no folder /Dataset_OpenCvDl_Hw1')
            return

        cv2.imshow('Sun.jpg', picture)
        # picture.shape 可以檢查一下這個 NumPy 陣列的大小
        print('Height = ', picture.shape[0])
        print('Width = ', picture.shape[1])

        print("-------------Q1_1 Finsh-------------\n")

    def Q1_2(self):
        print("-------------Q1_2-------------")

        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')

        if picture is None:
            print('no folder /Dataset_OpenCvDl_Hw1')
            return
        
        # uint8專門用於存儲各種圖像的（包括RGB，灰階圖片等）
        picture_B = np.zeros(picture.shape, dtype=np.uint8)
        picture_G = np.zeros(picture.shape, dtype=np.uint8)
        picture_R = np.zeros(picture.shape, dtype=np.uint8)

        # picture[mode, size, color]
        # OpenCV 讀取進來的圖片會以 BGR 的方式儲存三個顏色的 channel
        picture_B[:, :, 0] = picture[:, :, 0]
        picture_G[:, :, 1] = picture[:, :, 1]
        picture_R[:, :, 2] = picture[:, :, 2]

        cv2.imshow('Sun.jpg', picture)
        cv2.imshow('pictur B', picture_B)
        cv2.imshow('pictur G', picture_G)
        cv2.imshow('pictur R', picture_R)

        print("-------------Q1_2 Finsh-------------\n")

    def Q1_3(self):
        print("-------------Q1_3-------------")
        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')

        if picture is None:
            print('no folder /Dataset_OpenCvDl_Hw1')
            return

        picture_gary = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg',0)  #1彩色 0灰階

        if picture_gary is None:
            print('no folder /Dataset_OpenCvDl_Hw1')
            return

        cv2.imshow('I1', picture_gary)

        # 將圖片切分出 BGR三個顏色的 channel
        B, G, R = cv2.split(picture)
        Gray_picture = (0.333*B+0.333*G+0.333*R)
        cv2.imshow('I2', Gray_picture/255)

        print("-------------Q1_3 Finsh-------------\n")

    def Q1_4(self):
        print("-------------Q1_4-------------")

        picture_Strong = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg')
        picture_Weak = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg')

        if picture_Strong is None:
            print('no folder /Dataset_OpenCvDl_Hw1')
            return
        
        if picture_Weak is None:
            print('no folder /Dataset_OpenCvDl_Hw1')
            return

        # cv2.namedWindow('視窗名稱',0) 0表示視窗大小可以改變
        cv2.namedWindow('BLENDING',0)

        # cv2.addWeighted() 是將兩張相同大小，相同類型的圖片融合,可以實現圖片的特效
        # cv2.addWeighted(圖片1,圖片1的權重(alpha),圖片2,圖片2的權重(beta),(gamma)圖片1與圖片2合成後的數值 255表示白色)参数1：src1，第一个原数组.
        dst = cv2.addWeighted(picture_Weak, 0, picture_Strong, 1, 0.0)
        cv2.imshow('BLENDING', dst)
        def blending(x):
            beta = (1.0 - x / 255)
            dst = cv2.addWeighted(picture_Weak, 1 - beta, picture_Strong, beta, 0.0)
            cv2.imshow('BLENDING', dst)

        # 滑動條
        # cv2.createTrackbar(滑動條名稱, 視窗名稱, 0 (滑動條預設值), 滑動條最大值, (回條參數呼叫自己)self.opencv_calibration_node.on_scale)
        cv2.createTrackbar('BLENDING', 'BLENDING', 0, 255, blending)

        print("-------------Q1_4 Finsh-------------\n")

    def Q2_1(self):
        print("-------------Q2_1-------------")

        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg')

        if picture is None:
            print('no folder Dataset_OpenCvDl_Hw1')
            return

        # # cv2.namedWindow('視窗名稱',0) 0表示視窗大小可以改變
        # cv2.namedWindow('Original Image',0)
        # cv2.namedWindow('Gaussian Blur',0)

        cv2.imshow('Original Image', picture)

        # 高斯平滑 用作消除其他圖像處理中的雜訊點
        picture_blur = cv2.GaussianBlur(picture, (3, 3), 0)
        cv2.imshow('Gaussian Blur', picture_blur)

        print("-------------Q2_1 Finsh-------------\n")
    
    def Q2_2(self):
        print("-------------Q2_2-------------")

        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg')

        if picture is None:
            print('no folder Dataset_OpenCvDl_Hw1')
            return
        
        # # cv2.namedWindow('視窗名稱',0) 0表示視窗大小可以改變
        # cv2.namedWindow('Original Image',0)
        # cv2.namedWindow('Bilateral Filter',0)

        cv2.imshow('Original Image', picture)

        # Bilateral Filter(雙邊濾波器)
        # cv2.bilateralFilter(picture,d,p1,p2)  picture是圖片  d是區域的直徑  p1&p2是空間高斯函數標準差和灰階值相似性高斯函數標準差
        picture_blur = cv2.bilateralFilter(picture, 9, 90, 90)
        cv2.imshow('Bilateral Filter', picture_blur)

        print("-------------Q2_2 Finsh-------------\n")

    def Q2_3(self):
        print("-------------Q2_3-------------")

        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_pepperSalt.jpg')

        if picture is None:
            print('no folder Dataset_OpenCvDl_Hw1')
            return
        
        # # cv2.namedWindow('視窗名稱',0) 0表示視窗大小可以改變
        # cv2.namedWindow('Original Image',0)
        # cv2.namedWindow('3x3_Median Filter',0)
        # cv2.namedWindow('5x5_Median Filter',0)

        cv2.imshow('Original Image', picture)

        # medianBlur 中值濾波器是用卷積框中像素的中值代替中心值，達到去雜訊
        # cv.medianBlur(圖片, 濾波視窗大小) 
        picture_blur_3 = cv2.medianBlur(picture, 3)
        cv2.imshow('3x3_Median Filter', picture_blur_3)

        picture_blur_5 = cv2.medianBlur(picture, 5)
        cv2.imshow('5x5_Median Filter', picture_blur_5)


        print("-------------Q2_3 Finsh-------------\n")

    def Convolution2D(self, p_gaussian, kernel, padding = 1):

        # 宣告一個p_gaussian大小的0
        out = np.zeros(p_gaussian.shape)
        
        p_padding = np.zeros((p_gaussian.shape[0] + padding * 2, p_gaussian.shape[1] + padding * 2))

        
        p_padding[padding:-1*padding, padding:-1*padding] = p_gaussian

        for y in range(p_gaussian.shape[1]):
            for x in range(p_gaussian.shape[0]):
                out[x, y] = (kernel * p_padding[x: x + kernel.shape[0], y: y + kernel.shape[1]]).sum()
        return out
    
    def Q3_1(self, flag=0):
        print("-------------Q3_1-------------")

        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg')

        if picture is None:
            print('no folder Dataset_OpenCvDl_Hw1')
            return
        
        # picture[mode, size, color]
        # OpenCV 讀取進來的圖片會以 BGR 的方式儲存三個顏色的 channel
        # B = 0, G = 1, R = 2
        # 自己將彩色圖片轉成灰階 公式 I1 = 0.07*B + 0.72*G + 0.21*R
        picture_gray = 0.07 * picture[:, :, 0] + 0.72 * picture[:, :, 1] + 0.21 * picture[:, :, 2]

        # .astype() 為資料型別轉換方法，可以將元素的型別變成另外的一種
        # uint8專門用於存儲各種圖像的（包括RGB，灰階圖片等）
        picture_gray = picture_gray.astype('uint8')

        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian = np.exp(- (x ** 2 + y ** 2))
        # 正規化
        kernel = gaussian / gaussian.sum()

        self.picture_gaussian = self.Convolution2D(picture_gray, kernel)

        # flag = 0 表示是Q3_1要output的ans
        if flag == 0:
            cv2.imshow('Original Image', picture)
            cv2.imshow('Grayscale Image', picture_gray)
            cv2.imshow('My Gaussian Blur', self.picture_gaussian.astype('uint8'))
        
        # 此題找到的參考資料
        # https://medium.com/@bob800530/python-gaussian-filter-%E6%A6%82%E5%BF%B5%E8%88%87%E5%AF%A6%E4%BD%9C-676aac52ea17


        print("-------------Q3_1 Finsh-------------\n")

    def Q3_2(self):
        print("-------------Q3_2-------------")

        if self.picture_gaussian is None:
            self.Q3_1(flag=1)

        # Sobel Edge Detection 針對 X軸的垂直邊緣偵測
        kx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        x = self.Convolution2D(self.picture_gaussian, kx)
        # .astype() 為資料型別轉換方法，可以將元素的型別變成另外的一種
        # uint8專門用於存儲各種圖像的（包括RGB，灰階圖片等）
        cv2.imshow('Sobel X', abs(x).astype('uint8'))

        print("-------------Q3_2 Finsh-------------\n")

    def Q3_3(self):
        print("-------------Q3_3-------------")

        if self.picture_gaussian is None:
            self.Q3_1(flag=1)

        # Sobel Edge Detection 針對 y軸的水平邊緣偵測
        ky = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])

        y = self.Convolution2D(self.picture_gaussian, ky)
        # .astype() 為資料型別轉換方法，可以將元素的型別變成另外的一種
        # uint8專門用於存儲各種圖像的（包括RGB，灰階圖片等）
        cv2.imshow('Sobel Y', abs(y).astype('uint8'))

        print("-------------Q3_3 Finsh-------------\n")

    def Q3_4(self):
        print("-------------Q3_4-------------")

        if self.picture_gaussian is None:
            self.Q3_1(flag=1)

        # Sobel Edge Detection 針對 X軸的垂直邊緣偵測
        kx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        # Sobel Edge Detection 針對 y軸的水平邊緣偵測
        ky = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])
        x = self.Convolution2D(self.picture_gaussian, kx)
        y = self.Convolution2D(self.picture_gaussian, ky)

        # magnitude = cv2.magnitude(x, y)
        # magnitude的數學公式 根號(x^2 + y^2)
        magnitude = np.sqrt(x * x + y * y)

        # .astype() 為資料型別轉換方法，可以將元素的型別變成另外的一種
        # uint8專門用於存儲各種圖像的（包括RGB，灰階圖片等）
        cv2.imshow('Magnitude', abs(magnitude).astype('uint8'))

        print("-------------Q3_4 Finsh-------------\n")
    
    def Q4_1(self):
        print("-------------Q4_1-------------")

        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')

        if picture is None:
            print('no folder Dataset_OpenCvDl_Hw1')
            return
        
        pic = cv2.resize(picture, (256, 256))

        cv2.imshow('Original Image', picture)
        cv2.imshow('Resized image', pic)


        print("-------------Q4_1 Finsh-------------\n")
    
    def Q4_2(self):
        print("-------------Q4_2-------------")

        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')

        if picture is None:
            print('no folder Dataset_OpenCvDl_Hw1')
            return
        
        picture = cv2.resize(picture, (256, 256))

        (h, w, d) = picture.shape # 讀取圖片大小
        # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
        M = cv2.getRotationMatrix2D((128,128), 0, 1.0)
        tx = 0
        ty = 60
        M[0, 2] += tx
        M[1, 2] += ty
        # # 第三個參數變化後的圖片大小
        trans_pic = cv2.warpAffine(picture, M,  (w+128, h+60))
        
        cv2.imshow('After translate', trans_pic)


        print("-------------Q4_2 Finsh-------------\n")

    def Q4_3(self):
        print("-------------Q4_3-------------")

        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')

        if picture is None:
            print('no folder Dataset_OpenCvDl_Hw1')
            return
        
        picture = cv2.resize(picture, (256, 256))

        cv2.namedWindow("After rotate and scale", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("After rotate and scale", 400, 300) 

        (h, w, d) = picture.shape # 讀取圖片大小
        
        # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
        M = cv2.getRotationMatrix2D((128,128), 10, 0.5)

        tx = 0
        ty = 60
        M[0, 2] += tx
        M[1, 2] += ty
        
        # 第三個參數變化後的圖片大小
        rotate_pic = cv2.warpAffine(picture, M, (w+128, h+60))
        
        cv2.imshow('After rotate and scale', rotate_pic)

        # 此題找到的參考資料
        # https://ithelp.ithome.com.tw/articles/10236235

        print("-------------Q4_3 Finsh-------------\n")

    def Q4_4(self):
        print("-------------Q4_4-------------")

        picture = cv2.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')

        if picture is None:
            print('no folder Dataset_OpenCvDl_Hw1')
            return
        
        picture = cv2.resize(picture, (256, 256))

        # cv2.namedWindow("After rotate and scale", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("After rotate and scale", 400, 300) 

        (h, w, d) = picture.shape # 讀取圖片大小
        
        # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
        M = cv2.getRotationMatrix2D((128,128), 10, 0.5)

        tx = 0
        ty = 60
        M[0, 2] += tx
        M[1, 2] += ty
        
        # 第三個參數變化後的圖片大小
        rotate_pic = cv2.warpAffine(picture, M, (w+128, h+60))

        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        M = cv2.getAffineTransform(pts1,pts2)
        crop_pic = cv2.warpAffine(rotate_pic, M, (w+128, h+60))

        cv2.imshow('After Shearing', crop_pic)

        # 此題找到的參考資料
        # https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html

        print("-------------Q4_4 Finsh-------------\n")
    

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())