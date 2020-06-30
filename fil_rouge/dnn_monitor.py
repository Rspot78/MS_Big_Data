# imports for GUI
import sys
from tkinter import Tk
import time
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit, QFrame, \
QHBoxLayout, QCheckBox, QRadioButton, QButtonGroup, QStyle, QSlider, QStackedLayout, QComboBox
from PyQt5.QtCore import pyqtSlot, QRect, Qt, QRunnable, QThreadPool, QThread, QObject, QUrl, QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QImage, QPalette, QBrush, QIcon, QPixmap, QFont, QColor
from PyQt5.QtCore import pyqtSignal

# imports for Tensorflow
import tensorflow as tf
from tensorflow import keras, GradientTape
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History, LambdaCallback
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.backend import eval as tfeval

# data visualisation module
from data_visualisation import *

# module for tensorflow callbacks
from tensorflow_apps import *

# class DnnViewer: create and manage dashbord
class DnnMonitor(QWidget):             
    def __init__(self, path):
        super(DnnMonitor, self).__init__()
        self.path = path
        self.init_params()
        self.init_gui()
        self.mediaPlayer1.play()
        self.mediaPlayer1.pause()
        self.mediaPlayer2.play()
        self.mediaPlayer2.pause()
        self.mediaPlayer3.play()
        self.mediaPlayer3.pause()
        
    def init_params(self):
        self.parameters = {}
        self.parameters["epochs"] = 100
        self.parameters["convolution"] = 4 
        self.parameters["dense"] = 1   
        self.parameters["pooling"] = True 
        self.parameters["dropout"] = True  
        self.parameters["frequency"] = 50  
        self.parameters["window"] = 10  
        self.parameters["filters"] = [32, 32, 64, 64] 
        self.parameters["dimensions"] = [512]
        self.parameters["pooling_frequency"] = 2  
        self.parameters["dropout_frequency"] = 2
        self.parameters["visualise_conv"] = True 
        self.parameters["kernel_size"] = 5 
        self.parameters["pooling_size"] = 2
        self.parameters["dropout_rate"] = 0.25 
        self.parameters["visualise_layer"] = 1
        self.parameters["visualise_from"] = 1
        
    def init_gui(self):
        self.setFixedSize(1810, 990)
        root = Tk()
        screen_width = root.winfo_screenwidth()                            
        screen_height = root.winfo_screenheight()                         
        width, heigth = 1810, 990 
        self.setFixedSize(width, heigth)
        left, top = (screen_width - width) / 2, (screen_height - heigth) / 2                                    
        self.move(left, min(10,top)) 
        self.setWindowTitle("Deep Neural Network Monitor")                  
        self.setStyleSheet("background: white");
        
        self.mainTitle = QLabel(self)                                                                               
        self.mainTitle.setText("Deep Neural Network Monitor")                    
        self.mainTitle.move(80,5)                                                  
        self.mainTitle.setStyleSheet("font-size: 36px; font-family:  FreeSans; \
        font-weight: bold")
        
        self.binocularIcon = QLabel(self) 
        self.binocularIcon.setPixmap(QtGui.QPixmap(self.path + '/dash/binocular.png').scaled(40, 40))
        self.binocularIcon.move(30, 5)  
        
        # top left quadrant        
        self.topLeftFrame1 = QFrame(self)                                                   
        self.topLeftFrame1.move(30, 50)
        self.topLeftFrame1.resize(850, 3)
        self.topLeftFrame1.setFrameShape(QFrame.HLine); 
        self.topLeftFrame1.setLineWidth(2)
        
        self.topLeftFrame2 = QFrame(self)                                                   
        self.topLeftFrame2.move(30, 480)
        self.topLeftFrame2.resize(850, 3)
        self.topLeftFrame2.setFrameShape(QFrame.HLine); 
        self.topLeftFrame2.setLineWidth(2)
        
        self.topLeftFrame3 = QFrame(self)                                                   
        self.topLeftFrame3.move(30, 50)
        self.topLeftFrame3.resize(3, 430)
        self.topLeftFrame3.setFrameShape(QFrame.VLine); 
        self.topLeftFrame3.setLineWidth(2)
        
        self.topLeftFrame4 = QFrame(self)                                                   
        self.topLeftFrame4.move(880, 50)
        self.topLeftFrame4.resize(3, 430)
        self.topLeftFrame4.setFrameShape(QFrame.VLine); 
        self.topLeftFrame4.setLineWidth(2)
        
        self.architectureLabel = QLabel(self)    
        self.architectureLabel.setText("Architecture")                                
        self.architectureLabel.move(50, 70)                               
        self.architectureLabel.setStyleSheet("font-size: 24px; font-family: \
        FreeSans; font-weight: bold")

        self.epochLabel = QLabel(self)                                       
        self.epochLabel.setText("epochs")           
        self.epochLabel.move(50, 115)           
        self.epochLabel.setStyleSheet("font-size: 22px; font-family: FreeSans") 
        
        self.convolutionLabel = QLabel(self)                      
        self.convolutionLabel.setText("convolution layers")     
        self.convolutionLabel.move(50, 160)         
        self.convolutionLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")                  

        self.denseLabel = QLabel(self)
        self.denseLabel.setText("dense layers") 
        self.denseLabel.move(50, 205)  
        self.denseLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")         
        
        self.poolingLabel = QLabel(self)      
        self.poolingLabel.setText("max pooling") 
        self.poolingLabel.move(50, 250) 
        self.poolingLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")        
        
        self.dropoutLabel = QLabel(self)    
        self.dropoutLabel.setText("dropout")  
        self.dropoutLabel.move(50, 295)   
        self.dropoutLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")       
        
        self.visualisationLabel = QLabel(self)  
        self.visualisationLabel.setText("Visualisation")
        self.visualisationLabel.move(50, 340)      
        self.visualisationLabel.setStyleSheet("font-size: 24px; font-family: \
        FreeSans; font-weight: bold")
        
        self.frequencyLabel = QLabel(self) 
        self.frequencyLabel.setText("creation frequency")    
        self.frequencyLabel.move(50, 385)  
        self.frequencyLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")         
        
        self.windowLabel = QLabel(self)  
        self.windowLabel.setText("window size")  
        self.windowLabel.move(50, 430) 
        self.windowLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")    
        
        self.epochTextBox = QLineEdit(self) 
        self.epochTextBox.move(260,111)  
        self.epochTextBox.resize(60,30)          
        self.epochTextBox.setFont(QtGui.QFont("FreeSans", 12))  
        self.epochTextBox.setAlignment(Qt.AlignCenter)  
        self.epochTextBox.setText(str(self.parameters.get("epochs")))   
        self.epochTextBox.textChanged.connect(self.set_epoch_text) 
        
        self.convolutionTextBox = QLineEdit(self) 
        self.convolutionTextBox.move(260,156)       
        self.convolutionTextBox.resize(60,30)                                         
        self.convolutionTextBox.setFont(QtGui.QFont("FreeSans", 12))   
        self.convolutionTextBox.setAlignment(Qt.AlignCenter)    
        self.convolutionTextBox.setText(str(self.parameters.get("convolution")))  
        self.convolutionTextBox.textChanged.connect(self.set_convolution_text)
        
        self.denseTextBox = QLineEdit(self) 
        self.denseTextBox.move(260,201) 
        self.denseTextBox.resize(60,30) 
        self.denseTextBox.setFont(QtGui.QFont("FreeSans", 12)) 
        self.denseTextBox.setAlignment(Qt.AlignCenter) 
        self.denseTextBox.setText(str(self.parameters.get("dense")))
        self.denseTextBox.textChanged.connect(self.set_dense_text)   
        
        self.poolingCheckBox = QCheckBox(self)
        self.poolingCheckBox.move(295,250)  
        self.poolingCheckBox.setStyleSheet("QCheckBox::indicator {width: 25px; height: 25px}") 
        self.poolingCheckBox.setChecked(self.parameters.get("pooling"))  
        self.poolingCheckBox.stateChanged.connect(self.check_pooling)   
        
        self.dropoutCheckBox = QCheckBox(self)        
        self.dropoutCheckBox.move(295,295)  
        self.dropoutCheckBox.setStyleSheet("QCheckBox::indicator {width: 25px; height: 25px}") 
        self.dropoutCheckBox.setChecked(self.parameters.get("dropout"))
        self.dropoutCheckBox.stateChanged.connect(self.check_dropout)
        
        self.frequencyTextBox = QLineEdit(self)
        self.frequencyTextBox.move(260,384) 
        self.frequencyTextBox.resize(60,30)
        self.frequencyTextBox.setFont(QtGui.QFont("FreeSans", 12)) 
        self.frequencyTextBox.setAlignment(Qt.AlignCenter)  
        self.frequencyTextBox.setText(str(self.parameters.get("frequency")))
        self.frequencyTextBox.textChanged.connect(self.set_frequency_text)
        
        self.windowTextBox = QLineEdit(self) 
        self.windowTextBox.move(260,429)
        self.windowTextBox.resize(60,30) 
        self.windowTextBox.setFont(QtGui.QFont("FreeSans", 12)) 
        self.windowTextBox.setAlignment(Qt.AlignCenter) 
        self.windowTextBox.setText(str(self.parameters.get("window")))
        self.windowTextBox.textChanged.connect(self.set_window_text) 

        self.filterLabel = QLabel(self)  
        self.filterLabel.setText("filters") 
        self.filterLabel.move(400, 160)         
        self.filterLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")       
        
        self.dimensionLabel = QLabel(self) 
        self.dimensionLabel.setText("dimensions") 
        self.dimensionLabel.move(400, 205) 
        self.dimensionLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")   

        self.poolingFrequencyLabel = QLabel(self)   
        self.poolingFrequencyLabel.setText("frequency")  
        self.poolingFrequencyLabel.move(400, 250) 
        self.poolingFrequencyLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")
        
        self.dropoutFrequencyLabel = QLabel(self) 
        self.dropoutFrequencyLabel.setText("frequency") 
        self.dropoutFrequencyLabel.move(400, 295)
        self.dropoutFrequencyLabel.setStyleSheet("font-size: 22px; font-family: FreeSans") 
         
        self.radioConvolutionLabel = QLabel(self)  
        self.radioConvolutionLabel.setText("convolution")   
        self.radioConvolutionLabel.move(400, 385)   
        self.radioConvolutionLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")  

        self.radioDenseLabel = QLabel(self)  
        self.radioDenseLabel.setText("dense")  
        self.radioDenseLabel.move(400, 430)  
        self.radioDenseLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")  

        self.filterTextBox = QLineEdit(self)     
        self.filterTextBox.move(530,156)   
        self.filterTextBox.resize(60,30)   
        self.filterTextBox.setFont(QtGui.QFont("FreeSans", 12)) 
        self.filterTextBox.setAlignment(Qt.AlignCenter)     
        self.filterTextBox.setText(str(self.parameters["filters"])[1:-1])    
        self.filterTextBox.textChanged.connect(self.set_filter_text)         
       
        self.dimensionTextBox = QLineEdit(self)
        self.dimensionTextBox.move(530,201)          
        self.dimensionTextBox.resize(60,30)           
        self.dimensionTextBox.setFont(QtGui.QFont("FreeSans", 12))    
        self.dimensionTextBox.setAlignment(Qt.AlignCenter)      
        self.dimensionTextBox.setText(str(self.parameters["dimensions"])[1:-1])
        self.dimensionTextBox.textChanged.connect(self.set_dimension_text)    
        
        self.poolingFrequencyTextBox = QLineEdit(self) 
        self.poolingFrequencyTextBox.move(530,246) 
        self.poolingFrequencyTextBox.resize(60,30)     
        self.poolingFrequencyTextBox.setFont(QtGui.QFont("FreeSans", 12)) 
        self.poolingFrequencyTextBox.setAlignment(Qt.AlignCenter)  
        self.poolingFrequencyTextBox.setText(str(self.parameters.get("pooling_frequency"))) 
        self.poolingFrequencyTextBox.textChanged.connect(self.set_pooling_frequency_text)       
        
        self.dropoutFrequencyTextBox = QLineEdit(self) 
        self.dropoutFrequencyTextBox.move(530,291)  
        self.dropoutFrequencyTextBox.resize(60,30) 
        self.dropoutFrequencyTextBox.setFont(QtGui.QFont("FreeSans", 12))
        self.dropoutFrequencyTextBox.setAlignment(Qt.AlignCenter) 
        self.dropoutFrequencyTextBox.setText(str(self.parameters.get("dropout_frequency")))
        self.dropoutFrequencyTextBox.textChanged.connect(self.set_dropout_frequency_text)
              
        self.convolutionRadioButton = QRadioButton(self)  
        self.convolutionRadioButton.move(565,385)          
        self.convolutionRadioButton.setChecked(self.parameters.get("visualise_conv"))  
        self.convolutionRadioButton.setStyleSheet("QRadioButton::indicator {width: 25px; height: 25px}")  
        self.convolutionRadioButton.toggled.connect(self.select_convolution_dense)        
              
        self.denseRadioButton = QRadioButton(self)   
        self.denseRadioButton.move(565,430)        
        self.denseRadioButton.setChecked(not self.parameters.get("visualise_conv")) 
        self.denseRadioButton.setStyleSheet("QRadioButton::indicator {width: 25px; height: 25px}")    
        self.denseRadioButton.toggled.connect(self.select_convolution_dense) 
        
        self.convolutionButtonGroup = QButtonGroup(self)  
        self.convolutionButtonGroup.addButton(self.convolutionRadioButton) 
        self.convolutionButtonGroup.addButton(self.denseRadioButton)       
        
        self.kernelLabel = QLabel(self) 
        self.kernelLabel.setText("kernel size")
        self.kernelLabel.move(660, 160) 
        self.kernelLabel.setStyleSheet("font-size: 22px; font-family: FreeSans") 
        
        self.poolingSizeLabel = QLabel(self) 
        self.poolingSizeLabel.setText("pooling size") 
        self.poolingSizeLabel.move(660, 250) 
        self.poolingSizeLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")         

        self.dropoutRateLabel = QLabel(self) 
        self.dropoutRateLabel.setText("dropout rate")  
        self.dropoutRateLabel.move(660, 295)   
        self.dropoutRateLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")          
        
        self.layerLabel = QLabel(self) 
        self.layerLabel.setText("layer") 
        self.layerLabel.move(660, 385)
        self.layerLabel.setStyleSheet("font-size: 22px; font-family: FreeSans") 

        self.visualiseLabel = QLabel(self) 
        self.visualiseLabel.setText("visualise from") 
        self.visualiseLabel.move(660, 430)
        self.visualiseLabel.setStyleSheet("font-size: 22px; font-family: FreeSans")

        self.kernelTextBox = QLineEdit(self)
        self.kernelTextBox.move(800,156)
        self.kernelTextBox.resize(60,30) 
        self.kernelTextBox.setFont(QtGui.QFont("FreeSans", 12))  
        self.kernelTextBox.setAlignment(Qt.AlignCenter) 
        self.kernelTextBox.setText(str(self.parameters.get("kernel_size")))
        self.kernelTextBox.textChanged.connect(self.set_kernel_text) 
        
        self.poolingSizeTextBox = QLineEdit(self)
        self.poolingSizeTextBox.move(800,246)  
        self.poolingSizeTextBox.resize(60,30)  
        self.poolingSizeTextBox.setFont(QtGui.QFont("FreeSans", 12))  
        self.poolingSizeTextBox.setAlignment(Qt.AlignCenter) 
        self.poolingSizeTextBox.setText(str(self.parameters.get("pooling_size"))) 
        self.poolingSizeTextBox.textChanged.connect(self.set_pooling_size_text) 
        
        self.dropoutRateTextBox = QLineEdit(self) 
        self.dropoutRateTextBox.move(800,291)  
        self.dropoutRateTextBox.resize(60,30) 
        self.dropoutRateTextBox.setFont(QtGui.QFont("FreeSans", 12)) 
        self.dropoutRateTextBox.setAlignment(Qt.AlignCenter)  
        self.dropoutRateTextBox.setText(str(self.parameters.get("dropout_rate"))) 
        self.dropoutRateTextBox.textChanged.connect(self.set_dropout_rate_text) 

        self.layerTextBox = QLineEdit(self) 
        self.layerTextBox.move(800,381) 
        self.layerTextBox.resize(60,30) 
        self.layerTextBox.setFont(QtGui.QFont("FreeSans", 12)) 
        self.layerTextBox.setAlignment(Qt.AlignCenter)
        self.layerTextBox.setText(str(self.parameters.get("visualise_layer"))) 
        self.layerTextBox.textChanged.connect(self.set_layer_text)

        self.visualiseTextBox = QLineEdit(self) 
        self.visualiseTextBox.move(800,426)
        self.visualiseTextBox.resize(60,30)
        self.visualiseTextBox.setFont(QtGui.QFont("FreeSans", 12))
        self.visualiseTextBox.setAlignment(Qt.AlignCenter) 
        self.visualiseTextBox.setText(str(self.parameters.get("visualise_from"))) 
        self.visualiseTextBox.textChanged.connect(self.set_visualise_text) 
      
        self.goButton = QPushButton(self)
        self.goButton.move(740,73)  
        self.goButton.resize(120,60)
        self.goButton.setText("GO") 
        self.goButton.setStyleSheet("font-size: 26px; font-family: FreeSans; font-weight: bold") 
        self.goButton.clicked.connect(self.press_go_button) 
        
        # bottom left quadrant                
        self.bottomLeftFrame1 = QFrame(self)                                                   
        self.bottomLeftFrame1.move(30, 530)
        self.bottomLeftFrame1.resize(850, 3)
        self.bottomLeftFrame1.setFrameShape(QFrame.HLine); 
        self.bottomLeftFrame1.setLineWidth(2)
        
        self.bottomLeftFrame2 = QFrame(self)                                                   
        self.bottomLeftFrame2.move(30, 960)
        self.bottomLeftFrame2.resize(850, 3)
        self.bottomLeftFrame2.setFrameShape(QFrame.HLine); 
        self.bottomLeftFrame2.setLineWidth(2)
        
        self.bottomLeftFrame3 = QFrame(self)                                                   
        self.bottomLeftFrame3.move(30, 530)
        self.bottomLeftFrame3.resize(3, 430)
        self.bottomLeftFrame3.setFrameShape(QFrame.VLine); 
        self.bottomLeftFrame3.setLineWidth(2)
        
        self.bottomLeftFrame4 = QFrame(self)                                                   
        self.bottomLeftFrame4.move(880, 530)
        self.bottomLeftFrame4.resize(3, 430)
        self.bottomLeftFrame4.setFrameShape(QFrame.VLine); 
        self.bottomLeftFrame4.setLineWidth(2)
        
        self.bottomLeftTitle = QLabel(self)                       
        self.bottomLeftTitle.setText("Weights & gradients")
        self.bottomLeftTitle.move(600,540)
        self.bottomLeftTitle.setStyleSheet("font-size: 26px; font-family: FreeSans; font-weight: bold")  
        
        self.videoViewer1 = QVideoWidget(self)                                   
        self.videoViewer1.move(37,537)                                                
        self.videoViewer1.resize(540,380)
        self.videoViewer1.setStyleSheet("background-color:black;");
        
        self.video_dict1 = {"Welcome": self.path + "/dash/default.avi"}
        self.mediaPlayer1 = QMediaPlayer(self)                                         
        self.mediaPlayer1.setVideoOutput(self.videoViewer1)                                    
        fileName = self.video_dict1.get("Welcome")
        self.mediaPlayer1.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
        self.mediaPlayer1.stateChanged.connect(self.change_state_video1)                       
        self.mediaPlayer1.positionChanged.connect(self.change_position_video1)             
        self.mediaPlayer1.durationChanged.connect(self.change_duration_video1) 
        
        self.videoButton1 = QPushButton(self)                         
        self.videoButton1.move(37,925)                                             
        self.videoButton1.resize(40,30)                                        
        self.videoButton1.setIconSize(QSize(18,18))                                 
        self.videoButton1.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))   
        self.videoButton1.clicked.connect(self.play_stop_video1)                              
        
        self.videoSlider1 = QSlider(Qt.Horizontal,self)
        self.videoSlider1.move(97,925)                                             
        self.videoSlider1.resize(480,30)
        self.videoSlider1.sliderMoved.connect(self.move_slider1)
        
        self.ComboBox1 = QComboBox(self)
        self.ComboBox1.move(628,610)                                             
        self.ComboBox1.resize(200,30)
        self.ComboBox1.addItem("Welcome")
        self.ComboBox1.activated[str].connect(self.select_weight_video) 

        # top right quadrant
        self.topRightFrame1 = QFrame(self)                                                   
        self.topRightFrame1.move(930, 50)
        self.topRightFrame1.resize(850, 3)
        self.topRightFrame1.setFrameShape(QFrame.HLine); 
        self.topRightFrame1.setLineWidth(2)
        
        self.topRightFrame2 = QFrame(self)                                                   
        self.topRightFrame2.move(930, 480)
        self.topRightFrame2.resize(850, 3)
        self.topRightFrame2.setFrameShape(QFrame.HLine); 
        self.topRightFrame2.setLineWidth(2)
        
        self.topRightFrame3 = QFrame(self)                                                   
        self.topRightFrame3.move(930, 50)
        self.topRightFrame3.resize(3, 430)
        self.topRightFrame3.setFrameShape(QFrame.VLine); 
        self.topRightFrame3.setLineWidth(2)
        
        self.topRightFrame4 = QFrame(self)                                                   
        self.topRightFrame4.move(1780, 50)
        self.topRightFrame4.resize(3, 430)
        self.topRightFrame4.setFrameShape(QFrame.VLine); 
        self.topRightFrame4.setLineWidth(2)
        
        self.topRightTitle = QLabel(self)                       
        self.topRightTitle.setText("Loss & accuracy")
        self.topRightTitle.move(1520,60)
        self.topRightTitle.setStyleSheet("font-size: 26px; font-family: \
        FreeSans; font-weight: bold")         

        self.videoViewer2 = QVideoWidget(self)                                   
        self.videoViewer2.move(937,57)                                                
        self.videoViewer2.resize(540,380)
        self.videoViewer2.setStyleSheet("background-color:black;");
        
        self.video_dict2 = {"Welcome": self.path + "/dash/default.avi"}
        self.mediaPlayer2 = QMediaPlayer(self)                                         
        self.mediaPlayer2.setVideoOutput(self.videoViewer2)                                    
        fileName = self.video_dict2.get("Welcome")
        self.mediaPlayer2.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
        self.mediaPlayer2.stateChanged.connect(self.change_state_video2)                       
        self.mediaPlayer2.positionChanged.connect(self.change_position_video2)             
        self.mediaPlayer2.durationChanged.connect(self.change_duration_video2)
        
        self.videoButton2 = QPushButton(self)                         
        self.videoButton2.move(937,445)                                             
        self.videoButton2.resize(40,30)                                        
        self.videoButton2.setIconSize(QSize(18,18))                                 
        self.videoButton2.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))   
        self.videoButton2.clicked.connect(self.play_stop_video2)  
        
        self.videoSlider2 = QSlider(Qt.Horizontal,self)
        self.videoSlider2.move(997,445)                                             
        self.videoSlider2.resize(480,30)
        self.videoSlider2.sliderMoved.connect(self.move_slider2)
        
        self.ComboBox2 = QComboBox(self)
        self.ComboBox2.move(1528,130)                                             
        self.ComboBox2.resize(200,30)
        self.ComboBox2.addItem("Welcome")
        self.ComboBox2.activated[str].connect(self.select_loss_video)       
                
        # bottom right quadrant
        self.bottomRightFrame1 = QFrame(self)                                                   
        self.bottomRightFrame1.move(930, 530)
        self.bottomRightFrame1.resize(850, 3)
        self.bottomRightFrame1.setFrameShape(QFrame.HLine); 
        self.bottomRightFrame1.setLineWidth(2)
        
        self.bottomRightFrame2 = QFrame(self)                                                   
        self.bottomRightFrame2.move(930, 960)
        self.bottomRightFrame2.resize(850, 3)
        self.bottomRightFrame2.setFrameShape(QFrame.HLine); 
        self.bottomRightFrame2.setLineWidth(2)
        
        self.bottomRightFrame3 = QFrame(self)                                                   
        self.bottomRightFrame3.move(930, 530)
        self.bottomRightFrame3.resize(3, 430)
        self.bottomRightFrame3.setFrameShape(QFrame.VLine); 
        self.bottomRightFrame3.setLineWidth(2)
        
        self.bottomRightFrame4 = QFrame(self)                                                   
        self.bottomRightFrame4.move(1780, 530)
        self.bottomRightFrame4.resize(3, 430)
        self.bottomRightFrame4.setFrameShape(QFrame.VLine); 
        self.bottomRightFrame4.setLineWidth(2)
        
        self.bottomRightTitle = QLabel(self)                       
        self.bottomRightTitle.setText("Activations")
        self.bottomRightTitle.move(1560,540)
        self.bottomRightTitle.setStyleSheet("font-size: 26px; font-family: FreeSans; font-weight: bold")
        
        self.videoViewer3 = QVideoWidget(self)                                   
        self.videoViewer3.move(937,537)                                                
        self.videoViewer3.resize(540,380)
        self.videoViewer3.setStyleSheet("background-color:black;");        
        
        self.video_dict3 = {"Welcome": self.path + "/dash/default.avi"}
        self.mediaPlayer3 = QMediaPlayer(self)                                         
        self.mediaPlayer3.setVideoOutput(self.videoViewer3)                                    
        fileName = self.video_dict3.get("Welcome")
        self.mediaPlayer3.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
        self.mediaPlayer3.stateChanged.connect(self.change_state_video3)                       
        self.mediaPlayer3.positionChanged.connect(self.change_position_video3)             
        self.mediaPlayer3.durationChanged.connect(self.change_duration_video3)        
        
        self.videoButton3 = QPushButton(self)                         
        self.videoButton3.move(937,925)                                             
        self.videoButton3.resize(40,30)                                        
        self.videoButton3.setIconSize(QSize(18,18))                                 
        self.videoButton3.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))   
        self.videoButton3.clicked.connect(self.play_stop_video3)  
        
        self.videoSlider3 = QSlider(Qt.Horizontal,self)
        self.videoSlider3.move(997,925)                                             
        self.videoSlider3.resize(480,30)
        self.videoSlider3.sliderMoved.connect(self.move_slider3)        
        
        self.ComboBox3 = QComboBox(self)
        self.ComboBox3.move(1528,610)                                             
        self.ComboBox3.resize(200,30)
        self.ComboBox3.addItem("Welcome")
        self.ComboBox3.activated[str].connect(self.select_activation_video)         
        
        self.bottomRightLabel = QLabel(self)                       
        self.bottomRightLabel.setText("Class")
        self.bottomRightLabel.move(1600,720)
        self.bottomRightLabel.setStyleSheet("font-size: 20px; font-family: FreeSans; font-weight: bold")

        self.class_dict = {"airplane": self.path + "/dash/airplane.jpg",
                           "automobile": self.path + "/dash/automobile.jpg",
                           "bird": self.path + "/dash/bird.jpg",
                           "cat": self.path + "/dash/cat.jpg",
                           "deer": self.path + "/dash/deer.jpg",
                           "dog": self.path + "/dash/dog.jpg",
                           "frog": self.path + "/dash/frog.jpg",
                           "horse": self.path + "/dash/horse.jpg",
                           "ship": self.path + "/dash/ship.jpg",
                           "truck": self.path + "/dash/truck.jpg"}
        self.ComboBox4 = QComboBox(self)
        self.ComboBox4.move(1528,760)                                             
        self.ComboBox4.resize(200,30)
        self.ComboBox4.addItem("airplane")
        self.ComboBox4.addItem("automobile")
        self.ComboBox4.addItem("bird")
        self.ComboBox4.addItem("cat")
        self.ComboBox4.addItem("deer")
        self.ComboBox4.addItem("dog")
        self.ComboBox4.addItem("frog")
        self.ComboBox4.addItem("horse")
        self.ComboBox4.addItem("ship")
        self.ComboBox4.addItem("truck")
        self.ComboBox4.activated[str].connect(self.select_cifar_class)
        
        selected_color = QColor(220,220,220)
        self.imageFrame1 = QFrame(self)                                                   
        self.imageFrame1.move(1568, 815)
        self.imageFrame1.resize(120, 3)
        self.imageFrame1.setFrameShape(QFrame.HLine); 
        self.imageFrame1.setLineWidth(1)
        self.imageFrame1.setStyleSheet('QWidget { background-color: %s}' % selected_color.name())
        
        self.imageFrame2 = QFrame(self)                                                   
        self.imageFrame2.move(1568, 815)
        self.imageFrame2.resize(3, 120)
        self.imageFrame2.setFrameShape(QFrame.HLine); 
        self.imageFrame2.setLineWidth(1)
        self.imageFrame2.setStyleSheet('QWidget { background-color: %s}' % selected_color.name())
        
        self.imageFrame3 = QFrame(self)                                                   
        self.imageFrame3.move(1568, 935)
        self.imageFrame3.resize(120, 3)
        self.imageFrame3.setFrameShape(QFrame.VLine); 
        self.imageFrame3.setLineWidth(1)
        self.imageFrame3.setStyleSheet('QWidget { background-color: %s}' % selected_color.name())
        
        self.imageFrame4 = QFrame(self)                                                   
        self.imageFrame4.move(1688, 815)
        self.imageFrame4.resize(3, 120)
        self.imageFrame4.setFrameShape(QFrame.VLine); 
        self.imageFrame4.setLineWidth(1)
        self.imageFrame4.setStyleSheet('QWidget { background-color: %s}' % selected_color.name())        
        
        self.classIcon = QLabel(self) 
        self.classIcon.setPixmap(QtGui.QPixmap(self.path + '/dash/airplane.jpg').scaled(100, 100))  
        self.classIcon.move(1578, 825) 


    # top left quadrant connects
    def set_epoch_text(self):     
        self.parameters["epochs"] = self.epochTextBox.text()   
                  
    def set_convolution_text(self):     
        self.parameters["convolution"] = self.convolutionTextBox.text() 

    def set_dense_text(self):     
        self.parameters["dense"] = self.denseTextBox.text()
        
    def check_pooling(self, state):     
        if (state == Qt.Checked):
            self.parameters["pooling"] = True 
        else:
            self.parameters["pooling"] = False
            
    def check_dropout(self, state):     
        if (state == Qt.Checked):
            self.parameters["dropout"] = True 
        else:
            self.parameters["dropout"] = False  

    def set_frequency_text(self):     
        self.parameters["frequency"] = self.frequencyTextBox.text()         
        
    def set_window_text(self):     
        self.parameters["window"] = self.windowTextBox.text()  
        
    def set_filter_text(self):     
        self.parameters["filters"] = self.filterTextBox.text()
        
    def set_dimension_text(self):     
        self.parameters["dimensions"] = self.dimensionTextBox.text() 

    def set_pooling_frequency_text(self):     
        self.parameters["pooling_frequency"] = self.poolingFrequencyTextBox.text()      
        
    def set_dropout_frequency_text(self):     
        self.parameters["dropout_frequency"] = self.dropoutFrequencyTextBox.text()
        
    def select_convolution_dense(self):
        if self.convolutionRadioButton.isChecked()==True:
            self.parameters["visualise_conv"] = True
        if self.denseRadioButton.isChecked()==True:
            self.parameters["visualise_conv"] = False
              
    def set_kernel_text(self):     
        self.parameters["kernel_size"] = self.kernelTextBox.text()   

    def set_pooling_size_text(self):     
        self.parameters["pooling_size"] = self.poolingSizeTextBox.text()  
    
    def set_dropout_rate_text(self):     
        self.parameters["dropout_rate"] = self.dropoutRateTextBox.text()  
    
    def set_layer_text(self):     
        self.parameters["visualise_layer"] = self.layerTextBox.text()  
    
    def set_visualise_text(self):     
        self.parameters["visualise_from"] = self.visualiseTextBox.text()   
    
    def press_go_button(self):
        self.worker = Worker(self.parameters, self.path)
        self.worker.start()
        self.worker.updates.connect(self.update_videos)

    def update_videos(self, val):
        update_frequency = int(str(self.parameters.get("frequency")))
        if val % update_frequency == 0:
            number_updates = int(val / update_frequency)
            start_frequency = str(int(str(self.parameters.get("frequency")))*(number_updates-1)+1)
            end_frequency = str(int(str(self.parameters.get("frequency")))*number_updates)
            string1 = "epochs " + str(start_frequency) + "-" + str(end_frequency)
            string2 = self.path + "/medias/weights_and_gradients_epochs_" + str(start_frequency) + "_" + \
            str(int(end_frequency)) + ".avi"
            self.video_dict1[string1] = string2            
            self.ComboBox1.addItem(string1)            
            string3 = self.path + "/medias/loss_and_accuracy_epochs_" + str(start_frequency) + "_" + \
            str(int(end_frequency)) + ".avi"
            self.video_dict2[string1] = string3            
            self.ComboBox2.addItem(string1)            
            string4 = "_epochs_" + str(start_frequency) + "_" + str(int(end_frequency)) + ".avi"
            self.video_dict3[string1] = string4            
            self.ComboBox3.addItem(string1)             
     
    # bottom left quadrant connects
    def change_state_video1(self, state):
        if self.mediaPlayer1.state() == QMediaPlayer.PlayingState:                     
            self.videoButton1.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif self.mediaPlayer1.state() == QMediaPlayer.StoppedState:                    
            self.mediaPlayer1.play()
            self.mediaPlayer1.pause()
        else:
            self.videoButton1.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))  

    def change_position_video1(self, position):
        self.videoSlider1.setValue(position)

    def change_duration_video1(self, duration):
        self.videoSlider1.setRange(0, duration)
    
    def play_stop_video1(self):
        if self.mediaPlayer1.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer1.pause()
        else:
            self.mediaPlayer1.play()
            
    def move_slider1(self, position):            
        self.mediaPlayer1.setPosition(position)
        
    def select_weight_video(self, text):
        for key in self.video_dict1:
            if text == key:
                filename = self.video_dict1.get(key)
        self.mediaPlayer1.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
        self.mediaPlayer1.play()
        self.mediaPlayer1.pause()

    # top right quadrant connects
    def change_state_video2(self, state):
        if self.mediaPlayer2.state() == QMediaPlayer.PlayingState:                     
            self.videoButton2.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif self.mediaPlayer2.state() == QMediaPlayer.StoppedState:                    
            self.mediaPlayer2.play()
            self.mediaPlayer2.pause()
        else:
            self.videoButton2.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))  

    def change_position_video2(self, position):
        self.videoSlider2.setValue(position)

    def change_duration_video2(self, duration):
        self.videoSlider2.setRange(0, duration)
 
    def play_stop_video2(self):
        if self.mediaPlayer2.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer2.pause()
        else:
            self.mediaPlayer2.play() 
            
    def move_slider2(self, position):            
        self.mediaPlayer2.setPosition(position)
        
    def select_loss_video(self, text):
        for key in self.video_dict2:
            if text == key:
                filename = self.video_dict2.get(key)
        self.mediaPlayer2.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))        
        self.mediaPlayer2.play()
        self.mediaPlayer2.pause()      
        
    # bottom right quadrant connects    
    def change_state_video3(self, state):
        if self.mediaPlayer3.state() == QMediaPlayer.PlayingState:                     
            self.videoButton3.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif self.mediaPlayer3.state() == QMediaPlayer.StoppedState:                    
            self.mediaPlayer3.play()
            self.mediaPlayer3.pause()
        else:
            self.videoButton3.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))  

    def change_position_video3(self, position):
        self.videoSlider3.setValue(position)

    def change_duration_video3(self, duration):
        self.videoSlider3.setRange(0, duration)

    def play_stop_video3(self):
        if self.mediaPlayer3.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer3.pause()
        else:
            self.mediaPlayer3.play() 
            
    def move_slider3(self, position):            
        self.mediaPlayer3.setPosition(position)

    def select_activation_video(self, text):
        for key in self.video_dict3:
            if text == key:
                if key == 'Welcome':
                    filename = self.path + "/dash/default.avi"
                else:
                    current_class = str(self.ComboBox4.currentText())
                    filename = self.path + "/medias/activations_and_correlations_class_" + current_class + self.video_dict3.get(key)
        self.mediaPlayer3.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))        
        self.mediaPlayer3.play()
        self.mediaPlayer3.pause()

    def select_cifar_class(self, text):
        for key in self.class_dict:
            if text == key:
                current_class = key
                filename = self.class_dict.get(key)
        self.classIcon.setPixmap(QtGui.QPixmap(filename).scaled(100, 100))                
        video_text = str(self.ComboBox3.currentText())
        if video_text != "Welcome":
            video_filename = self.path + "/medias/activations_and_correlations_class_" + current_class + self.video_dict3.get(video_text)
            self.mediaPlayer3.setMedia(QMediaContent(QUrl.fromLocalFile(video_filename)))        
            self.mediaPlayer3.play()
            self.mediaPlayer3.pause() 


# Worker class: run dnn and produce videos    
class Worker(QtCore.QThread):
    
    updates = pyqtSignal(int)  # signal to communicate with main GUI
    
    def __init__(self, parameters, path, parent=None):
        super(Worker, self).__init__(parent)
        self.path = path
        self.parameters = parameters
    
    def run(self):
        # load the cifar10 dataset
        train_images, train_labels, test_images, test_labels, X_train, y_train, X_test, y_test = load_cifar10()
        
        # model architecture
        dnn_convolution = int(str(self.parameters.get("convolution")))
        dnn_dense = int(str(self.parameters.get("dense")))
        dnn_pooling = self.parameters.get("pooling")
        dnn_dropout = self.parameters.get("dropout")
        filters_string = str(self.parameters.get("filters")).replace("[","").replace("]","")
        dnn_filters = [int(i) for i in [i.replace(" ","") for i in list(filters_string.split(","))]]
        dimensions_string = str(self.parameters.get("dimensions")).replace("[","").replace("]","")
        dnn_dimensions = [int(i) for i in [i.replace(" ","") for i in list(dimensions_string.split(","))]]
        dnn_pooling_frequency = int(str(self.parameters.get("pooling_frequency")))
        dnn_dropout_frequency = int(str(self.parameters.get("dropout_frequency")))
        dnn_kernel_size = int(str(self.parameters.get("kernel_size")))      
        dnn_pooling_size = int(str(self.parameters.get("pooling_size")))
        dnn_dropout_rate = float(str(self.parameters.get("dropout_rate")))
        if dnn_pooling == False:
            pooling_layers = [False] * dnn_convolution
        else:
            pooling_layers = []
            for i in range(dnn_convolution):
                if (i+1)%dnn_pooling_frequency==0:
                    pooling_layers.append(True)
                else:
                    pooling_layers.append(False)
        if dnn_dropout == False:
            dropout_layers = [False] * (dnn_convolution + dnn_dense)
        else:
            dropout_layers = []
            for i in range(dnn_convolution + dnn_dense):
                if (i+1)%dnn_dropout_frequency==0:
                    dropout_layers.append(True)
                else:
                    dropout_layers.append(False)
        model = Sequential()
        model.add(Conv2D(filters=dnn_filters[0], kernel_size=(dnn_kernel_size, dnn_kernel_size), padding='same',\
                         input_shape=(32, 32, 3), activation='relu'))
        for i in range(1, dnn_convolution):
            model.add(Conv2D(filters=dnn_filters[i], kernel_size=(dnn_kernel_size, dnn_kernel_size),\
                             padding='same', activation='relu'))
            if pooling_layers[i]:
                model.add(MaxPooling2D(pool_size=(dnn_pooling_size,dnn_pooling_size)))
            if dropout_layers[i]:
                model.add(Dropout(dnn_dropout_rate))
        model.add(Flatten())
        for i in range(dnn_dense):
            model.add(Dense(units=dnn_dimensions[i], activation='relu'))
            if dropout_layers[dnn_convolution+i]:
                model.add(Dropout(dnn_dropout_rate))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(optimizer=RMSprop(learning_rate=0.0001, decay=1e-6), loss='categorical_crossentropy',\
                      metrics=['accuracy'])
        model.summary()
        
        # other estimation and visualisation parameters (some being default)
        dnn_epochs = int(str(self.parameters.get("epochs")))
        dnn_frequency = int(str(self.parameters.get("frequency")))
        dnn_window = int(str(self.parameters.get("window")))
        dnn_viz_layer = int(str(self.parameters["visualise_layer"]))
        dnn_viz_conv = self.parameters["visualise_conv"]
        dnn_viz_from =  int(str(self.parameters["visualise_from"]))
        if dnn_viz_conv == True:
            dnn_viz = 2 * (dnn_viz_layer - 1)
        else:
            dnn_viz = 2 * dnn_convolution + 2 * (dnn_viz_layer - 1)
        dnn_images = 10
        dnn_backwards = 1
        dnn_fps_loss = 6
        dnn_fps_weights = 3
        dnn_fps_activations = 3
        dnn_project_path = self.path
        dnn_show_images = False
        dnn_delete_images = True
        
        # initiate list of records
        loss_record = []
        val_loss_record = []
        accuracy_record = []
        val_accuracy_record = []
        weight_record = []
        gradient_record = []
        activation_record = []
        
        # define the callbacks to retrieve loss, accuracy, weights, gradients, activations and correlations at the end of each epoch        
        X_gradient, y_gradient = create_batch(labels=train_labels, X=X_train, y=y_train, images=dnn_images)
        X_class = create_class_batch(labels=train_labels, X=X_train, images=dnn_images)
        record_loss = LambdaCallback(on_epoch_end=lambda epoch, logs: loss_record.append(logs['loss']))
        record_val_loss = LambdaCallback(on_epoch_end=lambda epoch, logs: val_loss_record.append(logs['val_loss']))
        record_acc = LambdaCallback(on_epoch_end=lambda epoch, logs: accuracy_record.append(logs['accuracy']))
        record_val_acc = LambdaCallback(on_epoch_end=lambda epoch, logs: val_accuracy_record.append(logs['val_accuracy']))
        record_weights = LambdaCallback(on_epoch_begin=lambda epoch, logs: weight_record.append(model.get_weights()[dnn_viz]))
        record_gradients = LambdaCallback(on_epoch_end=lambda epoch, logs: \
                                          gradient_record.append(get_gradient(X_gradient,y_gradient,dnn_viz,model)))
        record_activations = LambdaCallback(on_epoch_end=lambda epoch, logs: \
                                            activation_record.append(get_activation(X_class,dnn_viz,model,dnn_viz_conv)))  
        create_loss_accuracy_video = LambdaCallback(on_epoch_end=lambda epoch, logs: loss_accuracy_video(epoch=epoch+1, frequency=dnn_frequency, window=dnn_window, loss=loss_record, val_loss=val_loss_record, accuracy=accuracy_record, val_accuracy=val_accuracy_record, path=dnn_project_path, fps=dnn_fps_loss, show=dnn_show_images, delete=dnn_delete_images))
        create_weight_gradient_video = LambdaCallback(on_epoch_end=lambda epoch, logs: weight_gradient_video(epoch=epoch+1, frequency=dnn_frequency, backwards=dnn_backwards, viz_conv=dnn_viz_conv, viz_from=dnn_viz_from, weights=weight_record, gradients=gradient_record, path=dnn_project_path,  fps=dnn_fps_weights, show=dnn_show_images, delete=dnn_delete_images))
        create_activation_correlation_video = LambdaCallback(on_epoch_end=lambda epoch, logs: activation_correlation_video(epoch=epoch+1, frequency=dnn_frequency, viz_conv=dnn_viz_conv, viz_from=dnn_viz_from, activations=activation_record, path=dnn_project_path, fps=dnn_fps_activations, show=dnn_show_images, delete=dnn_delete_images))
        emit_update_signal = LambdaCallback(on_epoch_end=lambda epoch, logs: self.updates.emit(epoch+1))
        
        # training
        model.fit(X_train, y_train, batch_size=128, epochs=dnn_epochs, verbose=1, validation_split=0.2, \
          callbacks=[record_loss, record_val_loss, record_acc, record_val_acc, record_weights,\
                     record_gradients, record_activations,create_loss_accuracy_video, create_weight_gradient_video,\
                     create_activation_correlation_video, emit_update_signal]);

# method to call DnnViewer
def dnn_monitor(path):
    app = QApplication(sys.argv)    
    monitor = DnnMonitor(path)         
    monitor.show()             
    sys.exit(app.exec_())  















