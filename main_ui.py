import os

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
# from PyQt5.uic.properties import QtGui
from PySide6 import QtCore
import sys
from datetime import datetime
import firebase_admin
import numpy
from modules.ui_main import *
from PyQt5.QtWidgets import *
from main import *
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import timedelta
import cv2
import os
from os.path import isfile, join

from modules import *
from modules.ui_splash_screen import Ui_SplashScreen

os.environ["QT_FONT_DPI"] = "96"  # FIX Problem for High DPI and Scale above 100%

# SET AS GLOBAL WIDGETS
# ///////////////////////////////////////////////////////////////
widgets = None
## ==> GLOBALS
counter = 0
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
time_stamp_folder = datetime.now().strftime('%m%d%Y-%H%M%S')

def convert_pictures_to_video(pathIn, pathOut, fps, time):
    ''' this function converts images to video'''
    frame_array = []
    files = [f for f in os.listdir(pathIn) if f.endswith(".jpeg")]
    for i in range(len(files)):
        filename = os.path.join(pathIn, files[i])
        print(filename)
        '''reading images'''
        try:
            img = cv2.imread(filename)
            img = cv2.resize(img, (1280, 720))
            height, width, layers = img.shape
            size = (width, height)
        except Exception as e:
            print(str(e))
        for k in range(time):
            frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()



def runningWithImgVideo(confidence_percent, yolo_string, hopframe, url):
    time_stamp = time_stamp_folder
    os.mkdir(time_stamp)
    directorypath = os.path.dirname(url)
    print(confidence_percent, yolo_string, hopframe, url)
    print(directorypath)
    gpu_flag = False
    with open(time_stamp + '/' + time_stamp + '.txt', 'w') as log_file:
        if url == '':
            alert = 'Choose file for directory'
            return alert
        else:
            print('OK')
            video_directory_list = getListOfFiles(directorypath + '/')

        # Set bi???n th??? t??? video ??ang th???c hi???n
        working_on_counter = 1

        for video_file in video_directory_list:
            print(
                f'Examining {video_file}: {working_on_counter} of {len(video_directory_list)}: {int((working_on_counter / len(video_directory_list) * 100))}%    ',
                end='')
            # Check ng?????i trong video
            human_detected, error_detected = humanChecker(str(video_file), time_stamp, yolo=yolo_string,
                                                          hop_frame=hopframe, confidence=confidence_percent,
                                                          gpu=gpu_flag)
            if human_detected:
                HUMAN_DETECTED_ALERT = True
                print(f'Human detected in {video_file} ')
                log_file.write(f'{video_file}  \n')

            if error_detected:
                ERROR_ALERT = True
                print(f'\nError in analyzing {video_file}')
                log_file.write(f'Error in analyzing {video_file} \n')
            working_on_counter += 1


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        # ///////////////////////////////////////////////////////////////
        title = "Py Application"

        # APPLY TEXTS
        self.setWindowTitle(title)
        #        widgets.titleRightInfo.setText(description)

        # TOGGLE MENU
        # ///////////////////////////////////////////////////////////////
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))

        # SET UI DEFINITIONS
        # ///////////////////////////////////////////////////////////////
        UIFunctions.uiDefinitions(self)

        # LEFT MENUS
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)

        # SHOW APP
        # ///////////////////////////////////////////////////////////////
        self.show()

        # SET CUSTOM THEME
        # ///////////////////////////////////////////////////////////////
        useCustomTheme = False
        themeFile = "themes\py_dracula_light.qss"

        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)

            # SET HACKS
            AppFunctions.setThemeHack(self)

        # SET HOME PAGE AND SELECT MENU
        # ///////////////////////////////////////////////////////////////
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))

        # ===============================================
        # BUTTONS
        # ===============================================
        self.ui.btnWebcam.clicked.connect(self.realtimeHumanDetect)
        self.comboBox()  # SHOW COMBO_BOX
        self.ui.btnOpenFile_2.clicked.connect(self.getFileName)
        self.ui.slideConfident.valueChanged.connect(self.numberConfident)
        self.ui.slideFrame.valueChanged.connect(self.numberFrame)
        self.ui.btnMedia.clicked.connect(self.btnMediaClicked)
        self.ui.btnClearLog.clicked.connect(self.clearLog)
        self.ui.browseButton.clicked.connect(self.openFile)
        self.ui.playButton.clicked.connect(self.playTimer)
        self.ui.stopButton.clicked.connect(self.stopTimer)
        self.ui.slider.valueChanged.connect(self.skipFrame)

        self.timer = QTimer()
        self.timer.timeout.connect(self.playVideo)

    def skipFrame(self):
        value = self.ui.slider.value()
        self.cap.set(1, value)


    def playVideo(self):
         # read image in BGR format
        ret, image = self.cap.read()
        if ret is True:
            progress = str(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))) + ' / ' \
                       + str(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            self.ui.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            # convert image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # crop image
            image = image[0:1280, 0:1280]
            # resize image
            image = cv2.resize(image, (1280, 720))
            # get image infos
            height, width, channel = image.shape
            step = channel * width
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            self.ui.display.setPixmap(QPixmap.fromImage(qImg))
        else:
            progress = str(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) + ' / ' \
                       + str(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.ui.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

    def playTimer(self):
        self.timer.start(20)

    def stopTimer(self):
        self.timer.stop()

    def openFile(self):
        self.videoFileName = QFileDialog.getOpenFileName(self, 'Select Video File')
        self.file_name = list(self.videoFileName)[0]
        self.cap = cv2.VideoCapture(self.file_name)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.ui.slider.setMinimum(0)
        self.ui.slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.timer.start(fps)


    def clearLog(self):
        self.ui.listLog.clear()

    def numberConfident(self):
        new_value = str(self.ui.slideConfident.value())
        self.ui.txtConfident.setText(new_value)

    def numberFrame(self):
        new_value = str(self.ui.slideFrame.value())
        self.ui.txtFrame.setText(new_value)

    def buttonClick(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # SHOW HOME PAGE
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW WIDGETS PAGE
        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.widgets)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # PRINT BTN NAME
        print(f'Button "{btnName}" pressed!')

    # RESIZE EVENTS
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')

    def btnMediaClicked(self):
        print('long')
        urlPath = self.ui.txtUrl.toPlainText()
        print(urlPath)
        if len(self.ui.txtConfident.toPlainText()) == 0:
            print("Ch???n confidence, m???c ?????nh 80\n")
            self.ui.txtConfident.setText("80")
        confident = int(self.ui.txtConfident.toPlainText()) / 100
        # confident = 0.8
        if len(self.ui.txtFrame.toPlainText()) == 0:
            print("Ch???n frame, m???c ?????nh 80\n")
            self.ui.txtFrame.setText("80")
        hop_frame = int(self.ui.txtFrame.toPlainText())
        yoloText = 'yolov4'
        urlPath = str(urlPath)
        time_stamp = datetime.now().strftime('%d%m%Y')
        print(confident, hop_frame, urlPath, yoloText)
        try:
            runningWithImgVideo(confidence_percent=confident, yolo_string=yoloText, hopframe=hop_frame, url=urlPath)
            with open(f'{ROOT_DIR}\\{time_stamp}.txt') as rf:
                line = rf.readline()
                while line:
                    print(line)
                    line = rf.readline()
                    if line == "":
                        break
                    self.ui.listLog.addItem(line)

        except Exception as e:
            print(e)
            analyze_error = True
        self.getDirectory_OutVideo()
        self.copytext()
        self.splitVideo()

    def comboBox(self):
        self.ui.cbModel.addItem('yolov4')
        self.ui.cbModel.addItem('yolov4-tiny')

    def realtimeHumanDetect(self, confident=0.7, yolo='yolov4-tiny', gpu=False):
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_num = 0
        is_human_found = False
        analyze_error = False
        start_time = time.time()
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (30, 30)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 1
        confidence_percent = int(self.ui.txtConfident.toPlainText())/100
        yoloText = self.ui.cbModel.currentText()
        # yoloText = 'yolov4'
        print(confidence_percent, yoloText)
        gpu_flag = False
        thresh = float(0.3)
        person_detection_counter = 0
        # while video is running
        while True:
            check, frame = vid.read()
            frame = cv2.flip(frame, 1)
            try:
                # Input: ( frame c???n check , l???y model t??? th?? vi???n ????? so s??nh,t??? l??? ch???p nh???n v???t th??? v?? ph????ng th???c x??? l?? )
                # Output: labels - ??ang x??t ph???i ng?????i kh??ng - n??n s??? check labels = person
                #        bbox - t???a ????? v???t th??? ????? v??? object border
                #        conf - gi???ng v???i confidence - n???u gi?? tr??? cao h??n confidence s??? ch???p nh???n v???t th???
                result = np.asarray(frame)
                bbox, labels, conf = cvlib.detect_common_objects(result, confidence=confidence_percent,
                                                                 nms_thresh=thresh, model=yoloText,
                                                                 enable_gpu=gpu_flag)
                marked_frame = cvlib.object_detection.draw_bbox(result, bbox, labels, conf, write_conf=True)
                counter = str((time.time() - start_time) // 1)
                cv2.putText(marked_frame, counter, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

                if 'person' in labels:  # x??t v???t th??? d???a tr??n model c???a yolo, n???u l?? person th?? bi???n ?????m t??ng v?? c??? set true
                    is_human_found = True
                    print(f'human found at {counter}')
                    chu = f'Alert! human found at second {counter}'
                    self.ui.listLog.addItem(chu)
                cv2.imshow("Output Video", marked_frame)
            except Exception as e:
                print(e)
                analyze_error = True
                vid.release()
                cv2.destroyAllWindows()
                break
            # if 'person' in labels:    # x??t v???t th??? d???a tr??n model c???a yolo, n???u l?? person th?? bi???n ?????m t??ng v?? c??? set true
            #     person_detection_counter += 1
            #     is_human_found = True
            # else:
            #     print('Video has ended or failed, try a different video format!')
            #     break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                vid.release()
                cv2.destroyAllWindows()
                break
        vid.release()
        cv2.destroyAllWindows()
        return is_human_found, analyze_error

    def copytext(self):
        time_stamp = str(datetime.now().strftime('%d%m%Y'))
        cred = credentials.Certificate(f"{ROOT_DIR}\\serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        # db.collection('test').document('www').set({'ddd':3})
        with open(f'{ROOT_DIR}\\' + time_stamp + '.txt') as rf:

            line = rf.readline()
            index = 1
            while line:
                a = line.split('-')
                print(a)
                db.collection(time_stamp).document(a[2]).set({'Second': a[3], 'frame': a[4]})
                index += 1
                line = rf.readline()
                if line == "":
                    break
        print("done")

    def splitVideo(self):
        startTime = 0
        endTime = 0
        time_stamp = datetime.now().strftime('%d%m%Y')
        with open(f'{ROOT_DIR}\\' + time_stamp + '.txt') as rf:
            line = rf.readline()
            x = line.split('-')
            startTime = round(float(x[2].strip()))
            nameFile = x[0].strip()
            index = 1
            while line:
                a = line.split('-')
                tempTime = round(float(a[2].strip()))
                if endTime < tempTime:
                    endTime = tempTime
                line = rf.readline()
            # if startTime > 60:
            #     startMinute = round(startTime // 60)
            #     startSecond = round(startTime % 60)
            #     startTime = f'00:0{startMinute}:{startSecond}'
            # if endTime > 60:
            #     endMinute = round(endTime // 60)
            #     endSecond = round(endTime % 60)
            #     endTime = f'00:0{endMinute}:{endSecond}'
            startTime = str(timedelta(seconds=startTime))
            endTime = str(timedelta(seconds=endTime))
            os.system(f'ffmpeg -ss {startTime} -i {nameFile}  -to {endTime} -filter:v scale=1280:720 -c:a copy {time_stamp_folder}\\{index}.mp4')
            print("done")


    def getFileName(self):
        fileName = QFileDialog.getOpenFileName()
        self.ui.txtUrl.setText(fileName[0])

    def getDirectory_OutVideo(self):
        pathIn = f'{ROOT_DIR}\\{time_stamp_folder}'
        #pathIn = os.path.dirname(directoryIn)
        print(pathIn)
        namevideo = time_stamp_folder
        pathOut = f'{ROOT_DIR}\\Demo\\'+namevideo+'.avi'
        #pathOut = os.path.dirname(directoryOut)
        fps = 1
        time = 200
        convert_pictures_to_video(pathIn, pathOut, fps, time)

    def runPython(self):
        confidence = self.ui.txtConfident.toPlainText()
        hopframe = self.ui.txtFrame.toPlainText()
        url = self.ui.txtUrl.toPlainText()
        yolo = self.ui.cbModel.currentText()
        return confidence, hopframe, url, yolo



# SPLASH SCREEN
class SplashScreen(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_SplashScreen()
        self.ui.setupUi(self)

        ## UI ==> INTERFACE CODES
        ########################################################################

        ## REMOVE TITLE BAR
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        ## DROP SHADOW EFFECT
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 60))
        self.ui.dropShadowFrame.setGraphicsEffect(self.shadow)

        ## QTIMER ==> START
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)
        # TIMER IN MILLISECONDS
        self.timer.start(35)

        # CHANGE DESCRIPTION

        # Initial Text
        self.ui.label_description.setText("<strong>WELCOME</strong> TO MY APPLICATION")

        # Change Texts
        QtCore.QTimer.singleShot(1500, lambda: self.ui.label_description.setText("<strong>LOADING</strong> DATABASE"))
        QtCore.QTimer.singleShot(3000,
                                 lambda: self.ui.label_description.setText("<strong>LOADING</strong> USER INTERFACE"))

        ## SHOW ==> MAIN WINDOW
        ########################################################################
        self.show()
        ## ==> END ##

    ## ==> APP FUNCTIONS
    ########################################################################
    def progress(self):
        global counter

        # SET VALUE TO PROGRESS BAR
        self.ui.progressBar.setValue(counter)

        # CLOSE SPLASH SCREE AND OPEN APP
        if counter > 100:
            # STOP TIMER
            self.timer.stop()

            # SHOW MAIN WINDOW
            self.main = MainWindow()
            self.main.show()

            # CLOSE SPLASH SCREEN
            self.close()

        # INCREASE COUNTER
        counter += 1


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = SplashScreen()
    sys.exit(app.exec())
