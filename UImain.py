import sys
from datetime import datetime
import firebase_admin
import numpy
from TestUi import Ui_MainWindow
from PyQt5.QtWidgets import *
from main import *
from firebase_admin import credentials
from firebase_admin import firestore

from new import Mathematics


def runningWithImgVideo(confidence_percent, yolo_string, hopframe, url):
    time_stamp = datetime.now().strftime('%m%d%Y-%H%M%S')
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

        # Set biến thứ tự video đang thực hiện
        working_on_counter = 1

        for video_file in video_directory_list:
            print(
                f'Examining {video_file}: {working_on_counter} of {len(video_directory_list)}: {int((working_on_counter / len(video_directory_list) * 100))}%    ',
                end='')

            # Check người trong video
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

class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)
        self.uic.btnOke.clicked.connect(self.copytext)
        self.uic.pushButton.clicked.connect(self.mybutton_clicked)
        self.uic.btnRun2.clicked.connect(self.runPython)
        self.uic.btnRealtime.clicked.connect(self.realtimeHumanDetect)
        self.uic.btnMedia.clicked.connect(self.btnMediaClicked)
        self.comboBox()



    def btnMediaClicked(self):
        urlPath = self.uic.txtUrl.toPlainText()
        confident = 0.8
        hop_frame = 80
        yoloText = 'yolov4'
        urlPath = str(urlPath)
        time_stamp = datetime.now().strftime('%d%m%Y')
        print(confident,hop_frame,urlPath,yoloText)
        try:
            runningWithImgVideo(confidence_percent=confident, yolo_string=yoloText, hopframe=hop_frame, url=urlPath)
            with open(f'C:\\Users\\Admin\\PycharmProjects\\People-Detecting\\{time_stamp}.txt') as rf:
                line = rf.readline()
                while line:
                    print(line)
                    line = rf.readline()
                    if line == "":
                        break
                    self.uic.listWidget.addItem(line)

        except Exception as e:
            print(e)
            analyze_error = True



    def comboBox(self):
        self.uic.cmbYolo.addItem('yolov4')
        self.uic.cmbYolo.addItem('yolov4-tiny')

    def realtimeHumanDetect(self, test, confident= 0.7, yolo ='yolov4-tiny', gpu=False):
        vid = cv2.VideoCapture(0)
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
        confidence_percent = 0.7
        yoloText = self.uic.cmbYolo.currentText()
        yoloText = str(yoloText)
        print(confidence_percent,yoloText)
        gpu_flag = False
        thresh = float(0.3)
        person_detection_counter = 0
        # while video is running
        while True:
            check, frame = vid.read()
            frame = cv2.flip(frame, 1)
            try:
                # Input: ( frame cần check , lấy model từ thư viện để so sánh,tỉ lệ chấp nhận vật thể và phương thức xử lý )
                # Output: labels - đang xét phải người không - nên sẽ check labels = person
                #        bbox - tọa độ vật thể để vẽ object border
                #        conf - giống với confidence - nếu giá trị cao hơn confidence sẽ chấp nhận vật thể
                result = np.asarray(frame)
                bbox, labels, conf = cvlib.detect_common_objects(result, confidence= confidence_percent, nms_thresh= thresh,  model=yoloText,
                                                                 enable_gpu=gpu_flag)
                marked_frame = cvlib.object_detection.draw_bbox(result, bbox, labels, conf, write_conf=True)
                counter = str((time.time() - start_time) // 1)
                cv2.putText(marked_frame, counter, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                if 'person' in labels:  # xét vật thể dựa trên model của yolo, nếu là person thì biến đếm tăng và cờ set true
                    is_human_found = True
                    print(f'human found at {counter}')
                    chu = f'Alert! human found at second {counter}'
                    self.uic.listWidget.addItem(chu)
                cv2.imshow("Output Video", marked_frame)
            except Exception as e:
                print(e)
                analyze_error = True
                break
            # if 'person' in labels:    # xét vật thể dựa trên model của yolo, nếu là person thì biến đếm tăng và cờ set true
            #     person_detection_counter += 1
            #     is_human_found = True
            # else:
            #     print('Video has ended or failed, try a different video format!')
            #     break
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        vid.release()
        cv2.destroyAllWindows()
        return is_human_found, analyze_error


    # def btnRealTimeClicked(self):
    #     confident, hop_frame, url, yoloText = self.runPython()
    #     print(confident, hop_frame, url, yoloText)
    #     gpu_flag = False
    #     realtimeHumanDetect(yolo=yoloText, confidence=confident, gpu=gpu_flag)



    def copytext(self):
        time_stamp = str(datetime.now().strftime('%d%m%Y'))
        cred = credentials.Certificate("C:\\Users\\Admin\\PycharmProjects\\People-Detecting\\serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        #db.collection('test').document('www').set({'ddd':3})
        with open('C:\\Users\\Admin\\PycharmProjects\\People-Detecting\\'+time_stamp+'.txt') as rf:
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

    def mybutton_clicked(self):
        fileName = QFileDialog.getOpenFileName()
        self.uic.txtUrl.setText(fileName[0])

    def runPython(self):
        confidence = self.uic.txtConfi.toPlainText()
        hopframe = self.uic.txtFrame.toPlainText()
        url = self.uic.txtUrl.toPlainText()
        yolo = self.uic.cmbYolo.currentText()
        return confidence, hopframe ,url, yolo

    def show(self):
        self.main_win.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
