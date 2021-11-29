import cvlib
import cv2
from argparse import ArgumentParser
import os
import sys
from datetime import datetime
import smtplib, ssl
from email.message import EmailMessage
import imghdr
import numpy as np
import time
import math


# Các đuôi file có thể check
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.gif']
VID_EXTENSIONS = ['.mov', '.mp4', '.avi', '.mpg', '.mpeg', '.m4v', '.mkv']

# Flag dùng để check ít nhất 1 file có thể đọc được
VALID_FILE_ALERT = False
# Flag dùng để check error - tạo thông báo error
ERROR_ALERT = False
# Flag dùng để báo đã check có object = người
HUMAN_DETECTED_ALERT = False


# function lấy File name (lấy cả extension - đuôi file), kiểm tra có vật thể = người không (gồm ảnh vào video - offline)
# lưu các frame xác định object = người vào folder 'saved_directory'
# hop_frame - bước nhảy frame trên video (vd: hop_frame = 10 -> cứ 10 frame check 1 )
# yolo = 'yolov4' - sử dụng yolov4 để lấy model người check (có thể dùng cái khác nhanh hơn nhưng sai số sẽ cao hơn )
# continuous - lúc dùng commandline sẽ set cái này để video dù check dc object = người thì vẫn check tiếp
# confidence = .65 - tiêu chuẩn check (vd nếu trên 65% thì sẽ báo là người )
# gpu - để mặc định là false, chạy trên cpu
def humanChecker(video_file_name, save_directory, yolo='yolov4', hop_frame=10, confidence=.65,
                 gpu=False):
    # Khai báo global để ngoài hàm vẫn thay đổi dc
    global VALID_FILE_ALERT
    # Chắc dễ hiểu lười ghi
    is_human_found = False
    analyze_error = False
    is_valid = False
    is_video_file = False
    is_img_file = False
    # biến đếm lượt người
    person_detection_counter = 0

    # Kiểm tra xem có phải file dạng ảnh
    if os.path.splitext(video_file_name)[1] in IMG_EXTENSIONS:
        frame = cv2.imread(video_file_name)  # frame sẽ là ảnh đó - biến dùng chung với video : video_file_name
        # file ảnh có thể lỗi - check xem có lỗi không
        if frame is not None:
            frame_count = 8  # Đặt sẵn biến này để check vòng lặp phía dưới
            hop_frame = 1 # nếu file ảnh thì check từng ảnh
            VALID_FILE_ALERT = True
            is_valid = True
            is_img_file = True
            print(f'Image')
        else:
            is_valid = False
            analyze_error = True


    # Kiểm tra xem có phải file video
    elif os.path.splitext(video_file_name)[1] in VID_EXTENSIONS:
        vid = cv2.VideoCapture(video_file_name)
        # Biến tổng số frame của file video - sẽ có thể không chính xác
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        # file video có thể lỗi - check xem có lỗi không
        if frame_count > 0:
            VALID_FILE_ALERT = True
            is_valid = True
            is_video_file = True
            print(f'{frame_count} frames')
        else:
            is_valid = False
            is_video_file = False
            analyze_error = True



    else:
        print(f'\nSkipping {video_file_name}')

    if is_valid:

        # Mỗi hop_frame của Video sẽ check 1 frame, frame đó sẽ thực hiện hàm detect_common_objects
        # Tăng bước nhảy hop_frame sẽ tăng tốc độ check, nhưng sai số sẽ tăng
        # Do frame_count của video có thể sai, nên sẽ -6 - giảm ở biên cuối tránh lỗi
        for frame_number in range(1, frame_count - 6 , hop_frame): # Từ frame 1 -> Tổng frame - 6 -> mỗi bước = hop_frame

            # Check xem file đang xử lý không phải file ảnh
            if is_video_file:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)            #cv2.CAP_PROP_POS_FRAMES để đọc đúng vị trí frame_number
                _, frame = vid.read()                                   # Đọc frame thứ frame number


            # Dùng frame ở trên ( đã xác định ảnh hay video ) để chạy hàm detect_common_objects
            try:
                #Input: ( frame cần check , lấy model từ thư viện để so sánh,tỉ lệ chấp nhận vật thể và phương thức xử lý )
                #Output: labels - đang xét phải người không - nên sẽ check labels = person
                #        bbox - tọa độ vật thể để vẽ object border
                #        conf - giống với confidence - nếu giá trị cao hơn confidence sẽ chấp nhận vật thể
                bbox, labels, conf = cvlib.detect_common_objects(frame, model=yolo, confidence=confidence,
                                                                 enable_gpu=gpu)
            except Exception as e:
                print(e)
                analyze_error = True
                break

            if 'person' in labels:    # xét vật thể dựa trên model của yolo, nếu là person thì biến đếm tăng và cờ set true
                person_detection_counter += 1
                is_human_found = True



                # Tạo ảnh đã check có hiện diện người, tạo khung bbox trong ảnh đó và lưu file ảnh đó
                marked_frame = cvlib.object_detection.draw_bbox(frame, bbox, labels, conf, write_conf=True)  # tạo ảnh
                save_file_name = os.path.basename(os.path.splitext(video_file_name)[0]) + '-' + str(
                    person_detection_counter) + '.jpeg'                # Lưu ảnh đuôi jpeg, tại folder directory
                if is_video_file:
                    first_found = int(float(str(vid.get(cv2.CAP_PROP_POS_MSEC)/1000//1)))
                    if first_found > 60:
                        startMinute = first_found // 60
                        startSecond = first_found % 60
                        startTime = f'00:0{startMinute}:{startSecond} {video_file_name}'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (50, 50)
                    fontScale = 1
                    color = (0, 0, 255)
                    thickness = 1
                    cv2.putText(marked_frame, startTime, org, font,
                                        fontScale, color, thickness, cv2.LINE_AA)
                    time_stamp = datetime.now().strftime('%d%m%Y')
                    with open(time_stamp + '.txt', 'a') as log_file:
                        print(f'{video_file_name} - times detect: {person_detection_counter} - {first_found}')
                        log_video = f'{video_file_name} - times detect: {person_detection_counter} - {first_found}'
                        log_file.write(f'{video_file_name} - times detect: {person_detection_counter} - {first_found} - hopframe = {hop_frame} - confidence = {confidence*100}\n')
                        cv2.imwrite(save_directory + '/' + save_file_name, marked_frame)

                cv2.imwrite(save_directory + '/' + save_file_name, marked_frame)

    return is_human_found, analyze_error        # return 2 cờ để check



# lấy đường dẫn và trả về tất cả file và đường dẫn thư mục trong đó
def getListOfFiles(dir_name):
    list_of_files = os.listdir(dir_name)
    all_files = list()
    # Lặp với tất cả file trong thư mục
    for entry in list_of_files:
        # Bỏ qua các file và thư mục hidden
        if entry[0] != '.':
            # Tạo biến lưu đường dẫn
            full_path = os.path.join(dir_name, entry)
            # Nếu entry là một thư mục ( đường dẫn ) thì sẽ chạy đệ quy lại lần nữa lấy các file trong thư mục đó
            if os.path.isdir(full_path): # hàm check đường dẫn đó là thư mục hay không
                all_files = all_files + getListOfFiles(full_path)
            else:
                all_files.append(full_path)  # Thêm file vào danh sách các file là biến all_files
    return all_files



#-------------------------------------------------------------------------------------------------------------------------------
# Hàm để gửi email thông báo đã check thấy người ------ đã test với hard code Tài khoản, mật khẩu gmail người gửi
                                                                        # và mail người nhận -------- sẽ triển khai trên GUI
# Hàm này là tính năng
def emailAlertSender(save_directory, SENDER_EMAIL, SENDER_PASS, RECEIVER_EMAIL):
    port = 465  # For SSL                                         # google thì nó v nên chắc nó thế
    smtp_server = "smtp.gmail.com"

    # Ghi nội dung email dựa trên log file ( nếu có )
    with open(save_directory + '/' + save_directory + '.txt') as f:
        msg = EmailMessage()
        msg.set_content(f.read())

    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    if HUMAN_DETECTED_ALERT is True:                                          # Điền subject dựa trên các Flag đã tạo
        msg['Subject'] = 'Intruder Alert'

    elif HUMAN_DETECTED_ALERT is False and VALID_FILE_ALERT is True:
        msg['Subject'] = 'All Clear'

    else:
        msg['Subject'] = 'No Valid Files Examined'

    list_of_files = os.listdir(save_directory)
    # Thêm file đính kèm, không bỏ file .txt vào
    for image_file_name in list_of_files:
        if image_file_name[-3:] != 'txt':
            with open(save_directory + '/' + image_file_name, 'rb') as image:
                img_data = image.read()
            msg.add_attachment(img_data, maintype='image', subtype=imghdr.what(None, img_data),
                               filename=image_file_name)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(SENDER_EMAIL, SENDER_PASS)
        server.send_message(msg)

#-------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Các argument này để có thể dùng trong cmd
    # Full commandline ( tự điền nếu muốn hiệu chỉnh ) VD: python main.py -d D:\ --continuous --confidence 70 --frames 12 ( tạm không email với tiny yolo, gpu )
                                            # chỗ ổ D:\ thay thư mục chi tiết hơn tránh nó tìm r xử lý tới nái
    parser = ArgumentParser()
    parser.add_argument('-d', '--directory', default='', help='Path to video folder')
    parser.add_argument('-f', default='', help='Used to select an individual file')
    parser.add_argument('--email', action='store_true', help='Flag to use email notification')
    parser.add_argument('--tiny_yolo', action='store_true',
                        help='Flag to indicate using YoloV4-tiny model instead of the full one. Will be faster but less accurate.')
    parser.add_argument('--continuous', action='store_true',
                        help='This option will go through entire video file and save all frames with people. Default behavior is to stop after first person sighting.')
    parser.add_argument('--confidence', type=int, choices=range(1, 100), default=65,
                        help='Input a value between 1-99. This represents the percent confidence you require for a hit. Default is 65')
    parser.add_argument('--frames', type=int, default=10, help='Only examine every hop frame. Default is 10')
    parser.add_argument('--gpu', action='store_true',
                        help='Attempt to run on GPU instead of CPU. Requires Open CV compiled with CUDA enables and Nvidia drivers set up correctly.')

    args = vars(parser.parse_args())

    # Sẽ chọn dạng model nào để check, default sẽ set là yolo4, có thể dùng tiny_yolo nhanh hơn nhưng sai số cao hơn yolov4
    if args['tiny_yolo']:
        yolo_string = 'yolov4-tiny'
    else:
        yolo_string = 'yolov4'

    # Check đường dẫn input, có thể dùng -f hoặc -d nhưng phải dùng 1 trong 2 ( f dành cho 1 file, d dành cho thư mục )
    if args['f'] == '' and args['directory'] == '':
        print('You must select either a directory with -d <directory> or a file with -f <file name>')
        sys.exit(1)
    if args['f'] != '' and args['directory'] != '':
        print('Must select either -f or -d but can''t do both')
        sys.exit(1)

    # -- email : nếu sử dụng argument này thì có thể hardcode như ở dưới --------> sẽ triển khai làm GUI để nhập sau không hardcode
    #                                                   không lộ hết -- đã test hàm hoạt động đủ xà

    if args['email']:
        try:
            SENDER_EMAIL = 'trananhvan8329@gmail.com'
            SENDER_PASS = '...'
            RECEIVER_EMAIL = 'trananhvan1102vn@gmail.com'
        except:
            print(
                'Something went wrong with Email variables. Either set your environment variables or hardcode values in to script')
            sys.exit(1)

    every_hop_frame = args['frames']                                # --frames : Set các bước nhảy frame, default là 10
    confidence_percent = args['confidence'] / 100                   # --confidence: Set tỷ lệ nhận diện object, default là 65%

    gpu_flag = False
    if args['gpu']:
        gpu_flag = True                                             # --gpu : xử lý bằng gpu
    # Tạo thư mục tên là thời gian máy chạy check, lưu trữ các snapshots và file log
    time_stamp = datetime.now().strftime('%m%d%Y-%H%M%S')
    os.mkdir(time_stamp)

    print('Beginning Detection')
    print(f'Directory {time_stamp} has been created')
    print(f"Email notifications set to {args['email']}.")
    print(f"Confidence threshold set to {args['confidence']}%")
    print(f'Examining every {every_hop_frame} frames.')
    print(f"Continous examination is set to {args['continuous']}")
    print(f"GPU is set to {args['gpu']}")
    print('\n\n')
    print(datetime.now().strftime('%m%d%Y-%H:%M:%S'))

    # Mở file log và lặp lại trên tất cả các file video
    def runningWithImgVideo(confidence_percent, yolo_string, hopframe , url):
        time_stamp = datetime.now().strftime('%m%d%Y-%H%M%S')
        os.mkdir(time_stamp)
        directorypath = os.path.dirname(url)
        print(confidence_percent,yolo_string,hopframe,url)
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
                                                              hop_frame= hopframe, confidence=confidence_percent, gpu= gpu_flag)

                if human_detected:
                    HUMAN_DETECTED_ALERT = True
                    print(f'Human detected in {video_file} ')
                    log_file.write(f'{video_file}  \n')

                if error_detected:
                    ERROR_ALERT = True
                    print(f'\nError in analyzing {video_file}')
                    log_file.write(f'Error in analyzing {video_file} \n')

                working_on_counter += 1


    # is_human_found, analyze_error = realtimeHumanDetect(yolo_string, confidence_percent, gpu_flag)
    #runningWithImgVideo()
    if VALID_FILE_ALERT is False:
        print('No valid image or video files were examined')

    if args['email'] is True:
        emailAlertSender(time_stamp, SENDER_EMAIL, SENDER_PASS, RECEIVER_EMAIL)
    print(datetime.now().strftime('%m%d%Y-%H:%M:%S'))