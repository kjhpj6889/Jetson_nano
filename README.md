#! /usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils
from scipy.spatial import distance
import time
import copy

# Jetson Nano와 라즈베리파이에서 호환되도록 GPIO 라이브러리 선택
try:
    import RPi.GPIO as GPIO  # 라즈베리파이용
except ImportError:
    import Jetson.GPIO as GPIO  # Jetson Nano용

# 피에조 부저 설정
BUZZER_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

face_parts_detector = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# 카메라 장치 열기
cap = cv2.VideoCapture(0)

# 카메라 프레임 크기를 VGA로 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로 480

# 프레임 크기 가져오기
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# 눈 감김을 감지하는 함수
def calc_eye(eye):
    p2_p6 = distance.euclidean(eye[1], eye[5])
    p3_p5 = distance.euclidean(eye[2], eye[4])
    p1_p4 = distance.euclidean(eye[0], eye[3])
    EAR = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return round(EAR, 3)

# 눈의 중심 좌표를 계산하는 함수
def eye_center(shape):
    eyel, eyer = np.array([0, 0]), np.array([0, 0])
    for i in range(36, 42):
        eyel[0] += shape.part(i).x
        eyel[1] += shape.part(i).y
    for i in range(42, 48):
        eyer[0] += shape.part(i).x
        eyer[1] += shape.part(i).y
    return eyel / 6, eyer / 6

# 부저를 울리는 함수
def beep_on():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)

def beep_off():
    GPIO.output(BUZZER_PIN, GPIO.LOW)

# 메인 처리 루프
while True:
    tick = cv2.getTickCount()
    ret, image = cap.read()
    image = cv2.resize(image, dsize=(int(frame_width), int(frame_height)))
    face_frame = copy.deepcopy(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    t1 = time.time()
    정류 = 검출기(gray, 1)
    t2 = time.time()

    eye_closed = 거짓

    직역에서 dets의 경우:
     face_parts_dets = face_parts_detector(gray, dets)
     face_parts = face_utils.shape_to_np(face_parts_dets)

     (xx, yy)에 대해 face_parts:
     CV2.circle(이미지, (xx, yy), 2, (0, 255, 0), 두께=-1)

     # 눈 감김 감지
     왼쪽 눈 = 얼굴 부분 [42:48]
     왼쪽_eye_ear = calc_eye (왼쪽_eye)
     오른쪽 눈 = 얼굴 부분 [36:42]
     오른쪽_eye_ear = calc_eye (오른쪽_eye)

     (왼쪽_눈_귀 + 오른쪽_눈_귀) < 0.47인 경우:
     눈_닫힌 = 참
     cv2.putText(이미지, "눈 감기!!!", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.라인_AA)
     그렇지 않으면:
     centre_right_eye, centre_left_eye = eye_center(face_parts_dets)
     CV2.circle(이미지, (int(센터_왼쪽_눈[0]), int(센터_왼쪽_눈[1]), 3, (0, 0, 255), -1)
            cv2.circle(image, (int(center_right_eye[0]), int(center_right_eye[1])), 3, (0, 0, 255), -1)
            eye_closed = False

    # 눈 감김 상태일 경우 부저 울리기
    if eye_closed:
     비프_온 ()
     시간.수면(3)
     비프_오프 ()
    
    # FPS 및 감지 시간 표시
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - 틱)
    detect_time = t2 - t1
    cv2.putText(이미지, "FPS: {}}.format(int(fps))), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.라인_AA)
    cv2.putText(이미지, "Detect Time: {:.2f}") 형식(Detect_time), (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.라인_AA)
    cv2.putText(이미지, "{}*{}}).format(int(프레임_폭), int(프레임_높이)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.라인_AA)

    cv2.moveWindow("DLIB_Landmark", 200, 100)
    cv2.imshow('DLIB_Landmark', 이미지)

    k = cv2.waitKey(10) & 0xff
    k == 27인 경우:
     브레이크.

cap.release()
GPIO.cleanup()
cv2.destroyAllWindows()
