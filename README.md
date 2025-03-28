# 젯슨 나노 졸음 방지 프로젝트 
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
    rects = detector(gray, 1)
    t2 = time.time()

    eye_closed = False

    for dets in rects:
        face_parts_dets = face_parts_detector(gray, dets)
        face_parts = face_utils.shape_to_np(face_parts_dets)

        for (xx, yy) in face_parts:
            cv2.circle(image, (xx, yy), 2, (0, 255, 0), thickness=-1)

        # 눈 감김 감지
        left_eye = face_parts[42:48]
        left_eye_ear = calc_eye(left_eye)
        right_eye = face_parts[36:42]
        right_eye_ear = calc_eye(right_eye)

        if (left_eye_ear + right_eye_ear) < 0.47:
            eye_closed = True
            cv2.putText(image, "Close Your Eye !!!", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            center_right_eye, center_left_eye = eye_center(face_parts_dets)
            cv2.circle(image, (int(center_left_eye[0]), int(center_left_eye[1])), 3, (0, 0, 255), -1)
            cv2.circle(image, (int(center_right_eye[0]), int(center_right_eye[1])), 3, (0, 0, 255), -1)
            eye_closed = False

    # 눈 감김 상태일 경우 부저 울리기
    if eye_closed:
        beep_on()
        time.sleep(3)
        beep_off()
    
    # FPS 및 감지 시간 표시
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    detect_time = t2 - t1
    cv2.putText(image, "FPS: {}".format(int(fps)), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Detect Time: {:.2f}".format(detect_time), (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "{}*{}".format(int(frame_width), int(frame_height)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.moveWindow("DLIB_Landmark", 200, 100)
    cv2.imshow('DLIB_Landmark', image)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cap.release()
GPIO.cleanup()
cv2.destroyAllWindows()
