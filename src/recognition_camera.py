"""
author: Yudong Han
datetime: 2021/06/02
desc: 利用摄像头实时检测
"""
import os

from tensorflow.python.training.tracking.util import capture_dependencies
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
from model import CNN2, CNN3
from utils import index2emotion, cv2_img_add_text
from flask import Flask, render_template
import random
import json
app = Flask(__name__)

WEIGHTS = np.array([-1,-2,0,1,-1,2,0.05,-2])
POINTS = 50
result_list = [0] * POINTS

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/data')
def get_data():
    return json.dumps({'name':'Yudong','value':get_attention_score()})


def load_model():
    """
    加载本地模型
    :return:
    """
    model = CNN3()
    model.load_weights('./models/cnn3_best_weights.h5')
    return model


def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """

    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img)
    resized_images.append(face_img[2:45, :])
    resized_images.append(face_img[1:47, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images

def dis(WH, xywh) -> float:
    return np.linalg.norm(
        np.array((WH[1]/2,WH[0]/2)) - np.array((xywh[1]+xywh[3]/2, xywh[0] + xywh[2]/2))
    )

def get_xywh(frame_gray):
    cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')  # 检测人脸
    faces = cascade.detectMultiScale(frame_gray, minNeighbors=1, minSize=(120, 120))
    # 如果检测到人脸
    if len(faces) > 0:
        return max(faces, key=lambda x:x[2]*x[3])
    else:
        return None

def get_user_photo(capture):
    _, frame = capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化
    result = get_xywh(frame_gray)
    if result is None:
        return b""
    else:
        x,y,w,h = result
        tmp = frame[y: y + h, x: x + w, :]
        retval, buffer = cv2.imencode('.jpg', tmp)
        return buffer

def get_attention_score(capture, exp_model):
    _, frame = capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化
    result = get_xywh(frame_gray)
    if result is None:
        attention = 0
    else:
        x,y,w,h = result
        face = frame_gray[y: y + h, x: x + w]  # 脸部图片
        faces = generate_faces(face)
        results = exp_model.predict(faces)
        result_sum = np.sum(results, axis=0).reshape(-1)
        result_sum = result_sum / np.sum(result_sum)
        attention = np.dot(result_sum, WEIGHTS)

    print(attention)
    # result_list = result_list[1:] + [attention]
    return attention


def predict_expression():
    print('predict_expression')
    # 参数设置
    exp_model = load_model()

    capture = cv2.VideoCapture(0)  # 指定0号摄像头
    
    return capture, exp_model


if __name__ == '__main__':
    predict_expression()
    app.run()