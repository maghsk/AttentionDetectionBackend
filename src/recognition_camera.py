"""
author: Yudong Han
datetime: 2021/06/02
desc: 利用摄像头实时检测
"""
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
from model import CNN2, CNN3
from utils import index2emotion, cv2_img_add_text
import threading
import matplotlib.pyplot as plt


WEIGHTS = np.array([-1,-2,0,1,-1,2,0,-2])
POINTS = 50
result_list = [0] * POINTS

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, default=0, help="data source, 0 for camera 1 for video")
parser.add_argument("--video_path", type=str, default=None)
opt = parser.parse_args()

if opt.source == 1 and opt.video_path is not None:
    filename = opt.video_path
else:
    filename = None


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


def get_attention_score(capture, model):
    # threading.Timer(1, get_attention_score, [capture, model]).start()
    _, frame = capture.read()  # 读取一帧视频，返回是否到达视频结尾的布尔值和这一帧的图像
    # plt.imshow(frame)
    # frame = cv2.resize(frame, (800, 600))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化
    cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')  # 检测人脸
    # 利用分类器识别出哪个区域为人脸
    faces = cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, minSize=(120, 120))
    global result_list
    # 如果检测到人脸
    if len(faces) > 0:
        # for (x, y, w, h) in faces:
        # TODO: 选择靠近中间的脸
        x, y, w, h = faces[0]
        distance = dis(frame.shape, faces[0])
        for xywh in faces:
            curdis = dis(frame.shape, xywh)
            if curdis < distance:
                x, y, w, h = xywh
        face = frame_gray[y: y + h, x: x + w]  # 脸部图片
        faces = generate_faces(face)
        results = model.predict(faces)
        result_sum = np.sum(results, axis=0).reshape(-1)
        result_sum = result_sum / np.sum(result_sum)
        attention = np.dot(result_sum, WEIGHTS)
    else:
        attention = 0

    result_list = result_list[1:] + [attention]
    

def show_plot(ax, line):
    global result_list
    line.set_ydata(result_list)
    ax.draw_artist(line)
    ax.figure.canvas.draw()
def show_cam(capture, ax, cam):
    _, frame = capture.read()  # 读取一帧视频，返回是否到达视频结尾的布尔值和这一帧的图像
    cam.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.draw_artist(cam)
    ax.figure.canvas.draw()


def predict_expression():
    print('predict_expression')
    # 参数设置
    model = load_model()

    border_color = (0, 0, 0)  # 黑框框
    font_color = (255, 255, 0)  # 白字字
    capture = cv2.VideoCapture(0)  # 指定0号摄像头
    if filename:
        capture = cv2.VideoCapture(filename)

    fig, (cam_ax, line_ax) = plt.subplots(1, 2)

    line_ax.set_ylim([-3.5, 3.5])
    line_ax.set_xlim([0, POINTS])
    line, = line_ax.plot(range(POINTS), result_list, label='output', color='cornflowerblue')
    _, frame = capture.read()  # 读取一帧视频，返回是否到达视频结尾的布尔值和这一帧的图像
    cam = cam_ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(get_attention_score, capture, model)
    timer.add_callback(show_plot, line_ax, line)
    timer.start()
    timer2 = fig.canvas.new_timer(interval=20)
    timer2.add_callback(show_cam, capture, cam_ax, cam)
    timer2.start()
    # get_attention_score(capture, model, ax)
    plt.show()


if __name__ == '__main__':
    predict_expression()
