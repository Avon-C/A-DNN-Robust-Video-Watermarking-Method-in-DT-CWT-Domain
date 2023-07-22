# Author: Aspertaine
# Date: 2022/6/16 0:19
import config
import random
from glob import glob
import torch
from cv2 import COLOR_BGR2YUV, cvtColor
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm
class MyData_H264(Dataset):
    def __init__(self, data_dir, data_h264_dir, frames_to_process, if_eval):
        self.if_eval = if_eval
        self.root_dir = data_dir
        self.frames_to_process = frames_to_process
        self.videos = []
        self.videos_h264 = []
        for ext in ["avi", "mp4"]:
            for path in glob(os.path.join(data_dir, "*.%s" % ext)):
                cap = cv2.VideoCapture(path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.videos.append((path, total_frames))
            for path in glob(os.path.join(data_h264_dir, "*.%s" % ext)):
                cap = cv2.VideoCapture(path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.videos_h264.append((path, total_frames))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        frameList = []
        frameList_h264 = []
        path, total_frames = self.videos[index]
        path_h264, _ = self.videos_h264[index]
        
        cap = cv2.VideoCapture(path)
        start_frame = random.randint(0, total_frames - self.frames_to_process - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(self.frames_to_process):
            _, frame = cap.read()  # numpy.ndarray [H, W, C]

            if (self.if_eval == False):
                frame = cv2.resize(frame, (512, 512))
            else:
                frame = cv2.resize(frame, (720, 720))

            frame = cvtColor(frame, COLOR_BGR2YUV)
            frame = frame / 127.5 - 1.0
            frameList.append(frame)
        frames = torch.FloatTensor(np.array(frameList))

        cap = cv2.VideoCapture(path_h264)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(self.frames_to_process):
            _, frame = cap.read()  # numpy.ndarray [H, W, C]
            if (self.if_eval == False):
                frame = cv2.resize(frame, (512, 512))
            else:
                frame = cv2.resize(frame, (720, 720))
            frame = cvtColor(frame, COLOR_BGR2YUV)
            frame = frame / 127.5 - 1.0
            frameList_h264.append(frame)
        frames_h264 = torch.FloatTensor(np.array(frameList_h264))
        return frames, frames_h264


def load_train(data_dir, batch_size, frames_to_process, if_eval):
    return DataLoader(
        MyData_H264("%s/train" % data_dir, "%s/train_h264_crf25" % data_dir, frames_to_process, if_eval=if_eval),
        shuffle=True,
        batch_size=batch_size,
        drop_last=True
    )

def load_eval(data_dir, frames_to_process, if_eval):
    return DataLoader(
        MyData_H264("%s/val" % data_dir, "%s/val_h264_crf25" % data_dir, frames_to_process, if_eval=if_eval),
        shuffle=False,
        batch_size=1,
        drop_last=True
    )


if __name__ == '__main__':
    train = load_train("../dataset", batch_size=2, frames_to_process=9)
    for f, g in tqdm(train):
        print(f.shape)
        print(g.shape)
    print("================")
    val = load_eval("../dataset", frames_to_process=9)
    for f, g in tqdm(val):
        print(f.shape)
        print(g.shape)
