# Author: Aspertaine
# Date: 2022/7/6 17:16

import config
import random
from glob import glob
import torch
from cv2 import COLOR_BGR2YUV, cvtColor
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from utils import DTCWT_highpass
from tqdm import tqdm
from utils.WatermarkProcess_highpass import watermark_generator

class MyData(Dataset):
    def __init__(self, data_dir, frames_to_process, operation):
        self.operation = operation
        self.root_dir = data_dir
        self.frames_to_process = frames_to_process
        self.videos = []
        for ext in ["avi", "mp4"]:
            for path in glob(os.path.join(data_dir, "*.%s" % ext)):
                cap = cv2.VideoCapture(path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.videos.append((path, total_frames))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        frames_Y = []
        frames_U = []
        frames_V = []
        path, total_frames = self.videos[index]
        cap = cv2.VideoCapture(path)
        start_frame = random.randint(0, total_frames - self.frames_to_process - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(self.frames_to_process):
            _, frame = cap.read() 
            if (self.operation == 'train'):
                frame = cv2.resize(frame, (512, 512))
            if (self.operation == 'eval'):
                frame = cv2.resize(frame, (768, 768))
            if (self.operation == 'test'):
                frame = cv2.resize(frame, (384, 384))
            if (self.operation == 'big'):
                frame = cv2.resize(frame, (1024, 1024))
            if (self.operation == 'dvmark'):
                frame = cv2.resize(frame, (128, 128))
            frame = cvtColor(frame, COLOR_BGR2YUV)
            frame = frame / 127.5 - 1.0
            frames_Y.append(frame[:, :, [0]])  
            frames_U.append(frame[:, :, [1]])
            frames_V.append(frame[:, :, [2]])
        frames_U = torch.FloatTensor(np.array(frames_U)).to(config.device)  
        frames_low, frames_high = DTCWT_highpass.frame_dtcwt_with_low(frames_U)
        frames_Y = torch.FloatTensor(np.array(frames_Y)).to(config.device)
        frames_V = torch.FloatTensor(np.array(frames_V)).to(config.device)
        return frames_Y, frames_U, frames_V, frames_low, frames_high


def load_train(data_dir, batch_size, frames_to_process, operation):
    return DataLoader(
        MyData("%s/train" % data_dir, frames_to_process, operation=operation),
        shuffle=True,
        batch_size=batch_size,
        drop_last=True
    )

def load_eval(data_dir, frames_to_process, operation):
    return DataLoader(
        MyData("%s/v" % data_dir, frames_to_process, operation=operation),
        shuffle=False,
        batch_size=1,
        drop_last=True
    )

def load_test(data_dir, frames_to_process, operation):
    return DataLoader(
        MyData("%s" % data_dir, frames_to_process, operation=operation),
        shuffle=False,
        batch_size=1,
        drop_last=True
    )

