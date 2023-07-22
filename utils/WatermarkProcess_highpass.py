# Author: Aspertaine
# Date: 2022/7/6 20:09
import torch
import cv2

def watermark_generator():
    random_w = torch.randint(0, 2, [32, 32]).to(torch.float32)
    return random_w.view(1, 1, 32, 32)


def watermark_expend(watermark, batch_size, channels, frames_to_process, H, W):
    return watermark.expand(batch_size, channels, frames_to_process, H, W)


