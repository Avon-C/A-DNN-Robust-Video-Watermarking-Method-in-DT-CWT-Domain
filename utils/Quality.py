# Author: Aspertaine
# Date: 2022/6/11 0:08

import torch
import pytorch_ssim

def psnr_all(frames1, frames2):
    frames1 = frames1.permute(0, 4, 1, 2, 3)
    frames2 = frames2.permute(0, 4, 1, 2, 3)
    videos_nums, _, frames_nums, _, _ = frames1.size()
    total_l_psnr = torch.tensor(0.0)  
    total_v_psnr = torch.tensor(0.0) 
    for v in range(videos_nums):
        for l in range(frames_nums):
            mse = torch.mean((frames1[v, :, l, :, :] - frames2[v, :, l, :, :]) ** 2)
            total_l_psnr += 20 * torch.log10(2 / torch.sqrt(mse))
        total_v_psnr += total_l_psnr / frames_nums
    return total_v_psnr / videos_nums


def ssim_all(frames1, frames2):
    frames1 = frames1.permute(0, 1, 4, 2, 3)
    frames2 = frames2.permute(0, 1, 4, 2, 3)
  
    videos_nums, _, _, _, _ = frames1.size()
    total_v_ssim = torch.tensor(0.0)
    for v in range(videos_nums):
        total_v_ssim += pytorch_ssim.ssim(frames1[v, :, :, :, :], frames2[v, :, :, :, :])
    return total_v_ssim / videos_nums
        

if __name__ == '__main__':
    img1 = torch.rand(10, 3, 256, 256)
    img2 = torch.rand(10, 3, 256, 256)
    print(pytorch_ssim.ssim(img1, img2))

    img1.permute(1, 2, 3, 0)
    print(img1.shape)

    f1 = torch.rand(1, 9, 256, 256, 3)
    f2 = f1
    print(psnr_all(f1, f2))
    print(ssim_all(f1, f2))



