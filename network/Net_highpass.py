# Author: Aspertaine
# Date: 2022/7/6 17:27

from doctest import FAIL_FAST
from random import randint
from tkinter.messagebox import NO
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from Encoder_highpass import Dense_Encoder
from Decoder_highpass import Dense_Decoder
import Noise
from datetime import datetime
from utils.Quality import psnr_all, ssim_all
from utils.Accuracy import get_accuracy
from utils.DataLoad_highpass import *
from utils.WatermarkProcess_highpass import watermark_generator
import gc
import json
import config
from tqdm import tqdm
from h264_network import Net_h264


def lr_decay(lr, epoch, opt):
    if epoch == 5000:
        lr = lr * 0.1
    else:
        lr = lr * (0.1 ** (epoch // 2))
    for param_group in opt.param_groups:
        param_group['lr'] = lr


class VWNet(nn.Module):
    def __init__(self, wm_factor):
        super().__init__()
        self.wm_factor = wm_factor
        self.encoder = Dense_Encoder(wm_expand=[1, 1, 1], wm_factor=wm_factor).to(config.device)
        self.decoder = Dense_Decoder().to(config.device)
        self.h264 = Net_h264.Toh264().to(config.device)

    def fit(self, data_dir, log_dir=False,
            frames_to_process=9, batch_size=2, lr=5e-4,
            epochs=300, use_noise=True):

        self.h264.load_state_dict(torch.load(
            "F:\\Pycharm\\Workspace\\VW_CNN_DTCWT\\h264_network\\exp_crf28\\2022.09.25-13.16.24\\model_state_2.pth",
            map_location='cuda:0'), strict=False)

        for p in self.h264.parameters():
            p.requires_grad = False

        if not log_dir:
            log_dir = "exp_highpass/%s" % (datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))
        os.makedirs(log_dir, exist_ok=True)

        train = load_train(data_dir,
                           batch_size=batch_size,
                           frames_to_process=frames_to_process,
                           operation='train')

        optimizer_encoder = optim.Adam(self.encoder.parameters(),
                                       lr=lr,
                                       weight_decay=0.0001)
        optimizer_decoder = optim.Adam(self.decoder.parameters(),
                                       lr=lr,
                                       weight_decay=0.0001)

        with open(os.path.join(log_dir, "config.json"), "wt") as out:
            out.write(json.dumps({
                "log_dir": log_dir,
                "dataset": data_dir,
                "batch_size": batch_size,
                "frames_to_process": frames_to_process,
                "lr": lr,
                "wm_factor": self.wm_factor,
            }, indent=2, default=lambda o: str(o)))

        crop = Noise.Crop()
        scale_noise = Noise.Scale_noise()
        scale_decode_train = Noise.Scale_decode_train()
        scale_decode_eval = Noise.Scale_decode_eval()
        gau_noise = Noise.GaussianNoise()

        def noise(x):
            choice = randint(0, 5)
            if use_noise:
                if choice == 0:
                    return crop(x)
                if choice == 1:
                    return scale_noise(x)
                if choice == 2:
                    return Noise.Del_frames(x)
                if choice == 3:
                    return Noise.Swap_frame(x)
                if choice == 4:
                    return Noise.gaussian_blur(x)
                if choice == 5:
                    return gau_noise(x)

        history = []

        indices_encoder = torch.tensor([0, 2]).to(config.device)
        indices_decoder = torch.tensor([0, 2, 3, 5]).to(config.device)

        for epoch in range(1, epochs + 1):
            gc.collect()
            metrics = {
                "train_loss": [],
                "train_acc": [],
                "val_h264_acc": [],
                "val_psnr": [],
                "val_ssim": [],
                "val_crop_acc": [],
                "val_scale_acc": [],
                "val_del_acc": [],
                "val_swap_acc": [],
                "val_blur_acc": [],
                "val_gauNoise_acc": []
            }
            self.encoder.train()
            self.decoder.train()
            iterator = tqdm(train)
            cur_lr = 0.0
            for Y, U, V, low, high in iterator:
                lr_decay(lr, epoch, optimizer_encoder)
                lr_decay(lr, epoch, optimizer_decoder)
                for param_group in optimizer_encoder.param_groups:
                    cur_lr = param_group['lr']
                gc.collect()

                w = watermark_generator().to(config.device)
                x = torch.index_select(high[1], 3, indices_encoder)
                x = x[:, :, :, :, :, :, 0].squeeze(2)
                ans = self.encoder(x.permute(0, 2, 1, 3, 4), w, 2, 9).permute(0, 2, 1, 3, 4)
                ans = ans.unsqueeze(2)
                high[1][:, :, :, [0, 2], :, :, 0] = ans
                u_embedded = DTCWT_highpass.dtcwt_frame(low, high, batch_size)
                

                frames_embedded = torch.cat([Y, u_embedded, V], dim=4)
                frames_embedded_h264 = self.h264.net(frames_embedded)  
                u_embedded_attack = noise(frames_embedded_h264[:, :, :, :, [1]]) 

         
                u_embedded_decode = scale_decode_train(u_embedded_attack)
                wm_position = []
                for i in range(batch_size):
                    high_embedded = DTCWT_highpass.frame_dtcwt_no_low(u_embedded_decode[i, :, :, :, :])
                    x_ = torch.index_select(high_embedded[1], 2, indices_decoder)
                    x_ = x_[:, :, :, :, :, 0].squeeze(1)
                    wm_position.append(x_) 
                extract_wm = self.decoder(torch.stack(wm_position).permute(0, 2, 1, 3, 4))

                _, _, H, W = w.size()
                w = w.view(-1, H, W).expand(batch_size, H, W)
                mse = nn.MSELoss().to(config.device)
                loss_W = 0.0
                loss_Q = 0.0
                for i in range(batch_size):
                    loss_W += mse(extract_wm[i, :, :], w[i, :, :])
                    loss_Q += mse(u_embedded[i, :, :], U[i, :, :])
                loss_total = loss_W + 1.5 * loss_Q
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                loss_total.backward()
                optimizer_encoder.step()
                optimizer_decoder.step()

                metrics["train_loss"].append(loss_total.item())
                metrics["train_acc"].append(
                    get_accuracy(extract_wm, w).detach().cpu())

                iterator.set_description("Epoch %s | Loss %.6f | Acc %.4f" % (
                    epoch,
                    np.mean(metrics["train_loss"]),
                    np.mean(metrics["train_acc"])
                ))

            gc.collect()
            
            self.encoder.eval()
            self.decoder.eval()
            val = load_eval(data_dir, frames_to_process=frames_to_process, operation='eval')
            iterator = tqdm(val)
            with torch.no_grad():
                for Y, U, V, low, high in iterator:
                    gc.collect()

                    w = watermark_generator().to(config.device)

                    x = torch.index_select(high[1], 3, indices_encoder)
                    x = x[:, :, :, :, :, :, 0].squeeze(2) 
                    ans = self.encoder(x.permute(0, 2, 1, 3, 4), w, 1, 9).permute(0, 2, 1, 3, 4)
                    ans = ans.unsqueeze(2) 
                    high[1][:, :, :, [0, 2], :, :, 0] = ans
                    u_embedded = DTCWT_highpass.dtcwt_frame(low, high, 1)

                    frames_embedded = torch.cat([Y, u_embedded, V], dim=4)
                    frames_embedded_h264 = self.h264.net(frames_embedded)
                    u_embedded_h264 = frames_embedded_h264[:, :, :, :, [1]]

                    high_embedded = DTCWT_highpass.frame_dtcwt_no_low(u_embedded_h264[0, :, :, :, :])
                    x_ = torch.index_select(high_embedded[1], 2, indices_decoder)
                    x_ = x_[:, :, :, :, :, 0].squeeze(1).unsqueeze(0)
                    extract_wm = self.decoder(x_.permute(0, 2, 1, 3, 4))

                    u_embedded_scale = scale_noise(u_embedded_h264)
                    u_embedded_decode = scale_decode_eval(u_embedded_scale)
                    high_embedded = DTCWT_highpass.frame_dtcwt_no_low(u_embedded_decode[0, :, :, :, :])
                    x_ = torch.index_select(high_embedded[1], 2, indices_decoder)
                    x_ = x_[:, :, :, :, :, 0].squeeze(1).unsqueeze(0)
                    extract_wm_scale = self.decoder(x_.permute(0, 2, 1, 3, 4))

                    u_embedded_crop = crop(u_embedded_h264)
                    high_embedded = DTCWT_highpass.frame_dtcwt_no_low(u_embedded_crop[0, :, :, :, :])
                    x_ = torch.index_select(high_embedded[1], 2, indices_decoder)
                    x_ = x_[:, :, :, :, :, 0].squeeze(1).unsqueeze(0)
                    extract_wm_crop = self.decoder(x_.permute(0, 2, 1, 3, 4))

                    u_embedded_del = Noise.Del_frames(u_embedded_h264)
                    high_embedded = DTCWT_highpass.frame_dtcwt_no_low(u_embedded_del[0, :, :, :, :])
                    x_ = torch.index_select(high_embedded[1], 2, indices_decoder)
                    x_ = x_[:, :, :, :, :, 0].squeeze(1).unsqueeze(0)
                    extract_wm_del = self.decoder(x_.permute(0, 2, 1, 3, 4))

                    u_embedded_swap = Noise.Swap_frame(u_embedded_h264)
                    high_embedded = DTCWT_highpass.frame_dtcwt_no_low(u_embedded_swap[0, :, :, :, :])
                    x_ = torch.index_select(high_embedded[1], 2, indices_decoder)
                    x_ = x_[:, :, :, :, :, 0].squeeze(1).unsqueeze(0)
                    extract_wm_swap = self.decoder(x_.permute(0, 2, 1, 3, 4))

                    u_embedded_blur = Noise.gaussian_blur(u_embedded_h264)
                    high_embedded = DTCWT_highpass.frame_dtcwt_no_low(u_embedded_blur[0, :, :, :, :])
                    x_ = torch.index_select(high_embedded[1], 2, indices_decoder)
                    x_ = x_[:, :, :, :, :, 0].squeeze(1).unsqueeze(0)
                    extract_wm_blur = self.decoder(x_.permute(0, 2, 1, 3, 4))

                    u_embedded_gauNoise = gau_noise(u_embedded_h264)
                    high_embedded = DTCWT_highpass.frame_dtcwt_no_low(u_embedded_gauNoise[0, :, :, :, :])
                    x_ = torch.index_select(high_embedded[1], 2, indices_decoder)
                    x_ = x_[:, :, :, :, :, 0].squeeze(1).unsqueeze(0)
                    extract_wm_gauNoise = self.decoder(x_.permute(0, 2, 1, 3, 4))

                    _, _, H, W = w.size()
                    w = w.view(1, H, W)
                    metrics["val_h264_acc"].append(get_accuracy(extract_wm, w).detach().cpu())
                    metrics["val_scale_acc"].append(get_accuracy(extract_wm_scale, w).detach().cpu())
                    metrics["val_crop_acc"].append(get_accuracy(extract_wm_crop, w).detach().cpu())
                    metrics["val_del_acc"].append(get_accuracy(extract_wm_del, w).detach().cpu())
                    metrics["val_swap_acc"].append(get_accuracy(extract_wm_swap, w).detach().cpu())
                    metrics["val_blur_acc"].append(get_accuracy(extract_wm_blur, w).detach().cpu())
                    metrics["val_gauNoise_acc"].append(get_accuracy(extract_wm_gauNoise, w).detach().cpu())

                    frames = torch.cat([Y, U, V], dim=4).detach().cpu() + 1.0
                    embedded_frames = torch.cat([Y, u_embedded, V], dim=4).detach().cpu() + 1.0
                    metrics["val_psnr"].append(psnr_all(frames, embedded_frames))
                    metrics["val_ssim"].append(ssim_all(frames, embedded_frames))
                    iterator.set_description(
                        "%s | PSNR %.3f | SSIM %.3f | H264_ACC %.6f | CROP %.4f | SCALE %.4f | DEL %.4f | SWAP %.4f | Gau_blur %.4f | Gau_noise %.4f" % (
                            epoch,
                            np.mean(metrics["val_psnr"]),
                            np.mean(metrics["val_ssim"]),
                            np.mean(metrics["val_h264_acc"]),
                            np.mean(metrics["val_crop_acc"]),
                            np.mean(metrics["val_scale_acc"]),
                            np.mean(metrics["val_del_acc"]),
                            np.mean(metrics["val_swap_acc"]),
                            np.mean(metrics["val_blur_acc"]),
                            np.mean(metrics["val_gauNoise_acc"]),
                        )
                    )

            metrics = {
                k: round(np.mean(v), 6) if len(v) > 0 else "NaN"
                for k, v in metrics.items()
            }
            metrics["epoch"] = epoch
            metrics["LR"] = cur_lr
            history.append(metrics)
            pd.DataFrame(history).to_csv(
                os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
            with open(os.path.join(log_dir, "metrics.json"), "at") as out:
                out.write(json.dumps(metrics, indent=2,
                                     default=lambda o: str(o)))
            torch.save(self, os.path.join(log_dir, f"model_{epoch}.pth"))
            torch.save(self.state_dict(), os.path.join(
                log_dir, f"model_state_{epoch}.pth"))
        return history

    def encode(self, high, low, w):
        indices_encoder = torch.tensor([0, 2]).to(config.device)

        x = torch.index_select(high[1], 3, indices_encoder
        x = x[:, :, :, :, :, :, 0].squeeze(2)  
        ans = self.encoder(x.permute(0, 2, 1, 3, 4), w, 1, 9).permute(0, 2, 1, 3, 4)
        ans = ans.unsqueeze(2)  
        high[1][:, :, :, [0, 2], :, :, 0] = ans
        u_embedded = DTCWT_highpass.dtcwt_frame(low, high, 1).squeeze(0)  # real
        # u_embedded = DTCWT_highpass.dtcwt_frame(low, high, 1)  # test
        return u_embedded

    def decode(self, u):
        u = u.squeeze(0)  
        indices_decoder = torch.tensor([0, 2, 3, 5]).to(config.device)
        high = DTCWT_highpass.frame_dtcwt_no_low(u)
        high[1] = high[1].unsqueeze(0) 
        x = torch.index_select(high[1], 3, indices_decoder)
        x = x[:, :, :, :, :, :, 0].squeeze(2)  
        ans = self.decoder(x.permute(0, 2, 1, 3, 4))
        return ans  # test

