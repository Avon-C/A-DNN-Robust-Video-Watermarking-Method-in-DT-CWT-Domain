# Author: Aspertaine
# Date: 2022/6/26 14:42

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from datetime import datetime
import gc
import json
import config
from h264_network import enc_dec, h264_DataLoad
from tqdm import tqdm


def lr_decay(lr, epoch, opt):
    if epoch == 3000:
        lr = lr * 0.1
    else:
        lr = lr * (0.1 ** (epoch // 3))
    for param_group in opt.param_groups:
        param_group['lr'] = lr


def get_acc(f1, f2):  # [B, 9, 256, 256, 3]
    total_cor = 0.0
    total_pixel = f1.size(2) * f1.size(3)
    for i in range(f1.size(0)):  # batch_size
        single_cor = 0.0
        for q in range(f1.size(1)):  # 9
            channel_cor = 0.0
            for k in range(f1.size(4)):  # 3
                cor = 1.0 - (torch.sqrt((f1[i, q, :, :, k] - f2[i, q, :, :, k]) ** 2).sum() / total_pixel)
                channel_cor += cor
            single_cor += channel_cor / f1.size(4)
        total_cor += single_cor / f1.size(1)
    acc = total_cor / f1.size(0)
    return abs(acc)


class Toh264(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = enc_dec.h264_().to(config.device)

    def fit(self, data_dir, log_dir=False,
            frames_to_process=9, batch_size=2, lr=5e-2,  # 0.005s
            epochs=300):

        if not log_dir:
            log_dir = "h264_network/exp/%s" % (datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))
        os.makedirs(log_dir, exist_ok=True)

        train = h264_DataLoad.load_train(data_dir,
                                         batch_size=batch_size,
                                         frames_to_process=frames_to_process,
                                         if_eval=False)
        optimizer = optim.Adam(self.net.parameters(),
                               lr=lr,
                               weight_decay=0.0001)

        with open(os.path.join(log_dir, "config.json"), "wt") as out:
            out.write(json.dumps({
                "log_dir": log_dir,
                "dataset": data_dir,
                "batch_size": batch_size,
                "frames_to_process": frames_to_process,
                "lr": lr,
            }, indent=2, default=lambda o: str(o)))

        history = []
        for epoch in range(1, epochs + 1):
            gc.collect()
            metrics = {
                "train_loss": [],
                "train_acc": [],
                "val_acc": [],
            }
            self.net.train()
            iterator = tqdm(train)
            cur_lr = 0.0
            for frames, frames_h264 in iterator:
                frames = frames.to(config.device)
                frames_h264 = frames_h264.to(config.device)
                lr_decay(lr, epoch, optimizer)
                lr_decay(lr, epoch, optimizer)
                for param_group in optimizer.param_groups:
                    cur_lr = param_group['lr']
                gc.collect()

                frames_to = self.net(frames)
                mse = nn.MSELoss().to(config.device)
                loss = 0.0
                for i in range(batch_size):
                    for k in range(frames_to_process):
                        for q in range(3):
                            loss += mse(frames_to[i, k, :, :, q], frames_h264[i, k, :, :, q])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics["train_loss"].append(loss.item())
                metrics["train_acc"].append(get_acc(frames_to, frames_h264).detach().cpu())

                iterator.set_description("Epoch %s | Loss %.6f | Acc %.4f" % (
                    epoch,
                    np.mean(metrics["train_loss"]),
                    np.mean(metrics["train_acc"])
                ))
            gc.collect()
            self.net.eval()
            val = h264_DataLoad.load_eval(data_dir, frames_to_process=frames_to_process, if_eval=True)
            iterator = tqdm(val)
            with torch.no_grad():
                for frames, frames_h264 in iterator:
                    frames = frames.to(config.device)
                    frames_h264 = frames_h264.to(config.device)

                    frames_to = self.net(frames)

                    metrics["val_acc"].append(get_acc(frames_to, frames_h264).detach().cpu())

                    iterator.set_description(
                        "%s | ACC %.4f" % (
                            epoch,
                            np.mean(metrics["val_acc"]),
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
                out.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))
            torch.save(self, os.path.join(log_dir, f"model_{epoch}.pth"))
            torch.save(self.state_dict(), os.path.join(log_dir, f"model_state_{epoch}.pth"))
        return history

    def to264(self, x):
        x = x.unsqueeze(0)
        return self.net(x).squeeze(0).detach().cpu()


if __name__ == '__main__':
    model = Toh264()
    model.fit("/home/zwl/cxm/Hollywood2")
