# Author: Aspertaine
# Date: 2022/7/6 15:51

import torch
import config
from pytorch_wavelets import DTCWTForward, DTCWTInverse



def frame_dtcwt_with_low(u):
    u = u.permute(0, 3, 1, 2)
    xfm = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b').to(config.device)
    low_pass, high_pass = xfm(u)
    return low_pass, high_pass


def frame_dtcwt_no_low(b):
    b = b.permute(0, 3, 1, 2)
    xfm = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b', skip_hps=[True, False]).to(config.device)
    _, high_pass = xfm(b)
    return high_pass


def dtcwt_frame(low_batch, high_batch, batch_size):
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(config.device)
    u_embedded = []
    for i in range(batch_size):
        high = []
        low = low_batch[i, :, :, :, :]
        for j in range(2):
            high.append(high_batch[j][i, :, :, :, :, :, :])
        u_embedded.append(ifm((low, high)))  
    u_embedded = torch.stack(u_embedded).permute(0, 1, 3, 4, 2)
    return u_embedded
