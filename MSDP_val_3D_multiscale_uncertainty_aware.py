import math
from glob import glob

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
import matplotlib.pyplot as plt


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, model_name=''):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it 
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
#                     if model_name == 'unet_3D_dv_semi':
                    outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4, log_alea_aux1, log_alea_aux2, log_alea_aux3, log_alea_aux4 = net(test_patch)
#     outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4, log_alea_aux1, log_alea_aux2, log_alea_aux3, log_alea_aux4  = model(volume_batch)
                    mu = torch.stack((outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4),dim=-1)
#             epistemic = torch.var(mu, dim=-1)
                    mu = torch.mean(mu, dim=-1)
#             aleatoric = torch.stack((torch.exp(log_alea_aux1), torch.exp(log_alea_aux2), torch.exp(log_alea_aux3), torch.exp(log_alea_aux4)),dim=-1)
#             aleatoric = torch.mean(aleatoric, dim=-1)
#             noise = torch.randn(batch_size, 2, 256, 256, 128, nb_mc).cuda()
#             reparameterized_output = (mu.unsqueeze(-1).repeat(1, 1, 1, 1, 1, nb_mc)) + noise * (aleatoric.unsqueeze(-1).repeat(1, 1, 1, 1, 1, nb_mc))
#             y_tru = label_batch.unsqueeze(-1).repeat(1, 1, 1, 1, 1, nb_mc)
#             mc_x = ce_loss(reparameterized_output[:args.labeled_bs], y_tru.squeeze(1).long())
#             mc_x = torch.mean(mc_x, dim=-1)
#             attenuated_ce_loss = torch.mean(mc_x)
                    y = torch.softmax(mu, dim=1)
#             loss = attenuated_ce_loss + dice_loss(outputs_soft, label_batch.float()) + torch.mean(epistemic)
#                     else:
#                         y1 = net(test_patch)
                    # ensemble
#                     y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net,val_loader, num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, model_name=''):
    first_total=0.0
    second_total=0.0
    first_metric = np.zeros((1, 2))
    second_metric = np.zeros((1, 2))
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    for sampled_batch in tqdm(val_loader):
        image = sampled_batch['image'][0][0].numpy()
        label = sampled_batch['label'][0][0].numpy()
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, model_name=model_name)


        for i in range(1, num_classes):
            total_metric[i - 1, :] += cal_metric(label == i, prediction == i)
    print("Validation end")
    return total_metric / len(val_loader)


