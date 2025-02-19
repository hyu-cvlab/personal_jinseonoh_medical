import math

# import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.util import compute_sdf



def getLargestCC(segmentation):
    labels = label(segmentation)

    assert(labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    return largestCC

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, model_name=""):
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
                    y1, sdf = net(test_patch)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
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
    return label_map, sdf[:,0,...].squeeze().cpu().data.numpy()


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, val_loader,val_files, method="unet_3D", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, save_result=True, test_save_path=None, metric_detail=0 , nms=0, mask_voxel_counts=[], pixel_spacing_xs=[] ,pixel_spacing_ys=[], slice_gaps=[], model_name=""):

    total_metric = np.zeros((num_classes-1, 4))
    print("Testing begin")
    with open(test_save_path + "/log.txt", "a") as f:
        ith = 0
        loader = tqdm(val_loader) if not metric_detail else val_loader
        dice_list = np.zeros((num_classes-1, len(val_files)))
        jacc_list = np.zeros((num_classes-1, len(val_files)))
        hd_list = np.zeros((num_classes-1, len(val_files)))
        ASD_list = np.zeros((num_classes-1, len(val_files)))
        for j, sampled_batch in enumerate(loader):
#             affine_ = sampled_batch['image_meta_dict']['affine']
            name = val_files[j]['image'].split("/")[-1].split(".")[0]
            #ids = sampled_batch.split("/")[-1].replace(".h5", "")
            image = sampled_batch['image'][0][0].numpy()
            label = sampled_batch['label'][0][0].numpy()

            '''
            plt.figure(figsize=(18, 18))
            # for idx in range(3):
            plt.subplot(2, 1, 1)
            plt.imshow(image[:, :, 90:91], cmap='gray')
            plt.subplot(2, 1, 2)
            plt.imshow(label[:, :, 90:91], cmap='gray')

            plt.tight_layout()
            plt.show()
            print()
            '''


            prediction, sdf = test_single_case(
                net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, model_name=model_name)
            print("jinijn check sdf:::::", sdf.shape)
            print("jinijn check prediction:::::", prediction.shape)

            if nms:
                prediction = getLargestCC(prediction)

            metric = np.zeros((num_classes - 1, 4))
            for i in range(1, num_classes):
                metric[i - 1, :] = calculate_metric_percase(label == i, prediction == i)
            total_metric += metric

            dice_list[:,j] = metric[:,0]      #dice 2class
            jacc_list[:,j] = metric[:,1]      #jacc 2class
            hd_list[:,j] = metric[:,2]        #hd 2class
            ASD_list[:,j] = metric[:,3]       #asd 2class

            for i in range(1, num_classes):
                f.writelines("num:{}, class :{},dice:{},jacc:{},hd:{},asd:{}\n".format(
                    ith, i, metric[i - 1 ,0], metric[i - 1,1], metric[i - 1,2], metric[i - 1,3]))
                f.writelines('\n')

            if metric_detail:
                for i in range(1, num_classes):
                    print('num:%02d \tclass:%d, \tdice:%.5f, jacc:%.5f, hd:%.5f, asd:%.5f' % (
                        ith, i , metric[i - 1,0], metric[i - 1,1], metric[i - 1,2], metric[i - 1,3]))

            if save_result:

                '''
                pred_nii = nib.Nifti1Image(prediction.astype(np.uint8), None)
                nib.save(pred_nii, test_save_path + "/{}_{}_pred.nii.gz".format(ith, name[:11]))

                img_nii = nib.Nifti1Image(image, None)
                nib.save(img_nii, test_save_path + "/{}_{}_img.nii.gz".format(ith, name[:11]))

                lab_nii = nib.Nifti1Image(label.astype(np.uint8),None)
                nib.save(lab_nii, test_save_path + "/{}_{}_gt.nii.gz".format(ith, name[:11]))
                '''
                prediction = np.transpose(prediction, (2,1,0))
                print("jinijn check prediction2:::::", prediction.shape)

                image = np.transpose(image, (2,1,0))
                label = np.transpose(label, (2,1,0))
                sdf = np.transpose(sdf, (2,1,0))


                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                if len(pixel_spacing_xs) > 0:
#                     print(type(pixel_spacing_xs[j]))
#                     print(pixel_spacing_xs[j])
#                     print(type(pixel_spacing_ys[j]))
#                     print(pixel_spacing_ys[j])
#                     print(type(slice_gaps[j]))
#                     print(slice_gaps[j])
                    
                    pred_itk.SetSpacing((pixel_spacing_xs[j].item(),pixel_spacing_ys[j].item(),slice_gaps[j].item()))
                else:
                    pred_itk.SetSpacing((0.8 ,0.8 ,0.8))
                sitk.WriteImage(pred_itk, test_save_path +
                                "/{}_{}_pred.nii.gz".format(ith,name[:11]))
                
                img_itk = sitk.GetImageFromArray(image)
                if len(pixel_spacing_xs) > 0:
                    img_itk.SetSpacing((pixel_spacing_xs[j].item(),pixel_spacing_ys[j].item(),slice_gaps[j].item()))
                else:
                    img_itk.SetSpacing((0.8 ,0.8 ,0.8))
                sitk.WriteImage(img_itk, test_save_path +
                                "/{}_{}_img.nii.gz".format(ith,name[:11]))

                lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
                if len(pixel_spacing_xs) > 0:
                    lab_itk.SetSpacing((pixel_spacing_xs[j].item(),pixel_spacing_ys[j].item(),slice_gaps[j].item()))
                else:
                    lab_itk.SetSpacing((0.8 ,0.8 ,0.8))
                sitk.WriteImage(lab_itk, test_save_path +
                                "/{}_{}_gt.nii.gz".format(ith,name[:11]))
                
                sdf_itk = sitk.GetImageFromArray(sdf)
                sdf_itk.SetSpacing((0.8,0.8,0.8))
                sitk.WriteImage(sdf_itk, test_save_path + "/{}_{}_sdf.nii.gz".format(ith, name[:11]))



            ith += 1




        #f.writelines("Mean metrics\tdice:{},jacc:{},hd:{},asd:{}".format(avg_total_metric[0] / len(val_loader), avg_total_metric[1] / len(
        #        val_loader), avg_total_metric[2] / len(val_loader), avg_total_metric[3] / len(val_loader)))

        for i in range(num_classes - 1):
            f.writelines('\nclass:{} \n'.format(i + 1))
            f.writelines('dice_mean:{}\n'.format(np.mean(dice_list[i])))
            #f.writelines('dice_std:{}'.format(np.std(dice_list[i])))
            f.writelines('jacc_mean:{}\n'.format(np.mean(jacc_list[i])))
            # f.writelines('jacc_std:{}'.format(np.std(jacc_list[i])))
            f.writelines('HD_mean:{}\n'.format(np.mean(hd_list[i])))
            # f.writelines('HD_std:{}'.format(np.std(hd_list[i])))
            f.writelines('ASD_mean:{}\n'.format(np.mean(ASD_list[i])))
            # f.writelines('ASD_std:{}'.format(np.std(ASD_list[i])))



    f.close()
    print("Testing end")
    return total_metric / len(val_loader), dice_list,jacc_list, hd_list, ASD_list


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(gt, pred):
    # 나중에 test할 때 shape 정확히 확인하고 수정하자
#     if pred.sum() > 0 and gt.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return np.array([dice, hd95])
#     else:
#         return np.zeros(2)
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    #ravd = abs(metric.binary.ravd(pred, gt))
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return np.array([dice, jc, hd, asd])
