import argparse
import os
import shutil
import numpy as np
import torch
from networks.vnet_kendall import VNet
from monai.networks.nets import UNETR
# from monai.networks.nets import AttentionUnet
from networks.attention_unet import Attention_UNet
from Prostate_test_3D_util_kendall import test_all_case
import torch.nn as nn
import nibabel as nib

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    CenterSpatialCropd
)



from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)

def read_nifti_spacing_and_mask_count(nifti_file_path):
    try:
        # NIfTI 파일 열기
        img = nib.load(nifti_file_path)

        # spacing 정보 추출
        spacing = img.header.get_zooms()
        mask_voxel_count = np.count_nonzero(img.get_fdata()!=0)
        return mask_voxel_count, spacing[0], spacing[1], spacing[2]
    except Exception as e:
        print(f"Error reading NIfTI file: {str(e)}")
        return None

def Inference(args,device):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),   #LPS ->
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8,0.8,0.8),
                mode=("bilinear", "nearest"),
            ),
            CenterSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128)),
            #SpatialPadd(keys=["image", "label"], spatial_size=(320, 320, 32), mode="constant"),
        ]
    )

    datasets = args.root_path + "/dataset_fold{}.json".format(args.fold)
#     print("total_prostate train : dataset.json")
#     if args.class_name == 1 :
#         datasets = args.root_path + "/dataset_fold{}.json".format(args.fold)
#         print("total_prostate train : dataset.json")
#     if args.class_name == 2:
#         datasets = args.root_path + "/dataset_2_fold{}.json".format(args.fold)
#         print("transition zone train :dataset_2.json")
#     train_files = load_decathlon_datalist(datasets, True, "training")      
    val_files = load_decathlon_datalist(datasets, True, args.phase)

    if args.class_name == 1:
        pass
    if args.class_name == 2:
        # '/label_trim/'을 '/label_2_trim/'으로 치환
#         for file_info in train_files:
#             file_info['label'] = file_info['label'].replace('/label_trim/', '/label_2_trim/')
        for file_info in val_files:
            file_info['label'] = file_info['label'].replace('/label_trim/', '/label_2_trim/')
    if args.class_name == -1:
        # '/label_trim/'을 '/label_multi_trim/'으로 치환
#         for file_info in train_files:
#             file_info['label'] = file_info['label'].replace('/label_trim/', '/label_multi_trim/')
        for file_info in val_files:
            file_info['label'] = file_info['label'].replace('/label_trim/', '/label_multi_trim/')

#     val_files = load_decathlon_datalist(datasets, True, args.phase)

    db_val = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_loader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    
    directory_path = "/data/hanyang_Prostate/50_example/trim/sl_data_wo_norm/centerCrop_350_350_200/label_trim/"
    mask_voxel_counts = []
    pixel_spacing_xs = []
    pixel_spacing_ys = []
    slice_gaps = []
    if args.origin_spacing:
        for _ in range(len(db_val)):
            no = db_val[_]['label_meta_dict']['filename_or_obj'].split('/')[-1][:8]

            original_nifti_file_path=''
            # 디렉토리 탐색
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if no in file:
                        # 검색한 문자열을 포함한 파일의 전체 경로 출력
                        original_nifti_file_path = os.path.join(root, file)
            mask_voxel_count, pixel_spacing_x, pixel_spacing_y, slice_gap = read_nifti_spacing_and_mask_count(original_nifti_file_path)
            mask_voxel_counts.append(mask_voxel_count)
            pixel_spacing_xs.append(pixel_spacing_x)
            pixel_spacing_ys.append(pixel_spacing_y)
            slice_gaps.append(slice_gap)


    if args.class_name == 1:
#         snapshot_path = "/data/sohui/Prostate/prostate_1c_train_result/{}/{}".format(args.exp, args.model)
        #snapshot_path = "/data/sohui/BraTS/data/brats_2class_train_result/BraTs19_label_40_290/{}/{}".format(args.exp, args.model)
        snapshot_path = "/data/hanyang_Prostate/Prostate/prostate_1c_train_result/{}/{}".format(args.exp, args.model)
    elif args.class_name == 2:
#         snapshot_path = "/data/sohui/Prostate/TZ_1c_train_result/{}/{}".format(args.exp, args.model)
        snapshot_path = "/data/hanyang_Prostate/Prostate/TZ_1c_train_result/{}/{}".format(args.exp, args.model)
    elif args.class_name == -1:
        snapshot_path = "/data/hanyang_Prostate/Prostate/multi_train_result/{}/{}".format(args.exp, args.model)
    num_classes = args.num_classes

    if args.class_name == 1:
#         test_save_path = "/data/sohui/Prostate/prostate_1c_test_result/{}/{}".format(args.exp, args.model)
        test_save_path = "/data/hanyang_Prostate/Prostate/prostate_1c_test_result/{}/{}/{}".format(args.exp, args.model, args.phase)
    elif args.class_name == 2:
#         test_save_path = "/data/sohui/Prostate/TZ_1c_test_result/{}/{}".format(args.exp, args.model)
        test_save_path = "/data/hanyang_Prostate/Prostate/TZ_1c_test_result/{}/{}/{}".format(args.exp, args.model, args.phase)
    elif args.class_name == -1:
        test_save_path = "/data/hanyang_Prostate/Prostate/multi_test_result/{}/{}/{}".format(args.exp, args.model, args.phase)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = None
    if "vnet" in args.model or "test" in args.model:
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)

    elif "attention_unet" in args.model:
        net = Attention_UNet(in_channels=1, n_classes=num_classes, is_batchnorm=True)

    elif "unetr" in args.model:
        net = UNETR(in_channels=1, out_channels=num_classes, img_size=(256,256,128), feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, conv_block=True, dropout_rate=0.0)
        
    else:
        pass

    #net = unet_3D(n_classes=num_classes, in_channels=1)
#     net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
#     net = Attention_UNet(in_channels=1, n_classes=num_classes, is_batchnorm=True)
#     net = AttentionUnet(
#     spatial_dims=3,        # 3D 데이터를 다루는 경우 (2D인 경우는 dimensions=2)
#     in_channels=1,  # 입력 이미지 채널 수
#     out_channels=num_classes,   # 출력 클래스 수
# #             channels=[64, 128, 256, 512, 1024],
#     channels=[16, 32, 64, 128],  # 채널 수 설정
#     strides=[2, 2, 2, 2],      # 스트라이드 설정
#     kernel_size=3,             # 컨볼루션 커널 크기
#     dropout= 0.1
# )
    if len(args.gpu.split(',')) > 1:
        net = nn.DataParallel(net).to(device)
    else :
        net = net.cuda()

    pth_files = [f for f in os.listdir(snapshot_path) if f.endswith(".pth")]
    # print(pth_files)
    if not pth_files:
        print("No .pth files found in the snapshot path.")
    else:
        max_score, best_model_filename = max(
            ((float(f.split("_dice_")[1][:-4]), f) for f in pth_files if "_dice_" in f),
            default=(None, None)
        )
    save_mode_path = os.path.join(snapshot_path, best_model_filename)#'iter_10000_dice_0.7169.pth')

#     net.load_state_dict(torch.load(save_mode_path))
    checkpoint = torch.load(save_mode_path)#["state_dict"]
    
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.','')] = checkpoint[key]
            del checkpoint[key]
            
    net.load_state_dict(checkpoint, strict=False)
    
    print("init weight from {}".format(save_mode_path))
    net.eval()
    metric, dice_list,jacc_list, hd_list, ASD_list = test_all_case(net, val_loader =val_loader, val_files=val_files, method=args.model, num_classes=num_classes,
                               patch_size=args.patch_size, stride_xy=64, stride_z=64, save_result=True, test_save_path=test_save_path,
                               metric_detail=args.detail,nms=args.nms, mask_voxel_counts=mask_voxel_counts, pixel_spacing_xs=pixel_spacing_xs, pixel_spacing_ys=pixel_spacing_ys, slice_gaps=slice_gaps, model_name=args.model)

    return metric, dice_list,jacc_list, hd_list, ASD_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
#                         default='/data/sohui/Prostate/data/trim/ssl_data/centerCrop_200', help='Name of Experiment')
                        default='/data/hanyang_Prostate/50_example/trim/sl_data_wo_norm/centerCrop_350_350_200', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='SSL/MT_ATO_350_350_200_rampup_refpaper', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='Vnet_3D_256_randomCrop_30000_fold3', help='model_name')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--patch_size', type=list, default=[256,256,128],
                        help='patch size of network input')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--gpu', type=str, default='4,5', help='GPU to use')
    parser.add_argument('--detail', type=int, default=1,
                        help='print metrics for every samples?')
    parser.add_argument('--nms', type=int, default=0,
                        help='apply NMS post-procssing?')
    parser.add_argument('--origin_spacing', type=int, default=0,
                        help='using original spacing:1 vs unifying spacing:0')
    parser.add_argument('--class_name', type=int, default=1)
    parser.add_argument('--fold', type=int, default=3, help='k fold cross validation')
    parser.add_argument('--phase', type=str,
                        default='test', help='training|val|test')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric, dice_list,jacc_list, hd_list, ASD_list = Inference(args, device=device)
    print("Dataset phase:{}".format(args.phase))
    for i in range((args.num_classes)-1):
        print('class:{}'.format(i+1))
        print('dice_mean:{}'.format(np.mean(dice_list[i])))
        #print('dice_std:{}'.format(np.std(dice_list[i])))
        print('jacc_mean:{}'.format(np.mean(jacc_list[i])))
        # print('jacc_std:{}'.format(np.std(jacc_list[i])))
        print('HD_mean:{}'.format(np.mean(hd_list[i])))
        #print('HD_std:{}'.format(np.std(hd_list[i])))
        print('ASD_mean:{}'.format(np.mean(ASD_list[i])))
        # print('ASD_std:{}'.format(np.std(ASD_list[i])))

