# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   16 Dec. 2021
# Implementation for Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer.
# # Reference:
#   @article{luo2021ctbct,
#   title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
#   author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
#   journal={arXiv preprint arXiv:2112.04894},
#   year={2021}}
#   In the original paper, we don't use the train_validation_test set to select checkpoints and use the last iteration to inference for all methods.
#   In addition, we combine the train_validation_test set and test set to report the results.
#   We found that the random data split has some bias (the train_validation_test set is very tough and the test set is very easy).
#   Actually, this setting is also a fair comparison.
#   download pre-trained denseUnet_3D to "code/pretrained_ckpt" folder, link:https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

#from config import get_config

# from Prostate.networks.attnetionunet_monai import AttentionUnet
from networks.vnet_multiscale_uncertainty_aware import VNet
# from networks.attention_unet_2d import AttU_Net
from networks.attention_unet import Attention_UNet
from monai.networks.nets import UNETR
# from networks.attention_unet import Attention_UNet
# from monai.networks.nets import AttentionUnet
from utils import ramps, losses
from MSDP_val_3D_multiscale_uncertainty_aware import test_all_case
from skimage import segmentation as skimage_seg
# from pytorch3dunet.unet3d.losses import HausdorffDistanceDiceLoss
from monai.losses.hausdorff_loss import HausdorffDTLoss
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
#from monai.networks.nets import UNet
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    Spacingd,
    RandRotate90d,
    CenterSpatialCropd,
    RandSpatialCropd
)

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)
# import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    #default='/data/sohui/Prostate/data/trim/sl_data/centerCrop_350_350_200', help='Name of Experiment')
                    default='/data/hanyang_Prostate/50_example/trim/sl_data_wo_norm_morphology_5/centerCrop_350_350_200', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='test', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_10000', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256,256,128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--class_name', type=int,  default=1)
#parser.add_argument(
#    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )


# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')

parser.add_argument('--gpu', type=str,  default='4', help='GPU to use')
parser.add_argument('--add', type=float,  default=1e-8)
parser.add_argument('--sw_batch_size', type=int,  default=8)
parser.add_argument('--overlap', type=float,  default=0.5)
parser.add_argument('--aggKernel', type=int,  default=11, help= 'Aggregation_module_kernelSize')
parser.add_argument('--fold', type=int,  default=None, help='k fold cross validation')
parser.add_argument('--use_weightloss', type=int, default=0, help='0: cee+dice, 1: cee+w_dice, 2: w_cee+w_dice')

args = parser.parse_args()
#config = get_config(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb.init(project="SL_HProstate", config={}, reinit=True)
# wandb.run.name = '{}/{}'.format(args.exp,args.model)

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

initial_weight = 0.1
weight_multiplier = 1.1
# 사용자 정의 스케줄링 함수
def weight_scheduler(epoch):
    # 매 10 epoch마다 weight를 증가
    return min(1, initial_weight * (weight_multiplier ** (epoch // 500)))


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    fold = args.fold

    def create_model(ema=False):
        # Network definition
#         model = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
#         model = Attention_UNet(in_channels=1, n_classes=num_classes, is_batchnorm=True)
        # Attention U-Net 모델 생성
#         model = AttentionUnet(
#             spatial_dims=3,        # 3D 데이터를 다루는 경우 (2D인 경우는 dimensions=2)
#             in_channels=1,  # 입력 이미지 채널 수
#             out_channels=num_classes,   # 출력 클래스 수
# #             channels=[64, 128, 256, 512, 1024],
#             channels=[16, 32, 64, 128],  # 채널 수 설정
#             strides=[2, 2, 2, 2],      # 스트라이드 설정
#             kernel_size=3,             # 컨볼루션 커널 크기
#             dropout= 0.1
#         )
#         model = AttU_Net(img_ch=1,output_ch=num_classes)
        model = None
        if "vnet" in args.model:
            model = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            if ema:
                for param in model.parameters():
                    param.detach_()
        
        elif "attention_unet" in args.model:
            model = Attention_UNet(in_channels=1, n_classes=num_classes, is_batchnorm=True)
        
        elif "unetr" in args.model:
            model = UNETR(in_channels=1, out_channels=num_classes, img_size=(256,256,128), feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, conv_block=True, dropout_rate=0.0)
        else:
            pass
        
        return model

    model = create_model()
    pth_files = [f for f in os.listdir(snapshot_path) if f.endswith(".pth")]
    max_score = 0.0
    if not pth_files:
        print("No .pth files found in the snapshot path.")
    else:
        max_score, best_model_filename = max(
            ((float(f.split("_dice_")[1][:-4]), f) for f in pth_files if "_dice_" in f),
            default=(None, None)
        )
        print("max_score: ", max_score)
    iteration_number = 0
    # print(pth_files)
    if not pth_files:
        print("No .pth files found in the snapshot path.")
#         model = create_model()
    else:        
        # 파일들을 수정된 시간을 기준으로 정렬
        pth_files.sort(key=lambda x: os.path.getmtime(os.path.join(snapshot_path, x)), reverse=True)

        # 정렬된 파일 중 가장 첫 번째 파일 (가장 최근에 저장된 파일) 선택
        latest_file = pth_files[0]
        if "best_model.pth" in latest_file:
            latest_file = pth_files[1]
        print("The latest file name:", latest_file)
        save_mode_path = os.path.join(snapshot_path, latest_file)#'iter_10000_dice_0.7169.pth')

    #     net.load_state_dict(torch.load(save_mode_path))
        checkpoint = torch.load(save_mode_path)#["state_dict"]

        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.','')] = checkpoint[key]
                del checkpoint[key]

        model.load_state_dict(checkpoint, strict=False)
        prefix = "model_iter_"
        suffix = ".pth"
        
        if prefix in latest_file and latest_file.endswith(suffix):
            start_index = latest_file.index(prefix) + len(prefix)
            end_index = latest_file.index(suffix, start_index)
#             print(latest_file[start_index:end_index])
            if 'dice' in latest_file[start_index:end_index]:
                end_index = latest_file.index('_dice_',start_index)
            iteration_number = int(latest_file[start_index:end_index]) 

        print("init weight from {}".format(save_mode_path))
        
    if len(args.gpu.split(',')) > 1:
        model = nn.DataParallel(model).to(device)
    else :
        model = model.cuda()

    model.train()


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    train_transforms1 = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),       #ALI
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8,0.8,0.8),
                mode=("bilinear", "nearest"),
            ),
#             CenterSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128)),
            RandSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128), random_size=False),
        ]
    )
    train_transforms2 = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),       #ALI
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8,0.8,0.8),
                mode=("bilinear", "nearest"),
            ),
#             CenterSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128)),
            RandSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128), random_size=False),#roi_size=(272,272,144), random_size=False),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=1.0,
            ),
        ]
    )
    train_transforms3 = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),       #ALI
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8,0.8,0.8),
                mode=("bilinear", "nearest"),
            ),
#             CenterSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128)),
            RandSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128), random_size=False),#roi_size=(272,272,144), random_size=False),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=1.0,
            ),

        ]
    )
    train_transforms4 = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),       #ALI
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8,0.8,0.8),
                mode=("bilinear", "nearest"),
            ),
#             CenterSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128)),
            RandSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128), random_size=False),#roi_size=(272,272,144), random_size=False),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=1.0,
            ),
        ]
    )
    train_transforms5 = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),       #ALI
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8,0.8,0.8),
                mode=("bilinear", "nearest"),
            ),
#             CenterSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128)),
            RandSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128), random_size=False),#roi_size=(272,272,144), random_size=False),
            RandRotate90d(
                keys=["image", "label"],
                prob=1.0,
                max_k=3,
            ),

        ]
    )
#     train_transforms = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             Orientationd(keys=["image", "label"], axcodes="RAI"),       #ALI
#             Spacingd(
#                 keys=["image", "label"],
#                 pixdim=(0.8,0.8,0.8),
#                 mode=("bilinear", "nearest"),
#             ),
# #             CenterSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128)),
#             RandSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128), random_size=False),#roi_size=(272,272,144), random_size=False),
#             RandFlipd(
#                 keys=["image", "label"],
#                 spatial_axis=[0],
#                 prob=0.10,
#             ),
#             RandFlipd(
#                 keys=["image", "label"],
#                 spatial_axis=[1],
#                 prob=0.10,
#             ),
#             RandFlipd(
#                 keys=["image", "label"],
#                 spatial_axis=[2],
#                 prob=0.10,
#             ),
#             RandRotate90d(
#                 keys=["image", "label"],
#                 prob=0.10,
#                 max_k=3,
#             ),

#         ]
#     )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8 ,0.8 ,0.8),
                mode=("bilinear", "nearest"),
            ),
            CenterSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128))
        ]
    )

    datasets = args.root_path + "/dataset_fold{}.json".format(fold)
    print("total_prostate train : dataset.json")
#     if args.class_name == 1 :
#         datasets = args.root_path + "/dataset_fold{}.json".format(fold)
#         print("total_prostate train : dataset.json")
#     if args.class_name == 2:
#         datasets = args.root_path + "/dataset_2_fold{}.json".format(fold)
#         print("transition zone train :dataset_2.json")
    train_files = load_decathlon_datalist(datasets, True, "training")      
    val_files = load_decathlon_datalist(datasets, True, "test")

    if args.class_name == 1:
        pass
    if args.class_name == 2:
        # '/label_trim/'을 '/label_2_trim/'으로 치환
        for file_info in train_files:
            file_info['label'] = file_info['label'].replace('/label_trim/', '/label_2_trim/')
        for file_info in val_files:
            file_info['label'] = file_info['label'].replace('/label_trim/', '/label_2_trim/')
    if args.class_name == -1:
        # '/label_trim/'을 '/label_merge_trim/'으로 치환
        for file_info in train_files:
            file_info['label'] = file_info['label'].replace('/label_trim/', '/label_multi_trim/')
        for file_info in val_files:
            file_info['label'] = file_info['label'].replace('/label_trim/', '/label_multi_trim/')

    # train_transforms 버전 설정 (예시로 3개의 버전 생성)
    train_transforms_list = [
        train_transforms1,
        train_transforms2,
        train_transforms3,
        train_transforms4,
        train_transforms5
        # 추가적인 버전이 있다면 계속해서 추가 가능
    ]

    ##########train dataload
    # 각각의 데이터셋 생성
    dataset_list = []
    for transform in train_transforms_list:
        db_train_SL_tmp = CacheDataset(
            data=train_files,
            transform=transform,
            cache_num=24,
            cache_rate=1.0,
            num_workers=8,
        )
        dataset_list.append(db_train_SL_tmp)

    # 데이터셋을 결합(concatenate)
    db_train_SL = ConcatDataset(dataset_list)


    SL_trainloader = DataLoader(db_train_SL, batch_size=args.batch_size,shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
#     ##########train dataload
#     db_train_SL = CacheDataset(
#         data=train_files,
#         transform=train_transforms,
#         cache_num=24,
#         cache_rate=1.0,
#         num_workers=8,
#     )


#     SL_trainloader = DataLoader(db_train_SL, batch_size=args.batch_size,shuffle=True,
#                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)  ## 40개 안에서 shuffle



    ##########val dataload
    db_val = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_loader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    

    optimizer1 = optim.SGD(model.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    # 가중치를 동적으로 조절할 스케줄러를 생성
#     scheduler = lr_scheduler.LambdaLR(optimizer1, lr_lambda=weight_scheduler)

    # 학습률 스케줄러 설정
#     scheduler = lr_scheduler.StepLR(optimizer1, step_size=3000, gamma=0.1)
        
    loss_weight = initial_weight#0.1
    class_weights = []
    if args.use_weightloss != 0: # 0: cee+dice, 1: cee+w_dice, 2: w_cee+w_dice
        from collections import defaultdict
        # 클래스별 개수 세기
        class_counts = defaultdict(int)  # 각 클래스별 개수를 저장할 딕셔너리

        for batch in SL_trainloader:
            data = batch['label'].cuda()
        #     print(data)
        #     print(data.shape)
            # 클래스별 개수 세기
            # 특정 값(0, 1, 2)의 개수 세기
            count_0 = torch.sum(torch.eq(data, 0)).item()
            count_1 = torch.sum(torch.eq(data, 1)).item()
            count_2 = torch.sum(torch.eq(data, 2)).item()

    #             print(f"Count of 0: {count_0}")
    #             print(f"Count of 1: {count_1}")
    #             print(f"Count of 2: {count_2}")
    #             print('--------------------------------------')

            class_counts[0] += count_0
            class_counts[1] += count_1
            class_counts[2] += count_2

        class_weights.append((class_counts[0]+class_counts[1]+class_counts[2])/(float(class_counts[0])))
        class_weights.append((class_counts[0]+class_counts[1]+class_counts[2])/(float(class_counts[1])))
        if args.class_name == -1:
            class_weights.append((class_counts[0]+class_counts[1]+class_counts[2])/(float(class_counts[2])))
    
        print(f'class_weights : {class_weights}')
    else: # 0: cee+dice
        class_weights = None
        
    if args.use_weightloss == 2:
        ce_loss = CrossEntropyLoss(weight=torch.tensor(class_weights).to('cuda'))
    else:
        ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    
#     # Create the HD loss function
#     hd_loss = HausdorffDTLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    #logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = iteration_number#0
    max_epoch = (max_iterations-iteration_number) // len(SL_trainloader) + 1
    best_performance1 = max_score#0.0
#     best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kl_distance = nn.KLDivLoss(reduction='none')
    lr_ = base_lr
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(SL_trainloader):       #0,1 : SL_trainloader(bs:1), UL_trainloader(bs:1)
            nb_mc = 10
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            '''
            plt.figure(figsize=(18, 18))
            # for idx in range(3):
            plt.subplot(2, 1, 1)
            plt.imshow(volume_batch[0][0][:, :,100:101].detach().cpu().numpy(), cmap='gray')
            plt.subplot(2, 1, 2)
            plt.imshow(label_batch[0][0][:, :,100:101].detach().cpu().numpy(), cmap='gray')

            plt.tight_layout()
            plt.show()
            print()
            '''


            #noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            #volume_batch = volume_batch + noise
            
            # 학습을 진행하면서 스케줄러에 따라 가중치 조절
#             scheduler.step()
            
            # 학습률 업데이트
#             scheduler.step()
            
            # 가중치 업데이트
#             loss_weight = weight_scheduler(epoch_num)
#             loss_weight = min(1, loss_weight)

            outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4, log_alea_aux1, log_alea_aux2, log_alea_aux3, log_alea_aux4  = model(volume_batch)
            mu = torch.stack((outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4),dim=-1)
            epistemic = torch.var(mu, dim=-1)
            mu = torch.mean(mu, dim=-1)
            aleatoric = torch.stack((torch.exp(log_alea_aux1), torch.exp(log_alea_aux2), torch.exp(log_alea_aux3), torch.exp(log_alea_aux4)),dim=-1)
            aleatoric = torch.mean(aleatoric, dim=-1)
            noise = torch.randn(batch_size, 2, 256, 256, 128, nb_mc).cuda()
            reparameterized_output = (mu.unsqueeze(-1).repeat(1, 1, 1, 1, 1, nb_mc)) + noise * (aleatoric.unsqueeze(-1).repeat(1, 1, 1, 1, 1, nb_mc))
            y_tru = label_batch.unsqueeze(-1).repeat(1, 1, 1, 1, 1, nb_mc)
            mc_x = ce_loss(reparameterized_output, y_tru.squeeze(1).long())
            mc_x = torch.mean(mc_x, dim=-1)
            attenuated_ce_loss = torch.mean(mc_x)
            outputs_soft = torch.softmax(mu, dim=1)
            loss = attenuated_ce_loss + dice_loss(outputs_soft, label_batch.float()) + torch.mean(epistemic)
#             outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
#             outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
#             outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
#             outputs_aux4_soft = torch.softmax(outputs_aux4, dim=1)
#             loss_ce_aux1 = ce_loss(outputs_aux1,
#                                    label_batch.squeeze(1).long())
#             loss_ce_aux2 = ce_loss(outputs_aux2,
#                                    label_batch.squeeze(1).long())
#             loss_ce_aux3 = ce_loss(outputs_aux3,
#                                    label_batch.squeeze(1).long())
#             loss_ce_aux4 = ce_loss(outputs_aux4,
#                                    label_batch.squeeze(1).long())

#             loss_dice_aux1 = dice_loss(
#                 outputs_aux1_soft, label_batch.float())
#             loss_dice_aux2 = dice_loss(
#                 outputs_aux2_soft, label_batch.float())
#             loss_dice_aux3 = dice_loss(
#                 outputs_aux3_soft, label_batch.float())
#             loss_dice_aux4 = dice_loss(
#                 outputs_aux4_soft, label_batch.float())

#             loss = (loss_ce_aux1+loss_ce_aux2+loss_ce_aux3+loss_ce_aux4 +
#                                loss_dice_aux1+loss_dice_aux2+loss_dice_aux3+loss_dice_aux4)/8
#             loss = ce_loss(output, label_batch.squeeze(1).long()) + dice_loss(output_soft, label_batch, weight=class_weights)


            optimizer1.zero_grad()
            loss.backward()

            optimizer1.step()

            iter_num = iter_num + 1

            
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/supervised_loss',
                              loss.item(), iter_num)  # .data.item()이랑 .item()이랑 비교해보기

#             wandb.log({
#                 "iter": iter_num,
#                 "total_loss": loss.item(),  # total loss
#                 "supervised_loss": loss.item(),


#             })

            logging.info('iteration %d : supervised_loss : %f'  % (
                iter_num, loss.item()))


            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_



            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric1 = test_all_case(
                    model, val_loader=val_loader, num_classes=num_classes, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64, model_name=args.model)
                if avg_metric1[0][0] > best_performance1:
                    best_performance1 = avg_metric1[0][0]
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/model_val_dice_score',
                                  avg_metric1[0][0],  iter_num)
                writer.add_scalar('info/model_val_hd95',
                                  avg_metric1[0][1],  iter_num)
                logging.info(
                    'iteration %d : model_dice_score : %f model_hd95 : %f ' % (
                        iter_num, avg_metric1[0][0], avg_metric1[0][1]))
                model.train()




            if iter_num % 2500 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model_iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))


            if iter_num >= max_iterations:
                break
            time1 = time.time()
            # Release GPU memory
            torch.cuda.empty_cache()
            
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":

    if args.class_name == 1:
        snapshot_path = "/data/hanyang_Prostate/Prostate/prostate_1c_train_result/{}/{}_{}_fold{}".format(args.exp, args.model,args.max_iterations,args.fold)
    elif args.class_name == 2:
        snapshot_path = "/data/hanyang_Prostate/Prostate/TZ_1c_train_result/{}/{}_{}_fold{}".format(args.exp, args.model,args.max_iterations,args.fold)
    elif args.class_name == -1:
        snapshot_path = "/data/hanyang_Prostate/Prostate/multi_train_result/{}/{}_{}_fold{}".format(args.exp, args.model, args.max_iterations, args.fold)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    if os.path.exists(snapshot_path + '/code/prostate'):
        shutil.rmtree(snapshot_path + '/code/prostate')
    shutil.copytree('../prostate', snapshot_path + '/code/prostate',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)


