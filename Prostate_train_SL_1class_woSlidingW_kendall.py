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
from networks.vnet_kendall import VNet
# from networks.attention_unet_2d import AttU_Net
from networks.attention_unet import Attention_UNet
from monai.networks.nets import UNETR
# from networks.attention_unet import Attention_UNet
# from monai.networks.nets import AttentionUnet
from utils import ramps, losses
from MSDP_val_3D_kendall import test_all_case
from skimage import segmentation as skimage_seg
# from pytorch3dunet.unet3d.losses import HausdorffDistanceDiceLoss
# from monai.losses.hausdorff_loss import HausdorffDTLoss
# from torch.optim import lr_scheduler

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
                    default='/data/hanyang_Prostate/50_example/trim/sl_data_wo_norm/centerCrop_350_350_200', help='Name of Experiment')
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
parser.add_argument('--base_lr', type=float,  default=0.001,
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
parser.add_argument('--nb_mc', type=int, default=10)

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
    nb_mc = args.nb_mc

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
    if len(args.gpu.split(',')) > 1:
        model = nn.DataParallel(model).to(device)
    else :
        model = model.cuda()

    model.train()


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)



    train_transforms = Compose(
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
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),

        ]
    )
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


    ##########train dataload
    db_train_SL = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )


    SL_trainloader = DataLoader(db_train_SL, batch_size=args.batch_size,shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)  ## 40개 안에서 shuffle



    ##########val dataload
    db_val = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_loader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    

    optimizer1 = optim.SGD(model.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    # 학습률 스케줄러 설정
#     scheduler = lr_scheduler.StepLR(optimizer1, step_size=3000, gamma=0.1)
    # 가중치를 동적으로 조절할 스케줄러를 생성
#     scheduler = lr_scheduler.LambdaLR(optimizer1, lr_lambda=weight_scheduler)
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

    iter_num = 0
    max_epoch = max_iterations // len(SL_trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kl_distance = nn.KLDivLoss(reduction='none')
    lr_ = base_lr
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(SL_trainloader):       #0,1 : SL_trainloader(bs:1), UL_trainloader(bs:1)
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

#             output = model(volume_batch)        # 2,1,176,176,176 -> 2,2,176,176,176
#             output_soft = torch.softmax(output, dim=1)
            mu, log_var = model(volume_batch)        # 2,1,176,176,176 -> 2,2,176,176,176
            
            
            mu_mc = mu.unsqueeze(-1).repeat(1, 1, 1, 1, 1, nb_mc)
            std_mc = torch.exp(log_var).unsqueeze(-1).repeat(1, 1, 1, 1, 1, nb_mc)

#             mu_mc = mu[:,1,...].unsqueeze(-1).repeat(1, 1, 1, 1, 1, nb_mc)
#             std = torch.exp(log_var)
            #print(std.shape,"chch")
            #print(std[:,1,...].unsqueeze(-1).repeat(1,1,1,1,nb_mc).shape)
            
            # Hard coded the known shape of the data
            noise = torch.randn(batch_size, 2, 256, 256, 128, nb_mc).cuda()

            reparameterized_output = mu_mc + noise * std_mc

#            print('ccc', reparameterized_output.shape)
#            print(label_batch.squeeze(1).unsqueeze(-1).shape)
#             print("label_batch.shape:",label_batch.shape) # label_batch.shape: torch.Size([2, 1, 256, 256, 128])
            
#             print("reparameterized_output.shape:",reparameterized_output.shape) # torch.Size([2, 2, 256, 256, 128, 10])

            y_tru = label_batch.unsqueeze(-1).repeat(1, 1, 1, 1, 1, nb_mc)
#             print("y_tru.shape:",y_tru.shape) # label_batch.shape: torch.Size([2, 1, 256, 256, 128])
#            print(y_tru.shape)
            mc_x = ce_loss(reparameterized_output, y_tru.squeeze(1).long())
            # Mean across mc samples
            mc_x = torch.mean(mc_x, dim=-1)
            # Mean across everything else
            attenuated_ce_loss = torch.mean(mc_x)
#             output_soft = torch.softmax(reparameterized_output, dim=1)  # dice loss에 영향을 줌
            output_soft = torch.softmax(mu, dim=1)
#             print("mu.shape:",mu.shape)
#             print("y_tru.shape:",y_tru.shape)
#             print("output_soft.shape:",output_soft.shape)
#             print("label_batch.shape:",label_batch.shape)
            
#             print('hd_loss:', hd_loss(output_soft[:, 1, ...].unsqueeze(1), label_batch))
            ##supervised :dice CE
#             loss = ce_loss(output, label_batch.squeeze(1).long()) + dice_loss(output_soft, label_batch, weight=class_weights) + (0.1 * hd_loss(output_soft[:, 1, ...].unsqueeze(1), label_batch))
            
#             loss = ce_loss(output, label_batch.squeeze(1).long()) + dice_loss(output_soft, label_batch, weight=class_weights)
#             loss = attenuated_ce_loss + dice_loss(output_soft, y_tru, weight=class_weights)  # dice loss에 영향을 줌
            loss = attenuated_ce_loss + dice_loss(output_soft, label_batch, weight=class_weights)
            # cross-entropy 는 실제 값과 예측값의 차이 (dissimilarity) 를 계산
#             print("loss", loss.shape, loss)
#             print("original loss", (ce_loss(output, label_batch.squeeze(1).long()) + dice_loss(output_soft, label_batch, weight=class_weights)).shape, ce_loss(output, label_batch.squeeze(1).long()) + dice_loss(output_soft, label_batch, weight=class_weights))


            optimizer1.zero_grad()
            loss.backward()

            optimizer1.step()

            iter_num = iter_num + 1

            
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/supervised_loss',
                              loss.data.item(), iter_num)  # .data.item()이랑 .item()이랑 비교해보기

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
