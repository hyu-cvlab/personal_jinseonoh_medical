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
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    Spacingd,
    RandRotate90d,
    CenterSpatialCropd,
    RandSpatialCropd,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    RandFlip,
    Spacing,
    RandRotate90,
    CenterSpatialCrop,
    RandSpatialCrop
)
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)
from itertools import cycle

# from dataloaders import utils
# from dataloaders.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
#                              RandomRotFlip, ToTensor,
#                              TwoStreamBatchSampler)
from networks.unet_3D_dv_semi import unet_3D_dv_semi
# from utils import losses, metrics, ramps
from utils import losses, ramps
# from val_urpc_util import test_all_case
from MSDP_val_3D import test_all_case


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/hanyang_Prostate/50_example/trim/ssl_data/centerCrop_200', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='GTV/uncertainty_rectified_pyramid_consistency', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D_dv_semi', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--class_name', type=int,  default=1)

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=40,
                    help='labeled data')
parser.add_argument('--total_labeled_num', type=int, default=290,
                    help='total labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=400.0, help='consistency_rampup')
parser.add_argument('--gpu', type=str,  default='2,3', help='GPU to use')
# parser.add_argument('--T', type=int,  default=8, help='ATD iter')
parser.add_argument('--fold', type=int,  default=None, help='k fold cross validation')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    num_classes = args.num_classes
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    fold = args.fold

    net = unet_3D_dv_semi(n_classes=num_classes, in_channels=1)
    
    if len(args.gpu.split(',')) > 1:
        model = nn.DataParallel(net).to(device)
#         ema_model = nn.DataParallel(ema_model).to(device)
    else :
        model = net.cuda()
#         model = model.cuda()
#         ema_model = ema_model.cuda()

#     db_train = BraTS2019(base_dir=train_data_path,
#                    split='train',
#                    num=None,
#                    transform=transforms.Compose([
#                        RandomRotFlip(),
#                        RandomCrop(args.patch_size),
#                        ToTensor(),
#                    ]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),  # ALI
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8, 0.8, 0.8),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128),random_size=False),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.20,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.30,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.40,
                max_k=3,
            ),

        ]
    )
    ul_train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAI"),  # ALI
            Spacingd(
                keys=["image"],
                pixdim=(0.8, 0.8, 0.8),
                mode=("bilinear"),              # mode가 sequence -> key 순서대로
            ),
            RandSpatialCropd(keys=['image'], roi_size=(256,256,128), random_size=False),
            RandFlipd(
                keys=["image"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image"],
                spatial_axis=[1],
                prob=0.20,
            ),
            RandFlipd(
                keys=["image"],
                spatial_axis=[2],
                prob=0.30,
            ),
            RandRotate90d(
                keys=["image"],
                prob=0.40,
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
                pixdim=(0.8, 0.8, 0.8),
                mode=("bilinear", "nearest"),
            )
        ]
    )
    
    if args.class_name == 1:
        datasets = args.root_path + "/dataset_fold{}.json".format(fold)
        ul_datasets = args.root_path + "/dataset_unlabeled.json"
        print("total_prostate train : dataset.json")
    if args.class_name == 2:
        datasets = args.root_path + "/dataset_2_fold{}.json".format(fold)
        ul_datasets = args.root_path + "/dataset_unlabeled.json"
        print("transition zone train :dataset_2.json")
    train_files = load_decathlon_datalist(datasets, True, "training")
    ul_train_files = load_decathlon_datalist(ul_datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "test")

    ########## label train dataload
    db_train_SL = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )

    SL_trainloader = DataLoader(db_train_SL, batch_size=args.batch_size-args.labeled_bs, shuffle=False,
                                num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)  ## 40개 안에서 shuffle

    ########## unlabel train dataload
    db_train_UL = CacheDataset(
        data=ul_train_files,
        transform=ul_train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )

    UL_trainloader = DataLoader(db_train_UL, batch_size=args.batch_size-args.labeled_bs, shuffle=False,
                                num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)  ## 40개 안에서 shuffle

    SL, UL = (cycle(SL_trainloader), UL_trainloader)

    ##########val dataload
    db_val = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_loader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )



#     labeled_idxs = list(range(0, args.labeled_num))
#     unlabeled_idxs = list(range(args.labeled_num, args.total_labeled_num))
#     batch_sampler = TwoStreamBatchSampler(
#         labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

#     trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
#                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(SL_trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(SL_trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kl_distance = nn.KLDivLoss(reduction='none')
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(zip(SL,UL)):

#             volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
#             unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            volume_batch, label_batch = sampled_batch[0]['image'].cuda(), sampled_batch[0]['label'].cuda()
            unlabeled_volume_batch = sampled_batch[1]['image'].cuda() ##unlabel
            volume_batch =  torch.cat((volume_batch,unlabeled_volume_batch),0)

            outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4,  = model(
                volume_batch)
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
            outputs_aux4_soft = torch.softmax(outputs_aux4, dim=1)

            loss_ce_aux1 = ce_loss(outputs_aux1[:args.labeled_bs],
                                   label_batch.squeeze(1).long())
            loss_ce_aux2 = ce_loss(outputs_aux2[:args.labeled_bs],
                                   label_batch.squeeze(1).long())
            loss_ce_aux3 = ce_loss(outputs_aux3[:args.labeled_bs],
                                   label_batch.squeeze(1).long())
            loss_ce_aux4 = ce_loss(outputs_aux4[:args.labeled_bs],
                                   label_batch.squeeze(1).long())

            loss_dice_aux1 = dice_loss(
                outputs_aux1_soft[:args.labeled_bs], label_batch[:args.labeled_bs].float())
            loss_dice_aux2 = dice_loss(
                outputs_aux2_soft[:args.labeled_bs], label_batch[:args.labeled_bs].float())
            loss_dice_aux3 = dice_loss(
                outputs_aux3_soft[:args.labeled_bs], label_batch[:args.labeled_bs].float())
            loss_dice_aux4 = dice_loss(
                outputs_aux4_soft[:args.labeled_bs], label_batch[:args.labeled_bs].float())

            supervised_loss = (loss_ce_aux1+loss_ce_aux2+loss_ce_aux3+loss_ce_aux4 +
                               loss_dice_aux1+loss_dice_aux2+loss_dice_aux3+loss_dice_aux4)/8

            preds = (outputs_aux1_soft +
                     outputs_aux2_soft+outputs_aux3_soft+outputs_aux4_soft)/4

            variance_aux1 = torch.sum(kl_distance(
                torch.log(outputs_aux1_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux1 = torch.exp(-variance_aux1)

            variance_aux2 = torch.sum(kl_distance(
                torch.log(outputs_aux2_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(kl_distance(
                torch.log(outputs_aux3_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            variance_aux4 = torch.sum(kl_distance(
                torch.log(outputs_aux4_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux4 = torch.exp(-variance_aux4)

            consistency_weight = get_current_consistency_weight(iter_num//150)

            consistency_dist_aux1 = (
                preds[args.labeled_bs:] - outputs_aux1_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux1 = torch.mean(
                consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

            consistency_dist_aux2 = (
                preds[args.labeled_bs:] - outputs_aux2_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux2 = torch.mean(
                consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

            consistency_dist_aux3 = (
                preds[args.labeled_bs:] - outputs_aux3_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux3 = torch.mean(
                consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

            consistency_dist_aux4 = (
                preds[args.labeled_bs:] - outputs_aux4_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux4 = torch.mean(
                consistency_dist_aux4 * exp_variance_aux4) / (torch.mean(exp_variance_aux4) + 1e-8) + torch.mean(variance_aux4)

            consistency_loss = (consistency_loss_aux1 +
                                consistency_loss_aux2 + consistency_loss_aux3 + consistency_loss_aux4) / 4
            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/supervised_loss',
                              supervised_loss.item(), iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss.item(), iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, supervised_loss: %f' %
                (iter_num, loss.item(), supervised_loss.item()))

#             if iter_num % 20 == 0:
#                 image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
#                     3, 0, 1, 2).repeat(1, 3, 1, 1)
#                 grid_image = make_grid(image, 5, normalize=True)
#                 writer.add_image('train/Image', grid_image, iter_num)

#                 image = torch.argmax(outputs_aux1_soft, dim=1, keepdim=True)[0, 0:1, :, :, 20:61:10].permute(
#                     3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
#                 grid_image = make_grid(image, 5, normalize=False)
#                 writer.add_image('train/Predicted_label',
#                                  grid_image, iter_num)

#                 image = label_batch[0, :, :, 20:61:10].unsqueeze(
#                     0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
#                 grid_image = make_grid(image, 5, normalize=False)
#                 writer.add_image('train/Groundtruth_label',
#                                  grid_image, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, val_loader, num_classes=num_classes, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64, model_name=args.model)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                for cls in range(1, num_classes):
                    writer.add_scalar('info/val_cls_{}_dice_score'.format(cls),
                                      avg_metric[cls - 1, 0], iter_num)
                    writer.add_scalar('info/val_cls_{}_hd95'.format(cls),
                                      avg_metric[cls - 1, 1], iter_num)
                writer.add_scalar('info/val_mean_dice_score',
                                  avg_metric[:, 0].mean(), iter_num)
                writer.add_scalar('info/val_mean_hd95',
                                  avg_metric[:, 1].mean(), iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (
                        iter_num, avg_metric[:, 0].mean(), avg_metric[:, 1].mean()))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.class_name == 1:
        snapshot_path = "/data/hanyang_Prostate/Prostate/prostate_1c_train_result/{}/{}_{}_fold{}".format(args.exp, args.model,
                                                                                        args.max_iterations,args.fold)
    elif args.class_name == 2:
        snapshot_path = "/data/sohui/Prostate/TZ_1c_train_result/{}/{}_{}_fold{}".format(args.exp, args.model,
                                                                                  args.max_iterations,args.fold)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('..', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
