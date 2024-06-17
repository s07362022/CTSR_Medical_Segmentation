# -*- coding: utf-8 -*-

import argparse
import os
import random
from copy import deepcopy
from typing import Optional

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (AverageMeter, color_map, count_params, init_log,
                   intersectionAndUnion, meanIOU)

# from model.semseg.resunet import build_resunetplusplus
from core.res_unet_plus import ResUnetPlusPlus
from core.unet import UNet, UNetSmall
from dataset.semi_cv import SemiDataset  # semi4
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from nets.unet import Unet
from nets.unet_training import (CE_Loss, ContrastiveLoss, Dice_loss,
                                Focal_Loss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils2 import ramps
from utils2.utils import (download_weights, seed_everything, show_config,
                          worker_init_fn)
# from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
#

import torch.backends.cudnn as cudnn

cudnn.benchmark =  True#False
cudnn.deterministic = False

seedx = 2025
random.seed(seedx)
np.random.seed(seedx)
torch.manual_seed(seedx)
torch.cuda.manual_seed(seedx)
local_rank = 0
device = 'cuda'

MODE = None

# 

def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings 
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes','ki67','nuc'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    # 新增這裡
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2','resunt','unet','unetpp'],
                        default='deeplabv3plus')
    # 為止
    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)

    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true',
                        help='whether to use ST++')

    args = parser.parse_args()
    return args

def replace_nan_with_zero(tensor):
    has_nan = torch.isnan(tensor)
    tensor[has_nan] = 0.0
    return tensor



def main(args):
    
    #################### wandb ####################
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="semi-ST_Center",
    #     name="sup_medical_immune",#final_allstage_CE_consistency_stage1cam_with_not3-inittmodel_ce/2_center_lossmse_731 
    #     tags=["label_train", "CE_consistency"],
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": args.lr,
    #     "architecture": "unet",
    #     "dataset": args.dataset,
    #     "epochs": args.epochs,
    #     }
    # )
    
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

        
        
    contrastive_=ContrastiveLoss()
    Loss_Tversky_B = TverskyLoss(0.1, 0.9).to(device)  ##
    criterion = CrossEntropyLoss(ignore_index=0)#ignore_index=255

    valset = SemiDataset(args.dataset, args.data_root, 'val',None, args.labeled_id_path)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
#     testset = SemiDataset(args.dataset, args.data_root, 'test',None, None)
#     testloader = DataLoader(testset, batch_size=4 if args.dataset == 'cityscapes' else 1,
#                            shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    
#     testset = SemiDataset(args.dataset, args.data_root, 'inf',None, None)
#     testloader = DataLoader(testset, batch_size=4 if args.dataset == 'cityscapes' else 1,
#                            shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    print('testloader',len(valloader))
    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))
    
    
    ############ Opinion ##################
    
    focal_loss = True
    
    #######################################
    
    print("\n=======Opinion=========")
    
    print("\nfocal_loss :", focal_loss)
    
    print("\nconsistency_loss :", True)
    
    print("\n Using Methods : D + C + F +A + average ")
    
    print("\n======================")
    
    MT = "ex19"
    
    
    global MODE 
    MODE = 'train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)
    
    model, optimizer = init_basic_elems(args)
    # train here
  
    best_model, checkpoints= train(model, trainloader, valloader, criterion,Loss_Tversky_B, optimizer,focal_loss,MT, args) #valloader
    
    # inference here
#     MODE = 'inf'
#     dataset = SemiDataset(args.dataset, args.data_root, 'inf', None, None, args.unlabeled_id_path)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
#     label(model, dataloader, args)
    
#     MODE = 'test'
#     label(model, testloader, args)
#     return
#     best_model = model
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')
    
    MODE = 'inf'
    
    dataset = SemiDataset(args.dataset, args.data_root, 'inf', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    select_reliable3(best_model, dataloader, args,checkpoints)
    
    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')
    
    MODE = 'inf'
    
    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'inf', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)
    
    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')

    MODE = 'semi_train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('retrain reliable data')
    MT = "ex19_2"
     
    best_model, checkpoints  = train(model, trainloader, valloader, criterion,Loss_Tversky_B, optimizer,focal_loss,MT, args) #valloader best_model
    
    
    
    print('\n\n\n================> Total stage 5/6: Select reliable images for the 1st stage re-training')
    
    MODE = 'inf'
    dataset = SemiDataset(args.dataset, args.data_root, 'inf', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # select_reliable(checkpoints, dataloader, args)
    select_reliable3(best_model, dataloader, args,checkpoints)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 5-2/6: Pseudo labeling reliable images')

#     cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
#     dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    
#     label(best_model, dataloader, args)
    dataset = SemiDataset(args.dataset, args.data_root, 'inf', None,None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 1st stage re-training on labeled and reliable unlabeled images')

    MODE = 'final_train'
    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)
    
    model, optimizer = init_basic_elems(args)
    MODE = 'final_train'
    print('full')
    MT = "ex19_3"
    best_model = train(model, trainloader, valloader, criterion,Loss_Tversky_B, optimizer,focal_loss,MT, args)
    
    # inference here
    MODE = 'test'
    label(best_model, valloader, args)
    # MODE = 'inf'
    # dataset = SemiDataset(args.dataset, args.data_root, 'inf', None, None, args.unlabeled_id_path)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    # label(model, dataloader, args)
    return
    
    
    
  
    

def init_basic_elems(args):
    # model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2,'resunt':ResUnetPlusPlus ,'unet':UNetSmall}#build_resunetplusplus}
    # model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    
    if args.model == 'resunt':
        model = ResUnetPlusPlus(3,2)
        # optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(),  lr=args.lr, weight_decay=1e-4)
    elif args.model == 'unetpp':
        import segmentation_models_pytorch as smp
#         aux_params=dict(
#             pooling='avg',             # one of 'avg', 'max'
#             dropout=0.1,               # dropout ratio, default is None
#             activation='softmax',      # activation function, default is None
#             classes=3,                 # define number of output labels
            
#         )
        model =  smp.UnetPlusPlus (
                        encoder_name = 'resnext50_32x4d',
                        encoder_weights="ssl",
                        classes = 3,
                        )
#         model = smp.UnetPlusPlus(
#                 encoder_name = 'resnext50_32x4d',
#                 encoder_weights="ssl",
#                 classes = 3,
#                 ) #resnext101_32x8d resnet18 encoder_weights="ssl", 
#         model = smp.MAnet(
#                 encoder_name = 'resnext50_32x4d',
#                 classes = 3,)
        model.load_state_dict(torch.load('/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex17/unetpp_resnet50_83.83_RS101_2_good.pth'))
        print('use UnetPlusPlus')
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            
    elif args.model == 'unet2':
        model = UNetSmall(2)
        # optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
        optimizer = SGD(  model.parameters(),  lr=args.lr, momentum=0.9, weight_decay=1e-4)
        
    elif args.model == 'unet':
        backbone = 'resnet50'
        # download_weights(backbone)
        pretrained  = True
#         model_path = "/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex11/unet_resnet50_90.31_DFCAS_border.pth"
        model_path = 'unet_resnet_medical.pth'#"unet_resnet50_83.53.pth"
        model = Unet(num_classes=3, pretrained=pretrained, backbone=backbone).train()
        if not pretrained:
            weights_init(model)
        if model_path != '':
            if local_rank == 0:
                print('Load weights {}.'.format(model_path))

   
            model_dict      = model.state_dict()
            pretrained_dict = torch.load(model_path, map_location = device)
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)
            print('load weight')
        
        
        optimizer = torch.optim.Adam(model.parameters(),  lr=args.lr, weight_decay=1e-8)
        
        
    else :
        model = model_zoo[args.model](args.backbone,  2)
        print('model :',args.model)
        head_lr_multiple = 10.0
        if args.model == 'deeplabv2':
            assert args.backbone == 'resnet101'
            model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
            head_lr_multiple = 1.0 
        # print(args.lr)
        # if stage ==34 :
        
        
        optimizer = torch.optim.Adam(model.parameters(),  lr=args.lr, weight_decay=1e-8)
        # else:
            # optimizer = SGD(  model.parameters(),  lr=args.lr, momentum=0.9, weight_decay=1e-4)
#             optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
#                              {'params': [param for name, param in model.named_parameters()
#                                          if 'backbone' not in name],
#                               'lr': args.lr * head_lr_multiple}],
#                             lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    if torch.cuda.device_count() > 1:
        cudnn.benchmark = True
        print(f"using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model).cuda()
        
    else:
        cudnn.benchmark = True
        model = model.cuda()

    return model, optimizer

def consistency_f(input_pre,target,batch_size):
    cosine_dist = 1 - F.cosine_similarity(input_pre, target, dim=1)
    loss =  cosine_dist.mean()
    loss= replace_nan_with_zero(loss)
    return loss


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1 * ramps.sigmoid_rampup(epoch, 100)
def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


fea_all_w = []

def train(model, trainloader, valloader, criterion,Loss_Tversky_B, optimizer,focal_loss,MT,  args):
    contrastive_=ContrastiveLoss()
    kl_distance = nn.KLDivLoss(reduction='none')
    iters = 0
    total_iters = len(trainloader) * args.epochs
    

    
    cls_weights     = np.array([0.9,1.2,1.6], np.float32)
    weights = torch.from_numpy(cls_weights)
    weights = weights.cuda()
    print("focal weights",weights)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    num_classes = 3
    
        

    
    
    previous_best = 0.0

    global MODE

    if MODE == 'train' :
        print('train stage 1 ')
        checkpoints = []
        
        # tbar = tqdm(trainloader)
        epochs_max = int(args.epochs *1)
        for epoch in range(epochs_max): #(args.epochs):
            # set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            print("\n==> Epoch %i, learning rate = %.6f\t\t\t\t\t previous best = %.2f" %
                    (epoch, optimizer.param_groups[0]["lr"], previous_best))

            model.train()
            total_loss = 0.0
            total_consistency_loss=0.0
            
            tbar = tqdm(trainloader)
            for i, (img, mask, img2, mask2) in enumerate(tbar):#, border,border2
                
#                 print(np.unique(mask))
                optimizer.zero_grad()
                
                
                img, mask = img.cuda(), mask.cuda()
                
                img2, mask2 = img2.cuda(), mask2.cuda()
                
                # border, border2 = border.cuda(), border2.cuda() # wb,sb
                
                
                
                # Dice loss 1 
                pred = model(img) #border_w ,need_fp=True
                predx = torch.softmax(pred, dim=1)

                
                loss_di1=Dice_loss(predx, mask, num_classes, smooth = 1e-5)
                

                # border_w=torch.softmax(border_w, dim=1)
                
                
                # loss_t_b1 = Loss_Tversky_B(border_w, border) ## 
                
                # Dice loss 1 
                pred2= model(img2) #,need_fp=True
                pred2x=torch.softmax(pred2, dim=1)
                # border_s=torch.softmax(border_s, dim=1)

                loss_di2=Dice_loss(pred2x, mask2, num_classes, smooth = 1e-5)
                
                
               
                # loss_t_b2 = Loss_Tversky_B(border_s , border2) ## 
                
                dice_loss = loss_di1*0.5 + loss_di2*0.5 
                # t_loss = loss_t_b1*0.5 + loss_t_b2*0.5
                
                if focal_loss:
                    f_loss1 = Focal_Loss(pred, mask, weights, num_classes = num_classes) 
                    f_loss2 = Focal_Loss(pred2, mask2, weights, num_classes = num_classes) 
                    f_loss = (f_loss1+f_loss2)/2
                    
                    
                    total_loss = dice_loss + f_loss #+ t_loss
                    
                consistency_loss,conf_mask= consistency_criterion_loss( pred2,pred.detach(),0.8, 3,epoch)
                # variance_1 = torch.sum(kl_distance(torch.log(pred2x), predx), dim=1, keepdim=True)# 只是用来计算kl，固定操作，多加一个log
                # exp_variance_1 = torch.exp(-variance_1)
                # consis_dist_1 = (predx - pred2x) ** 3
                # loss_consist = consis_loss_1 = torch.mean(consis_dist_1 * exp_variance_1) / (torch.mean(exp_variance_1) + 1e-8) + torch.mean(variance_1)
                
                
                # total_loss = loss_consist_v1 +loss_consist_v2 + supervised_loss
                total_loss = total_loss +  consistency_loss*0.5# + Self_labeling_loss*0.1 dice_loss

                
                total_loss.backward()
                optimizer.step()



                iters += 1
                lr = args.lr * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr  

                tbar.set_description('Loss: %.3f , dice_loss: %.3f, focal_loss: %.3f, consist: %.3f' % (total_loss,dice_loss,f_loss,consistency_loss)) #dice_loss , t_loss: %.3f
                
                total_loss_mean  = (total_loss / (i + 1))
                
            # wandb.log({"{}_dice_loss".format(times): dice_loss, "{}_total_Loss".format(times): total_loss_mean, "{}_learning_rate".format(times) : lr})        
            metric = meanIOU(num_classes)

            model.eval()
            tbar2 = tqdm(valloader)

            with torch.no_grad():
#                 eval_mode = 'sliding_window' if args.dataset == 'cityscapes' else 'original'
#                 mIOU, iou_class = evaluate(model, valloader, eval_mode, args)
                
                # print('mIOU: %.2f' % (mIOU))
                for img, mask,_ in tbar2:
                    img = img.cuda()
                    pred = model(img) #,need_fp=True
                    pred = torch.argmax(pred, dim=1)
                    # print(pred.shape , mask.shape)
                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    mIOU = metric.evaluate()[-1]

                    tbar2.set_description('mIOU: %.2f' % (mIOU* 100.0 ))#* 100.0
#             print(iou_class)

            mIOU *= 100.0

            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(os.path.join(args.save_path, '%s_%s_%.2f_%s.pth' % (args.model, args.backbone, previous_best,MT)))
                previous_best = mIOU
                if len(checkpoints)==4:
                    checkpoints.pop(0)
                else:
                    checkpoints.append(deepcopy(model))
                        
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(),
                                os.path.join(args.save_path, '%s_%s_%.2f_%s.pth' % (args.model, args.backbone, mIOU,MT)))
                else:
                    torch.save(model.state_dict(),
                                os.path.join(args.save_path, '%s_%s_%.2f_%s.pth' % (args.model, args.backbone, mIOU,MT)))

                best_model = deepcopy(model)
            else:
                best_model = deepcopy(model)

            
            # wandb.log({"{}_val_mIOU".format(times): mIOU})
    if MODE == 'semi_train' :
        print('train stage 2 ')
        checkpoints = []
        num_classes = 3
        
        # tbar = tqdm(trainloader)
          
       
        epochs_max = int(args.epochs *1.0)
        for epoch in range(epochs_max): #(args.epochs):
            # set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            print("\n==> Epoch %i, learning rate = %.6f\t\t\t\t\t previous best = %.2f" %
                    (epoch, optimizer.param_groups[0]["lr"], previous_best))
            model.cuda()
            model.train()
            total_loss = 0.0
            total_consistency_loss=0.0
            
            tbar = tqdm(trainloader)
            for i, (img, mask, img2, mask2) in enumerate(tbar):
                
                optimizer.zero_grad()
                
#                 print('mask',np.unique(mask))
#                 print('mask2',np.unique(mask2))
                
                
                img, mask = img.cuda(), mask.cuda()
                
                img2, mask2 = img2.cuda(), mask2.cuda()
                
                # border, border2 = border.cuda(), border2.cuda() # wb,sb
                
                
                
                # Dice loss 1 
                pred = model(img) #border_w ,need_fp=True
                predx = torch.softmax(pred, dim=1)

                
                loss_di1=Dice_loss(predx, mask, num_classes, smooth = 1e-5)
                

                # border_w=torch.softmax(border_w, dim=1)
                
                
                # loss_t_b1 = Loss_Tversky_B(border_w, border) ## 
                
                # Dice loss 1 
                pred2= model(img2) #,need_fp=True
                pred2x=torch.softmax(pred2, dim=1)
                # border_s=torch.softmax(border_s, dim=1)

                loss_di2=Dice_loss(pred2x, mask2, num_classes, smooth = 1e-5)
                
                
               
                # loss_t_b2 = Loss_Tversky_B(border_s , border2) ## 
                
                dice_loss = loss_di1*0.5 + loss_di2*0.5 
                # t_loss = loss_t_b1*0.5 + loss_t_b2*0.5
                
                if focal_loss:
                    f_loss1 = Focal_Loss(pred, mask, weights, num_classes = num_classes) 
                    f_loss2 = Focal_Loss(pred2, mask2, weights, num_classes = num_classes) 
                    f_loss = (f_loss1+f_loss2)#*0.5
                    
                    
                    total_loss = dice_loss + f_loss #+ t_loss
                    
#                 consistency_loss,conf_mask= consistency_criterion_loss( pred2,pred.detach(),0.8, 3,epoch)
                # variance_1 = torch.sum(kl_distance(torch.log(pred2x), predx), dim=1, keepdim=True)# 只是用来计算kl，固定操作，多加一个log
                # exp_variance_1 = torch.exp(-variance_1)
                # consis_dist_1 = (predx - pred2x) ** 3
                # loss_consist = consis_loss_1 = torch.mean(consis_dist_1 * exp_variance_1) / (torch.mean(exp_variance_1) + 1e-8) + torch.mean(variance_1)
                
                
                # total_loss = loss_consist_v1 +loss_consist_v2 + supervised_loss
                total_loss = total_loss #+  consistency_loss# + Self_labeling_loss*0.1 dice_loss

                
                total_loss.backward()
                optimizer.step()



                iters += 1
#                 lr =  optimizer.param_groups[0]["lr"] 
                lr = args.lr * (1 - iters / total_iters) ** 0.92
                optimizer.param_groups[0]["lr"] = lr #consistency_loss,: %.3f loss_consist

                # tbar.set_description('Loss: %.3f , dice_loss: %.3f , focal_loss: %.3f' % (total_loss,dice_loss,f_loss)) #dice_loss
                tbar.set_description('Loss: %.3f  ' % (total_loss))
                
                total_loss_mean  = (total_loss / (i + 1))
                
            # wandb.log({"{}_dice_loss".format(times): dice_loss, "{}_total_Loss".format(times): total_loss_mean, "{}_learning_rate".format(times) : lr})        
            metric = meanIOU(num_classes)

            model.eval()
            tbar2 = tqdm(valloader)

            with torch.no_grad():
#                 eval_mode = 'original'#'sliding_window' if args.dataset == 'cityscapes' else 'original'
#                 mIOU, iou_class = evaluate(model, valloader, eval_mode, args)
                
                # print('mIOU: %.2f' % (mIOU))
                for img, mask,_ in tbar2:
                    img = img.cuda()
                    pred = model(img)
                    pred = torch.argmax(pred, dim=1)
                    # print(pred.shape , mask.shape)
                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    mIOU = metric.evaluate()[-1]

                    tbar2.set_description('mIOU: %.2f' % (mIOU* 100.0 ))#* 100.0
#             print(iou_class)
            # checkpoints.append(deepcopy(model))
            mIOU *= 100.0
            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(os.path.join(args.save_path, '%s_%s_%.2f_%s.pth' % (args.model, args.backbone, previous_best,MT)))
                previous_best = mIOU
                if len(checkpoints)==4:
                    checkpoints.pop(0)
                else:
                    checkpoints.append(deepcopy(model))
                        
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(),
                                os.path.join(args.save_path, '%s_%s_%.2f_%s.pth' % (args.model, args.backbone, mIOU,MT)))
                else:
                    torch.save(model.state_dict(),
                                os.path.join(args.save_path, '%s_%s_%.2f_%s.pth' % (args.model, args.backbone, mIOU,MT)))

                best_model = deepcopy(model)
            else:
                best_model = deepcopy(model)
        
    if MODE == 'final_train':
        print('final train stage ')
        checkpoints = []
        
        # tbar = tqdm(trainloader)
          
       
        epochs_max = int(args.epochs *1.0)
        for epoch in range(epochs_max): #(args.epochs):
            # set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            print("\n==> Epoch %i, learning rate = %.6f\t\t\t\t\t previous best = %.2f" %
                    (epoch, optimizer.param_groups[0]["lr"], previous_best))

            model.train()
            total_loss = 0.0
            total_consistency_loss=0.0
            
            tbar = tqdm(trainloader)
            for i, (img, mask, img2, mask2) in enumerate(tbar):
                
                optimizer.zero_grad()
                
                img, mask = img.cuda(), mask.cuda()
                
                img2, mask2 = img2.cuda(), mask2.cuda()
                
                
                # Dice loss 1 
                pred = model(img)
                predx = torch.softmax(pred, dim=1)
                # mask = np.squeeze(mask,-1)
                loss_di1=Dice_loss(predx, mask, num_classes, smooth = 1e-5)
                
                # Dice loss 1 
                pred2 = model(img2)
                pred2x=torch.softmax(pred2, dim=1)
                loss_di2=Dice_loss(pred2x, mask2, num_classes, smooth = 1e-5)
                
                dice_loss = loss_di1*0.5 + loss_di2*0.5 
                
                if focal_loss:
                    f_loss1 = Focal_Loss(pred, mask, weights, num_classes ) 
                    f_loss2 = Focal_Loss(pred2, mask2, weights, num_classes ) 
                    f_loss = (f_loss1+f_loss2)/2
                    total_loss = dice_loss + f_loss


                
                total_loss.backward()
                optimizer.step()



                iters += 1
                lr =  optimizer.param_groups[0]["lr"] 
#                 lr = args.lr * (1 - iters / total_iters) ** 0.9
#                 optimizer.param_groups[0]["lr"] = lr #consistency_loss

                tbar.set_description('Loss: %.3f , dice_loss: %.3f , focal_loss: %.3f' % (total_loss,dice_loss,f_loss)) #dice_loss
                
                total_loss_mean  = (total_loss / (i + 1))
                
            # wandb.log({"{}_dice_loss".format(times): dice_loss, "{}_total_Loss".format(times): total_loss_mean, "{}_learning_rate".format(times) : lr})        
            metric = meanIOU(num_classes)

            model.eval()
            tbar2 = tqdm(valloader)

            with torch.no_grad():
#                 eval_mode = 'sliding_window' if args.dataset == 'cityscapes' else 'original'
#                 mIOU, iou_class = evaluate(model, valloader, eval_mode, args)
                
                # print('mIOU: %.2f' % (mIOU))
                for img, mask,_ in tbar2:
                    img = img.cuda()
                    pred = model(img)
                    pred = torch.argmax(pred, dim=1)
                    # print(pred.shape , mask.shape)
                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    mIOU = metric.evaluate()[-1]

                    tbar2.set_description('mIOU: %.2f' % (mIOU* 100.0 ))#* 100.0
#             print(iou_class)

            mIOU *= 100.0
            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(os.path.join(args.save_path, '%s_%s_%.2f_%s.pth' % (args.model, args.backbone, previous_best,MT)))
                previous_best = mIOU
                if len(checkpoints)==3:
                    checkpoints.pop(0)
                else:
                    checkpoints.append(deepcopy(model))
                        
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(),
                                os.path.join(args.save_path, '%s_%s_%.2f_%s.pth' % (args.model, args.backbone, mIOU,MT)))
                else:
                    torch.save(model.state_dict(),
                                os.path.join(args.save_path, '%s_%s_%.2f_%s.pth' % (args.model, args.backbone, mIOU,MT)))

                best_model = deepcopy(model)
            else:
                best_model = deepcopy(model)
        
    
    if MODE == 'train' or MODE=='semi_train':
        return best_model, checkpoints

    return best_model

def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in tqdm(loader):
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg.crop_size
                b, _, h, w = img.shape
                final = torch.zeros(b, 3, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred, fea  = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg.crop_size) // 2, (w - cfg.crop_size) // 2
                    img = img[:, :, start_h:start_h + cfg.crop_size, start_w:start_w + cfg.crop_size]
                    mask = mask[:, start_h:start_h + cfg.crop_size, start_w:start_w + cfg.crop_size]

                pred= model(img) #,need_fp=True
                pred=pred.argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), 3, 0)#255

            intersection = torch.from_numpy(intersection)
            union = torch.from_numpy(union)
            target = torch.from_numpy(target)

            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


def loss_fn_kd(input_logits, target_logits, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = temperature
    KD_loss = nn.KLDivLoss(reduce=False, size_average=False)(F.log_softmax(input_logits/T, dim=1),
                             F.softmax(target_logits/T, dim=1))* T * T
    return KD_loss

def consistency_criterion_loss(con_logit_student, con_logit_teacher,confidence_thresh, temperature,epoch):
    '''
    Confidence-selected knowledge distillation (CS-KD)
    select high confidence pseudo labels and check the disagreement between teahcer and student model
    :param consistency_criterion: define the loss function for KD
    :param con_logit_student: the prediction of student model
    :param con_logit_teacher: the prediction of teacher model (as pseudo labels without computing gradient)
    :param confidence_thresh: pseudo label confidenc threshold
    :param step:
    :return:
    CS_KD_loss: resulting loss with confidence thresholding
    conf_mask: pseudo label mask
    label_tea: the class prediction of the teacher model
    disagree_id: the index of pseudo labels which have different prediction with high confidence
    '''

    softmax = nn.Softmax(dim=1)
    aug_loss  = loss_fn_kd(con_logit_student, con_logit_teacher, temperature)
    aug_loss = aug_loss.mean(dim=1)

    # select by confidence score
    con_logit_teacher = softmax(con_logit_teacher)
    conf_tea = torch.max(con_logit_teacher, dim=1).values
    # conf_tea = torch.max(con_logit_teacher,1)[0]
    label_tea = torch.argmax(con_logit_teacher, dim=1)
    # label_tea = torch.max(con_logit_teacher, 1)[1]

    con_logit_student = softmax(con_logit_student)
    conf_stu = torch.max(con_logit_student, dim=1).values
    # conf_stu = torch.max(con_logit_student,1)[0]
    label_stu = torch.argmax(con_logit_student, dim=1)
    # label_stu = torch.max(con_logit_student, 1)[1]

    if epoch <=5:
        confidence_thresh = 0.7
        # conf_mask  = torch.where(conf_tea > 0.2, 1, 0)
    elif 5<epoch<=10:
        confidence_thresh = 0.8
        # conf_mask  = torch.where(conf_tea > 0.4, 1, 0)
    elif 10<epoch<=20:
        confidence_thresh =0.9
        # conf_mask  = torch.where(conf_tea > 0.5, 1, 0)
    else :
        confidence_thresh = 0.8
        # conf_mask  = torch.where(conf_tea > 0.8, 1, 0)

    
#     conf_mask = (conf_tea > confidence_thresh) & (label_tea == 1)
    conf_mask = (conf_tea > confidence_thresh) 
    conf_mask = conf_mask.int()
    
#     conf_mask2 = (conf_tea > confidence_thresh) & (label_tea == 2)
#     conf_mask2 = conf_mask2.int()
    
    
    # conf_stu_mask = torch.where(conf_stu > 0.8, 1, 0)
#     conf_stu_mask = (conf_stu > confidence_thresh) & (label_tea == 1)
#     conf_stu_mask = conf_stu_mask.int()
    
    conf_mask_count  = conf_mask.sum()
#     conf_mask_count2  = conf_mask2.sum()
    # print('aug_loss',aug_loss)
    CS_KD_loss = (aug_loss * conf_mask).sum() /conf_mask_count
#     CS_KD_loss2 = (aug_loss * conf_mask2).sum() /conf_mask_count2
#     CS_KD_loss = CS_KD_loss1+CS_KD_loss2
    # print('sum(aug_loss * conf_mask)',(aug_loss * conf_mask).sum())
    return CS_KD_loss, conf_mask

from skimage.filters import sobel
def create_point_annotation2(show_range:tuple, contours:list, filter_pixel:int):  
    """
    Create point annotation from contour points by finding the gravity center
    
    Args:
        show_range(int of tuple): size of patch
        contours(list of float): all the countours
        x (int): x in top left point of patch
        y (int): y in top left point of patch
    
    Return:
        coord_list (list of dict): list of dictionary with 
                                    point & contour annotation 
    """

    coord_list = []
    
    for cnt in contours:
        # Contour area lower than setting pixels is filtered 
        if cv2.contourArea(cnt) < filter_pixel:
            continue

        M = cv2.moments(cnt)
        # if it is a zero area, contour point be tha annotaion 
        if M['m00'] == 0:
            cx = cnt[0][0][0]
            cy = cnt[0][0][1]

        # else calculate the center of gravity 
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        # make sure the annotation within patch
        if cx <= show_range[1] and cy <= show_range[0]:
            # calculate the absolute coordinate ni WSI
            cx = cx 
            cy = cy 
            contour_2_save = [[c[0][0] , c[0][1] ] for c in cnt.tolist()]
            Coordinates = dict(contour=contour_2_save, x = int(cx), y = int(cy))
            coord_list.append(Coordinates)

    return coord_list


import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
from postprocess_functions import *

from skimage.segmentation import watershed as ws
from skimage.feature import peak_local_max
from scipy.ndimage import label as lb, generate_binary_structure
from skimage.morphology import binary_dilation, disk


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes= 3)
    cmap = color_map(args.dataset)
    miou_score = []
    if  MODE == 'semi_train':
        with torch.no_grad():
            for img, mask, id in tbar:
                img = img.cuda()
                pred  = model(img)
                pred = torch.argmax(pred, dim=1).cpu()

                metric.add_batch(pred.numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]
                img2 = img.cpu().squeeze(0)
                # print(img.shape)

                # 找邊緣
                # binary_mask = Image.fromarray((pred.squeeze(0).numpy()>0.5).astype(np.uint8))
                binary_mask = (pred.cpu().permute(1, 2, 0).numpy()).astype(np.uint8) 
                gray_image = (binary_mask * (255 / 2)).astype(np.uint8) 
                
                binary_mask_cv2 = gray_image  
                contours, _ = cv2.findContours(binary_mask_cv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
               
                img2 = (img2.permute(1, 2, 0).numpy() )#.copy()
              
                result = cv2.imread(id[0].split(' ')[0])#img2
#                 print('id[0].split(' ')[0]',id[0].split(' ')[0])
                # result = cv2.imread('/work/u5914116/ki67/{}'.format(id[0].split(' ')[0]))
                
#                 cv2.drawContours(result, contours, -1, (0,255,0), 2)
#                 cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/mix_label', 'mix_'+os.path.basename(id[0].split(' ')[0])),result)
#                 cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/org_label', 'mix_'+os.path.basename(id[0].split(' ')[0])),img2)
#                 cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/mix_label', 'gray_'+os.path.basename(id[0].split(' ')[0])),gray_image)

                pred = pred.squeeze(0).numpy().astype(np.uint8)
                # pred[pred == 1] = 255
                pred = Image.fromarray(pred, mode='L')
                # pred.putpalette(cmap)
               

                pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
                miou_score.append(mIOU * 100.0)
                # print('test_img{}_miou'.format(len(miou_score)),mIOU * 100.0)
        res_miou = sum(miou_score)/(len(miou_score)+0.0005)

        # wandb.log({"testlabel_mIOU": res_miou})
    elif MODE == 'inf':
        red_count_sum = 0
        bule_count_sum = 0
        with torch.no_grad():
            for img, id in tbar:
                
                img = img.cuda()
                pred = model(img)#border
                
                
                
                pred_ = torch.argmax(pred, dim=1).cpu().numpy() 
                ##
                # border= torch.argmax(border, dim=1).cpu().numpy()

                # border_r = np.zeros_like(border, dtype=np.uint8)
                # border_r[border>=1]=255
                # print('border_r',border)
         
                
                # border_b = np.zeros_like(border_, dtype=np.uint8)
                # border_b[border_==2] =127
                # cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/border', os.path.basename(id[0].split(' ')[0])),border_r)
                # cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/border', 'b'+os.path.basename(id[0].split(' ')[0])),border_b)
                
                
                
                ##

                pred_r = np.zeros_like(pred_, dtype=np.uint8)
#                 pred_b = np.zeros_like(pred_, dtype=np.uint8)

    
                pred_r[pred_ == 1] = 1 
#                 pred_b[pred_ == 2] = 1  
                
                
                pred_r_2d = pred_r[0, :, :]

                pred = pred.cpu()
                
                pred = torch.argmax(pred, dim=1).cpu()
                pred = pred.squeeze(0).numpy().astype(np.uint8)

#                 pred2 = np.zeros_like(pred)
#                 pred3 = np.zeros_like(pred)
#                 pred3[pred == 1] = 255
#                 pred2[pred == 2] = 255
#                 cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/test_18_pseudo_r', os.path.basename(id[0].split(' ')[0])),pred3)
#                 cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/test_18_pseudo_b', os.path.basename(id[0].split(' ')[0])),pred2)

                pred = Image.fromarray(pred, mode='L')

               
                
                pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))
                
                
                
            
#             ratio_ = red_count_sum / (bule_count_sum+red_count_sum)
#             ratio_ = round(ratio_, 2)
#             print("Number of positive: {}, Number of negative: {},  ratio: {}".format(red_count_sum,bule_count_sum,ratio_))
#             re_txt = "ID: {}, Number of positive: {}, Number of negative: {},  ratio: {}".format("18-ki67",red_count_sum,bule_count_sum,ratio_)
#             with open('inf_result.txt','a') as f:
#                 f.write(re_txt + '\n')
            
                
                
                
                
    elif MODE == 'test':
        red_count_sum = 0
        bule_count_sum = 0
        with torch.no_grad():
            eval_mode = 'sliding_window' if args.dataset == 'cityscapes' else 'original'
            mIOU, iou_class = evaluate(model, dataloader, eval_mode, args)
            print('mIOU:',mIOU)
            for img, mask, id in tbar:
                image = cv2.imread(id[0].split(' ')[0])
                
                img = img.cuda()
                pred  = model(img)
                pred = torch.argmax(pred, dim=1).cpu()

                metric.add_batch(pred.numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]
                img2 = img.cpu().squeeze(0)
                
                # print(img2.shape)

                # 找邊緣
                # binary_mask = Image.fromarray((pred.squeeze(0).numpy()>0.5).astype(np.uint8))
                binary_mask = (pred.cpu().permute(1, 2, 0).numpy()).astype(np.uint8) 
                gray_image = (binary_mask * (255 / 2)).astype(np.uint8) 
                
                binary_mask_cv2 = gray_image  
            
                contours, _ = cv2.findContours(binary_mask_cv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                
                
                
                img2 = (img2.permute(1, 2, 0).numpy() ).copy()
                # print(np.unique(img2))
                
                # print(id[0].split(' ')[0])
                # result = img2
                result = cv2.imread(id[0].split(' ')[0])#
                # cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/18_ori', 'ori_'+os.path.basename(id[0].split(' ')[0])),result)
                # print(result.shape)
               

                cv2.drawContours(result, contours, -1, (0,255,0), 1)
                
                mask = (mask.permute(1, 2, 0).numpy() ).copy()
                mask_ = np.array(mask).astype(np.uint8)
                mask_ = (mask_ * (255 / 2)).astype(np.uint8) 
                # print(mask.shape)
                
                # _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
                contours2, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                cv2.drawContours(result, contours2, -1, (0,0,255), 1)
                
                # cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/18_mix', 'mix_'+os.path.basename(id[0].split(' ')[0])),result)
                
                # cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/18_mix', 'gray_'+os.path.basename(id[0].split(' ')[1])),gray_image)


                # pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='L')
                # pred.putpalette(cmap)
               
    
                # pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))
                
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
                miou_score.append(mIOU * 100.0)
                
                ############################################################################################
                # 初始化二值掩码
                pred_r = np.zeros_like(mask, dtype=np.uint8)
                pred_b = np.zeros_like(mask, dtype=np.uint8)

                # 根据类别填充掩码
                pred_r[mask == 1] = 255  # 假设类别1对应pred_r
                pred_b[mask == 2] = 255  # 假设类别2对应pred_b
                
                
                pred_r_2d = pred_r[:, :, :]
                pred_b_2d = pred_b[:, :, :]
                
                # image = cv2.imread(r'F:\nuck\data\ki67\ki67\patch512\57-F1\57-F1-54272-164352.png')
                # mask = cv2.imread(r'F:\nuck\romal_tool\pseudo57_b\57-F1-54272-164352.png', 0)
                # print(np.unique(mask))
                # 計算count跟 watershed
                color = (255,0,0)
                # print("image shape",image.shape)
                # print("pred_b_2d shape",pred_b_2d.shape)
                bule_count, image_watershed, image_remove,image_3= count_cell(image,pred_b_2d,color)
                # mask2 = cv2.imread(r'F:\nuck\romal_tool\pseudo57_r\57-F1-54272-164352.png', 0)
                color = (0,255,0)
                red_count, image_watershed2, image_remove,image_3= count_cell(image_watershed,pred_r_2d,color)
                red_count_sum +=red_count
                bule_count_sum+=bule_count
                # print(red_count_sum)
                ##
                
                contours_r, _ = cv2.findContours(pred_b_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#pred_r_2d
                contours_b, _ = cv2.findContours(pred_r_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                min_area = 200  # 设置轮廓的最小面积
                contours_r = [cnt for cnt in contours_r if cv2.contourArea(cnt) > min_area]
                contours_b = [cnt for cnt in contours_b if cv2.contourArea(cnt) > min_area]
                # img2 = (img2.permute(1, 2, 0).numpy() ).copy()
                result = cv2.imread(id[0].split(' ')[0])#
                # result2 = cv2.imread(id[0].split(' ')[0])#
                cv2.drawContours(result, contours_r, -1, (255,0,0), 2) 
                
                cv2.drawContours(result, contours_b, -1, (0,0,255), 2) 
                
                # if 1 in pred:
#                 cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/test_18_gt', os.path.basename(id[0].split(' ')[0])),result) #'red_'+
                # cv2.imwrite('%s/%s' % ('/work/u5914116/ki67/inf_image/inf17_', os.path.basename(id[0].split(' ')[0])),image_watershed2)
                # cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/inf3_', 'blue_'+os.path.basename(id[0].split(' ')[0])),result2)
                # cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/org_label', 'mix_'+os.path.basename(id[0].split(' ')[0])),img2)
                # cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/mix_label', 'gray_'+os.path.basename(id[0].split(' ')[0])),gray_image)
                # pred = pred.cpu()
                
#                 pred = torch.argmax(pred, dim=1).cpu()
#                 pred = pred.squeeze(0).numpy().astype(np.uint8)

#                 pred2 = np.zeros_like(pred)
#                 pred3 = np.zeros_like(pred)
#                 pred3[pred == 1] = 255
#                 pred2[pred == 2] = 255
#                 cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/test_18_pseudo_r', os.path.basename(id[0].split(' ')[0])),pred3)
#                 cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/test_18_pseudo_b', os.path.basename(id[0].split(' ')[0])),pred2)

#                 pred = Image.fromarray(pred, mode='L')

               
                
#                 pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))
            
            ratio_ = red_count_sum / (bule_count_sum+red_count_sum)
            ratio_ = round(ratio_, 2)
            print("Number of positive: {}, Number of negative: {},  ratio: {}".format(red_count_sum,bule_count_sum,ratio_))
#             re_txt = "ID: {}, Number of positive: {}, Number of negative: {},  ratio: {}".format("GT_18-ki67",red_count_sum,bule_count_sum,ratio_)
#             with open('inf_result.txt','a') as f:
#                 f.write(re_txt + '\n')
                
                ##############################################################
                
                # print('test_img{}_miou'.format(len(miou_score)),mIOU * 100.0)
        res_miou = sum(miou_score)/(len(miou_score)+0.0005)

        # wandb.log({"test_mIOU": res_miou})
    # with open('miou.txt','a') as f:
    #     f.write('test_miou: ' + str(res_miou) + '\n')


def count_cell(image,mask,color):
    image_1 = image.copy()
    image_2 = image.copy()
    image_3 = image.copy()
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 8)

    # Sure background area (dilation enlarges the white region)
    sure_bg = cv2.dilate(opening, kernel, iterations = 1)

    # Finding sure foreground area using distance transform and thresholding
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

    # Sure foreground is now float, we need it to be in uint8
    sure_fg = np.uint8(sure_fg)

    # Unknown region is the area where sure foreground and background do not overlap
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Labeling the sure foreground
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so that the background is not 0 but 1
    markers = markers + 1

    # Mark the unknown region with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image_3, markers)

    # The boundary region will be marked -1, we want to separate them as individual cells.
    # For the cells, let's assign each cell a different value, starting from 2 (as 1 is for background)
    new_mask_separated = np.zeros_like(mask, dtype=np.uint8)
    new_mask_separated[markers > 1] = 255

    contours_r, _ = cv2.findContours(new_mask_separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#pred_r_2d

    # image_1 = image.copy()

    cv2.drawContours(image_1, contours_r, -1, color, 2) 

    # for i in contours_r:
    #     print('position: ', i)

    small_cnt = []
    contours_b, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours_b))
    cv2.drawContours(image_3, contours_b, -1, color, 2) 
    contours_b_ = []
    for cnt in contours_b:
        if cv2.contourArea(cnt) > 200:
            contours_b_.append(cnt)
    # print(len(contours_b_))

    for cnt in contours_b_:
        if cv2.contourArea(cnt) < 400  : # Define area_threshold based on your observations and cv2.contourArea(cnt) < 400
            cv2.drawContours(image_1, [cnt],-1, color, 2)
            small_cnt.append(cnt)

    overlap_count = 0

    # Iterate through all contours in contours_r
    for cnt_r in contours_r:
        # Convert contour to an array of points
        points_r = np.squeeze(cnt_r)
        # Iterate through all contours in small_cnt
        for cnt_s in small_cnt:
            # Check if any point of cnt_r is inside cnt_s
            for point in points_r:
                try:
                    if cv2.pointPolygonTest(cnt_s, (float(point[0]), float(point[1])), False) >= 0:
                        overlap_count += 1
                        break  #
                except:
                    break
    
    real_count = len(contours_r)-overlap_count+len(small_cnt)
    # print("Number of overlapping contours:", overlap_count)
    # print("Number of real contours:", real_count)

    cv2.drawContours(image_2, contours_b_, -1, (255,0,0), 2) 

    return real_count, image_1, image_2,image_3

def select_reliable3(model, dataloader, args,checkpoints=None):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    model.eval()
    print('number checkpoints:', len(checkpoints))

    for i in range(len(checkpoints)):
        checkpoints[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []
    id_to_unreliability = []
    num_= len(tbar)
    re_num=0
    un_num=0
    for i in range(len(checkpoints)):
        checkpoints[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability_miou = []

    with torch.no_grad():
        for img, id in tbar:
            img = img.cuda()

            preds = []
            for models in checkpoints:
                pred = models(img)
                preds.append(torch.argmax(pred, dim=1).cpu().numpy())
                # preds.append(torch.argmax(pred, dim=1).cpu().numpy())
            
            last_preds  = model(img)
            last_preds = torch.argmax(last_preds, dim=1).cpu().numpy()
            # ft = torch.argmax(ft, dim=1).cpu().numpy()

            mIOU = []
            for i in range(len(preds) - 1): # preds
                metric = meanIOU(num_classes=3)
                metric.add_batch(preds[i], last_preds)
                mIOU.append(metric.evaluate()[-1])
                
            # try:
                # print('miou:',mIOU)
            if (max(mIOU)-min(mIOU)) < 0.2:
                # reliability = sum(mIOU) / len(mIOU)
                id_to_reliability_miou.append(id[0])#, reliability
            # except:
            #     pass
    re_count=0
    with torch.no_grad():
        for img,  id in tbar:
            img = img.cuda()
            if id[0] in id_to_reliability_miou:
                
                
                
            # preds = []
            
                pred = model(img)
                # torch.argmax(pred, dim=1).cpu().numpy()
                predxx = pred
                # predxx = torch.softmax(pred, dim=1)
                confidence_values = torch.max(predxx, dim=1).values
                # 筛选出置信度大于0.8的像素点
                threshold = 0.8
                high_confidence_pixels = (confidence_values > threshold).float()
                reliability = high_confidence_pixels.sum() / high_confidence_pixels.numel()
                # print('reliability ratio:',reliability)
                if reliability > 0.8:
                    # print('reliability')
                    re_count+=1
                    
                    id_to_reliability.append((id[0], reliability))
                    re_num+=1
                else :
                    id_to_unreliability.append((id[0], reliability))
                    # print('unreliability')

            # print('reliability value:', reliability)
            # reliability = sum(high_confidence_pixels) / len(high_confidence_pixels)
            else: 
                id_to_unreliability.append((id[0],0))
                # print('unreliability')
            #     un_num+=1
            #     if un_num > (num_)//2:
            #         id_to_reliability.append((id[0], reliability))
                
            # elif reliability > 0.8:
            #     id_to_reliability.append((id[0], reliability))
            #     re_num+=1
    
        print('have {} images are reliabel'.format(re_count))

    # id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)

    if os.path.exists(os.path.join(args.reliable_id_path, 'reliable_ids.txt')) and os.path.exists(os.path.join(args.reliable_id_path, 'unreliable_ids.txt')):
        os.remove(os.path.join(args.reliable_id_path, 'reliable_ids.txt'))
        os.remove(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'))
                                                                                                 
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_unreliability:
            f.write(elem[0] + '\n')
    if len(id_to_unreliability) > re_num:
        return False
    else :
        return True

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.5, beta=0.5, smooth=1):     
        # print('inputs',inputs.shape)
        # print('targets',targets.shape)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky    
    
if __name__ == '__main__':
    args = parse_args()

    # if args.epochs is None:
    #     args.epochs = {'pascal': 15, 'cityscapes': 240}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.002, 'cityscapes': 0.004, 'ki67':0.00001, 'nuc':0.00001}[args.dataset] # / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 512, 'cityscapes': 721, 'ki67':512, 'nuc':512}[args.dataset] #321

    print()
    print(args)

    main(args)
