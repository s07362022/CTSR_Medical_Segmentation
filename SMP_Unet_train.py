# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:00:42 2024

@author: user
"""
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from torchsummary import summary
import os
from tqdm import tqdm
from dataset.semi_cv3 import SemiDataset
import random
import numpy as np
import cv2
from copy import deepcopy
from nets.unet_training import CE_Loss, Dice_loss, ContrastiveLoss , Focal_Loss
from utils import count_params, meanIOU, color_map, AverageMeter, intersectionAndUnion, init_log




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

pseudo_mask_path = 'outdir/pseudo_masks_ki67/ki67/ex21'





# model.load_state_dict(torch.load(all_weights))
#summary(model,input_size=(3,512,512))

# 損失函數和優化器
def Weight_Dice_loss(predicted, target, num_classes=3, smooth=1e-6):
    smooth = 1e-6  # 
    losses = []

    for class_index in range(num_classes):
        predicted_class = predicted[:, class_index]  # size = (batch,512,512) 
        target_class = (target == class_index).float()  # size = (batch,1,512,512)
        #batch_size = predicted_class.size(0)

        predicted_class = predicted_class.view(predicted_class.size(0), -1) # flatten(512x512 => 262144)
        target_class = target_class.view(target_class.size(0), -1) # flatten(512x512 => 262144)

        intersection = torch.sum(predicted_class * target_class) # 
        union = torch.sum(predicted_class) + torch.sum(target_class) # 
        dice = (2.0 * intersection) / (union + smooth)
        #weight = (262144 * batch_size - (torch.sum(predicted_class))) / (262144 * batch_size)
        dice_loss = 1.0 - dice
        losses.append(dice_loss)

    loss = sum(losses) / num_classes  # 

    return loss

def Weight_CE_loss(predicted, target, num_classes=3, smooth=1e-6):
    smooth = 1e-6
    gamma = 2
    losses = []

    for class_index in range(num_classes):
        predicted_class = predicted[:, class_index]  
        target_class = (target == class_index).float()  # 
        batch_size = predicted_class.size(0)

        predicted_class = predicted_class.view(predicted_class.size(0), -1) # flatten(512x512 => 262144)
        target_class = target_class.view(target_class.size(0), -1) # flatten(512x512 => 262144)

        pred_log = torch.log(predicted_class + smooth) # 
        comp_pred_log = torch.log(1 - predicted_class + smooth) # 
        scale_1 = (1 - predicted_class) ** gamma
        scale_2 = predicted_class ** gamma
        term_1  = torch.sum(scale_1 * target_class * pred_log) / (262144 * batch_size)
        term_2 = torch.sum(scale_2 * (1 - target_class) * comp_pred_log) / (262144 * batch_size)
        #weight_2 = torch.sum(predicted_class) / (262144 * batch_size)
        #weight_1 = 1 - weight_2
        loss = term_1 + term_2
        losses.append(loss)

    loss = - (sum(losses) / num_classes)  # 

    return loss





def train_model(sup_model, bor_model, optimizer_sup,optimizer_bor, train_loader, val_loader, num_epochs=20, n_classes=3):
    best_miou = 0.0  # Initialize the best MIoU score
    best_weights = None  # Variable to store the best weights
    total_iters = len(train_loader) * num_epochs
    iters = 0
    cls_weights     = np.array([0.9,1.5,1.0], np.float32)
    weights = torch.from_numpy(cls_weights)
    weights = weights.cuda()
    print("focal weights",weights)
    num_classes = 3
    checkpoints = []

    for epoch in range(num_epochs):
        sup_model.train()  # Set model to training mode
        bor_model.train()
        running_loss = 0.0

        tbar = tqdm(train_loader)
        # Training loop
        for img, mask_cell, img2, mask2, img4, mask_border  in tbar:
            
            
            img = img.cuda()
            mask_cell = mask_cell.cuda()
            img2 = img2.cuda()
            mask2 = mask2.cuda()
            
            img4 = img4.cuda()
            mask_border = mask_border.cuda()
            
            optimizer_sup.zero_grad()
            optimizer_bor.zero_grad()

            # weak aug
            outputs_w = sup_model(img)
            predx_w = torch.softmax(outputs_w, dim=1)
            loss_di1=Dice_loss(predx_w, mask_cell, 3, smooth = 1e-5)
            mask_cell=mask_cell.reshape(mask_cell.shape[0],mask_cell.shape[1],mask_cell.shape[2])
#             print("mask_cell",mask_cell.shape)
            f_loss1 = Focal_Loss(outputs_w, mask_cell, weights, num_classes = num_classes) 
            loss_w = loss_di1 + f_loss1
            
#             loss_w = criterion_1(predx_w, mask_cell) + criterion_2(predx_w, mask_cell)
            
                    
           

            # strong aug
            outputs_s= sup_model(img2)
            predx_s = torch.softmax(outputs_s, dim=1)
            loss_di2=Dice_loss(predx_s, mask2, 3, smooth = 1e-5)
            mask2=mask2.reshape(mask2.shape[0],mask2.shape[1],mask2.shape[2])
            f_loss2 = Focal_Loss(outputs_s, mask2, weights, num_classes = num_classes) 
            loss_s = loss_di2+ f_loss2
#             loss_s = criterion_1(predx_s, mask2) + criterion_2(predx_s, mask2)

            sup_loss = loss_s*0.5 + loss_w*0.5

            sup_loss.backward()
            optimizer_sup.step()
            
            outputs_bor= bor_model(img4)
            outputs_bor = torch.softmax(outputs_bor, dim=1)
            loss_di3=Dice_loss(outputs_bor, mask_border, 3, smooth = 1e-5)
#             f_loss3 = Focal_Loss(outputs_bor, mask_border, weights, num_classes = num_classes) 
            loss_bor = loss_di3 #+ f_loss3
#             loss_bor = criterion_1(outputs_bor, mask_border) + criterion_2(outputs_bor, mask_border)
            loss_bor.backward()
            optimizer_bor.step()
            
            
            tbar.set_description('sup_loss: %.3f, bor_loss: %.3f' % (sup_loss,loss_bor))
            iters += 1
            lr = 1e-4 * (1 - iters / total_iters) ** 0.9
            optimizer_sup.param_groups[0]["lr"] = lr 
            optimizer_bor.param_groups[0]["lr"] = lr 
    

        # Evaluation loop
        val_miou, bor_miou, MIoU = evaluate_model(sup_model,bor_model, val_loader, n_classes=n_classes)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {sup_loss:.3f}, Val MIoU: {val_miou:.3f}, Bor MIoU: {bor_miou:.3f}, MIoU2: {MIoU:.3f}, lr: {lr}")

        # Update the best model if the current model is better
        if val_miou > best_miou:
            best_miou = val_miou
            best_weights_sup = sup_model.state_dict().copy()  # Copy the model's state_dict
            best_weights_bor = bor_model.state_dict().copy()
            
        if val_miou > 70:
            # After all epochs, load the best model weights
            sup_model.load_state_dict(best_weights_sup)
            # Optionally, save the best model to disk
            torch.save(best_weights_sup, "/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex21/unet_best_sup_weights.pth")
            bor_model.load_state_dict(best_weights_bor)
#             # Optionally, save the best model to disk
            torch.save(best_weights_bor, "/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex21/unet_last_border_weights.pth")
        
            if len(checkpoints)==3:
                checkpoints.pop(0)
            else:
                checkpoints.append(deepcopy(sup_model))


            

    print(f"Training complete. Best Val MIoU: {best_miou:.4f}")

    return sup_model,bor_model, checkpoints # Return the model with the best weights




def inference_model(sup_model, bor_model,inf_dataloader ):
    tbar = tqdm(inf_dataloader)
    red_count_sum = 0
    bule_count_sum = 0
    
    for img, id  in tbar:
#         image = cv2.imread(id[0].split(' ')[0])#
        img = img.cuda()
        
        pred = sup_model(img)
        pred_= torch.argmax(pred, dim=1).cpu().numpy() # 0,1,2 
        
         
        
        pred_bor = bor_model(img)
        pred_bor_= torch.argmax(pred_bor, dim=1).cpu().numpy() #0,1,2
        pred_bor_mask= np.zeros_like(pred_bor_)
        pred_bor_mask[pred_bor_==1] = 1
        pred_bor_mask[pred_bor_==2] = 0
        
        
        
        # draw (mask x pred)
        final_pred = (pred_bor_mask * pred_)
        
        pred_r = np.zeros_like(final_pred, dtype=np.uint8)
        pred_b = np.zeros_like(final_pred, dtype=np.uint8)
        
        pred_bormask = np.zeros_like(final_pred, dtype=np.uint8)
        pred_bormask[final_pred == 1] = 255
        pred_bormask[final_pred == 2] = 127
        pred_bormask = pred_bormask.reshape(512,512,1)
#         print('pred_bormask',pred_bormask.shape)
        
        pred_nobormask = np.zeros_like(pred_, dtype=np.uint8)
        pred_nobormask[pred_ == 1] = 255
        pred_nobormask[pred_ == 2] = 127
        pred_nobormask = pred_nobormask.reshape(512,512,1)

        pred_r[final_pred == 1] = 1  
        pred_b[final_pred == 2] = 1  

        pred_r_2d = pred_r[0, :, :]
        pred_b_2d = pred_b[0, :, :]
        contours_r, _ = cv2.findContours(pred_b_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_b, _ = cv2.findContours(pred_r_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        xx = '/work/u5914116/ki67/ex15/'+id[0].split(' ')[0]
        result2 = cv2.imread(xx) #id[0].split(' ')[0]
        
#         cv2.drawContours(result2, contours_r, -1, (255,0,0), 1) 
                
#         cv2.drawContours(result2, contours_b, -1, (0,0,255), 1) 
        
#         cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/border_inf0522_1', os.path.basename(id[0].split(' ')[0])),result2)
#         cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/border_inf0522_bormask_1', os.path.basename(id[0].split(' ')[0])),pred_bormask)
#         cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/border_inf0522_nobormask_1', os.path.basename(id[0].split(' ')[0])),pred_nobormask)
        
        # draw (only pred)
#         final_pred = (pred_bor_mask * pred_)
        
#         pred_r2 = np.zeros_like(pred_, dtype=np.uint8)
#         pred_b2 = np.zeros_like(pred_, dtype=np.uint8)

#         pred_r2[pred_ == 1] = 1  
#         pred_b2[pred_ == 2] = 1  

#         pred_r_2dx = pred_r2[0, :, :]
#         pred_b_2dx = pred_b2[0, :, :]
#         contours_rx, _ = cv2.findContours(pred_b_2dx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours_bx, _ = cv2.findContours(pred_r_2dx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         result = cv2.imread(id[0].split(' ')[0])
        
#         cv2.drawContours(result, contours_rx, -1, (255,0,0), 1) 
                
#         cv2.drawContours(result, contours_bx, -1, (0,0,255), 1) 
        
# #         cv2.imwrite('%s/%s' % ('./outdir/pseudo_masks/ki67/border_inf0509_NoDot', os.path.basename(id[0].split(' ')[0])),result)
        
#         b_pred_c = len(contours_b)
#         r_pred_c = len(contours_r)
#         bule_count_sum+=b_pred_c
#         red_count_sum+=r_pred_c

        pred = pred.cpu()
        pred = torch.argmax(pred, dim=1).cpu()
        pred = pred.squeeze(0).numpy().astype(np.uint8)
        pred = Image.fromarray(pred, mode='L')
        pred.save('%s/%s' % (pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))
        
#     ratio_ = red_count_sum / (bule_count_sum+red_count_sum)
#     ratio_ = round(ratio_, 2)
#     print("Number of positive: {}, Number of negative: {},  ratio: {}".format(red_count_sum,bule_count_sum,ratio_))
#     re_txt = "ID: {}, Number of positive: {}, Number of negative: {},  ratio: {}".format("18-ki67",red_count_sum,bule_count_sum,ratio_)
#     with open('inf_result.txt','a') as f:
#         f.write(re_txt + '\n')
            
        
        



#import torch
from torchvision.utils import save_image

def miou_score(pred, target, smooth=1e-6, n_classes=3):
    """
    Compute the Mean Intersection over Union (MIoU) score.
    :param pred: the model's predicted probabilities
    :param target: the ground truth
    :param smooth: a small value to avoid division by zero
    :param n_classes: the number of classes in the dataset
    :return: the MIoU score
    """
    pred = torch.argmax(pred, dim=1)  # Convert probabilities to class predictions size = (batch,512,512)
    miou_total = 0.0
    for class_id in range(n_classes):
        true_positive = ((pred == class_id) & (target == class_id)).sum()
        false_positive = ((pred == class_id) & (target != class_id)).sum()
        false_negative = ((pred != class_id) & (target == class_id)).sum()
        intersection = true_positive
        union = true_positive + false_positive + false_negative + smooth
        miou = intersection / union
        miou_total += miou
    return miou_total / n_classes

def evaluate_model(model,model_bor, loader, n_classes=3):
    model.eval()
    total_miou = 0
    total_miou_bor = 0
    b_miou=0
    mIOU = []
    tbar = tqdm(loader)
    with torch.no_grad():
        for img, mask_cell, mask_border, id in tbar:
            images = img.cuda()
            masks = mask_cell.cuda()
            mask_border = mask_border.cuda()
            outputs = model(images)
            
            ####
            metric = meanIOU(num_classes=3)
            metric.add_batch(torch.argmax(outputs, dim=1).cpu().numpy(), masks.cpu().numpy())
            mIOU.append(metric.evaluate()[-1])   
            #####
            
            total_miou += miou_score(outputs, masks, n_classes=n_classes)
            if model_bor !=None:
                outputs_bor = model_bor(images)
                total_miou_bor += miou_score(outputs_bor, mask_border, n_classes=n_classes)

                b_miou = (total_miou_bor / len(loader))*100
            
    return (total_miou / len(loader))*100 ,  b_miou , (sum(mIOU)/len(mIOU))*100

def predict(model, loader, save_dir="predicted_masks"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, package in enumerate(loader):
            images = package[0]
            images = images.cuda()
            outputs = model(images)
            masks = torch.argmax(outputs, dim=1)  # Convert probabilities to predictions
            
            r_channel = (torch.where(masks == 1,
                               torch.tensor(255,dtype=torch.uint8,device = 'cuda'),
                               torch.tensor(0,dtype=torch.uint8,device = 'cuda')
                               )).unsqueeze(1)
            
            g_channel = (torch.where(masks == 2,
                               torch.tensor(255,dtype=torch.uint8,device = 'cuda'),
                               torch.tensor(0,dtype=torch.uint8,device = 'cuda')
                               )).unsqueeze(1)
            
            b_channel = (torch.where(masks == 0,
                               torch.tensor(0,dtype=torch.uint8,device = 'cuda'),
                               torch.tensor(0,dtype=torch.uint8,device = 'cuda')
                               )).unsqueeze(1)

            preds = torch.cat((r_channel,g_channel),1)
            preds = torch.cat((preds,b_channel),1)
            
            for j, pred in enumerate(preds):
                save_image(pred.float(), os.path.join(save_dir, f"predict_{idx * loader.batch_size + j}.png"))

def test_eval(model, loader):
    model.eval()
    val_miou= 0
    bor_miou= 0
    with torch.no_grad():
        
        val_miou, bor_miou, x_miou = evaluate_model(model,None, loader, n_classes=3)
        print('test_miou: ' + str(val_miou) + '\n')
        print('test_miou2: ' + str(x_miou) + '\n')
    with open('miou.txt','a') as f:
        f.write('test_miou: ' + str(val_miou) + '\n')

     
        
def select_reliable3(model, dataloader,checkpoints=None):
    if not os.path.exists(reliable_id_path):
        os.makedirs(reliable_id_path)

    model.eval()

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

    if os.path.exists(os.path.join(reliable_id_path, 'reliable_ids.txt')) and os.path.exists(os.path.join(reliable_id_path, 'unreliable_ids.txt')):
        os.remove(os.path.join(reliable_id_path, 'reliable_ids.txt'))
        os.remove(os.path.join(reliable_id_path, 'unreliable_ids.txt'))
                                                                                                 
    with open(os.path.join(reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability:
            f.write(elem[0] + '\n')
    with open(os.path.join(reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_unreliability:
            f.write(elem[0] + '\n')
    if len(id_to_unreliability) > re_num:
        return False
    else :
        return True
    
# 建立模型
from nets.unet_training import CE_Loss, Dice_loss, ContrastiveLoss , Focal_Loss
from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils2.utils import (download_weights, seed_everything, show_config, worker_init_fn)
from utils import count_params, meanIOU, color_map, AverageMeter, intersectionAndUnion, init_log
'''
backbone = 'resnet50'
# download_weights(backbone)
pretrained  = True
#model_path = "/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex11/unet_resnet50_90.31_DFCAS_border.pth"
model_path = 'unet_resnet_medical.pth'#"unet_resnet50_83.53.pth"
sup_model = Unet(num_classes=3, pretrained=pretrained, backbone=backbone).train()
if not pretrained:
    weights_init(model)
if model_path != '':
    if local_rank == 0:
        print('Load weights {}.'.format(model_path))
    model_dict      = sup_model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    sup_model.load_state_dict(model_dict)
    print('load weight')


backbone = 'resnet50'
download_weights(backbone)
pretrained  = True
#model_path = "/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex11/unet_resnet50_90.31_DFCAS_border.pth"
# model_path = 'unet_resnet_medical.pth'#"unet_resnet50_83.53.pth"
bor_model = Unet(num_classes=3, pretrained=pretrained, backbone=backbone).train()
'''



# sup_model = smp.Unet(encoder_name="resnet50",
#                  decoder_channels=(512, 256, 64, 32, 16),
#                  in_channels=3,
#                  classes=3)

# bor_model = smp.Unet(encoder_name="resnet50",
#                  decoder_channels=(512, 256, 64, 32, 16),
#                  in_channels=3,
#                  classes=3)

# sup_model = smp.UnetPlusPlus(encoder_name="resnet50",
#                  decoder_channels=(512, 256, 64, 32, 16),
#                  in_channels=3,
#                  classes=3)

# bor_model = smp.UnetPlusPlus(encoder_name="resnet50",
#                  decoder_channels=(512, 256, 64, 32, 16),
#                  in_channels=3,
#                  classes=3)

sup_model = smp.UnetPlusPlus(
                encoder_name = 'resnext50_32x4d',
                encoder_weights="ssl",
                classes = 3,
                )
bor_model = smp.UnetPlusPlus(
                encoder_name = 'resnext50_32x4d',
                encoder_weights="ssl",
                classes = 3,
                )


sup_model.load_state_dict(torch.load(r'/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex17/unet_best_sup_weights.pth')) # ex11/unetpp_resnet50_91.14_resnet50_ssl_class3.pth
bor_model.load_state_dict(torch.load(r"/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex17/unet_last_border_weights.pth"))

# for param in sup_model.encoder.parameters():
#     param.requires_grad = False

sup_model.cuda()
bor_model.cuda()

criterion_1, criterion_2 = Weight_Dice_loss, Weight_CE_loss
optimizer_sup = torch.optim.Adam(sup_model.parameters(), lr=2e-5)
optimizer_bor = torch.optim.Adam(bor_model.parameters(), lr=2e-5)

trainset = SemiDataset('ki67', '/work/u5914116/ki67/ex15/', 'train', None, 'dataset/splits/ki67/ex21/labeled.txt')
trainloader = DataLoader(trainset, batch_size=8, shuffle=True,
                             pin_memory=True, num_workers=4, drop_last=True)

valset = SemiDataset('ki67', '/work/u5914116/ki67/ex15/', 'val',None,'dataset/splits/ki67/ex21/val.txt')
valloader = DataLoader(valset, batch_size= 1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

MODE = 'inf'
dataset = SemiDataset('ki67', '/work/u5914116/ki67/ex15/', 'inf', None, None, 'dataset/splits/ki67/ex21/unlabeled.txt') #ex17/bortest.txt
inf_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

MODE = 'test'
dataset = SemiDataset('ki67', '/work/u5914116/ki67/test/', 'test', None, None, 'dataset/splits/ki67/test/test.txt')
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

reliable_id_path = '/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/reliable_ids_ki67/ex17'

if __name__ == "__main__":
    best_weights_path = 'best_model_weights.pth'
    # if os.path.exists(best_weights_path):
    #     # 匯入已有的權重
    #     model.load_state_dict(torch.load(best_weights_path))
    #     print("Loaded existing model weights.")
    # else:
    #     print("No existing model weights found, starting training from scratch.")

    sup_model_ , bor_model_, checkpoints = train_model(sup_model,bor_model, optimizer_sup,optimizer_bor, trainloader, valloader, num_epochs=70, n_classes=3)
    test_eval(sup_model_, test_dataloader)
    
    print('\n\n\n================> Total stage 2/5: Select reliable images for the 1st stage re-training')
    
    MODE = 'inf'
    select_reliable3(sup_model_, inf_dataloader,checkpoints)
    
    print('\n\n\n================> Total stage 3/5: Pseudo labeling reliable images')
    cur_unlabeled_id_path = os.path.join('/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/reliable_ids_ki67', 'reliable_ids.txt')
    dataset = SemiDataset('ki67', '/work/u5914116/ki67/ex15/', 'inf', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    inference_model(sup_model, bor_model,inf_dataloader )
    
    print('\n\n\n================> Total stage 4/5: The 1st stage re-training on labeled and reliable unlabeled images')

    MODE = 'semi_train'

    trainset = SemiDataset('ki67', '/work/u5914116/ki67/ex15/', MODE, None,
                           'dataset/splits/ki67/ex21/labeled.txt', cur_unlabeled_id_path, pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)
    
    sup_model_ , bor_model_, checkpoints = train_model(sup_model_,bor_model_, optimizer_sup,optimizer_bor, trainloader, valloader, num_epochs=40, n_classes=3)
    lab_list = os.listdir(r"/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex16_quater/")
    for ix in lab_list:
        print('\n\n\n================> Total stage 5/5: TEST ')
        sup_model.load_state_dict(torch.load(r'/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex21/{}'.format(ix)))
        print('model: ',ix)
        test_eval(sup_model, test_dataloader)
    
   
#     ix = 'unetpp_resnet50_81.28_resnext50_32x4d_ssl_class3_2.pth'
#     sup_model.load_state_dict(torch.load(r'/home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/models/ki67/ex15_half/{}'.format(ix)))
#     test_eval(sup_model, test_dataloader)
    # 逕行測試與預測
#     test_MIoU = evaluate_model(sup_model_, test_loader)
    # print(f"test MIoU: {test_MIoU}")
    # predict(sup_model_, test_loader, save_dir="predicted_masks")
    # print("Finsh All Training and Testing")
