import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm
import numpy as np
from utils2.utils import get_lr
from utils2.utils_metrics import f_score
from utils import count_params, meanIOU, color_map, AverageMeter, intersectionAndUnion, init_log

def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs,img2,labels , pngs = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs,_ = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                labels = torch.softmax(labels, dim=1)
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs,_ = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    labels = torch.softmax(labels, dim=1)
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs,img2,labels , pngs = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            #----------------------#
            #   前向传播
            #----------------------#
            outputs,_ = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                outputs = torch.softmax(outputs, dim=1)
                main_dice = Dice_loss(outputs, labels)
                loss  = loss + main_dice
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank,valloader,args):
    total_loss      = 0
    total_mIOU   = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for  iteration, (img, mask, img2, mask2)  in enumerate(gen):
        if iteration >= epoch_step: 
            break
        # imgs,img2,labels , pngs = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                img, mask = img.cuda(), mask.cuda()
                
                img2, mask2 = img2.cuda(), mask2.cuda()
                # imgs    = imgs.cuda(local_rank)
                # pngs    = pngs.cuda(local_rank)
                # labels  = labels.cuda(local_rank)
                weights = weights.cuda()

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            pred,fea = model_train(img,True)
            pred2,fea2 = model_train(img2,True)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                f_loss1 = Focal_Loss(pred, mask, weights, num_classes = num_classes) 
                f_loss2 = Focal_Loss(pred2, mask2, weights, num_classes = num_classes) 
                loss = f_loss1+f_loss2
                
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                predx = torch.softmax(pred, dim=1)
                loss_di1=Dice_loss(predx, mask, 2, smooth = 1e-5)
                pred2x=torch.softmax(pred2, dim=1)
                loss_di2=Dice_loss(pred2x, mask2, 2, smooth = 1e-5)
                main_dice = (loss_di2+loss_di1)/2
                loss      = loss + main_dice

            # with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                # eval_mode = 'sliding_window' if args.dataset == 'cityscapes' else 'original'
                # mIOU, iou_class = evaluate(model, valloader, eval_mode, args)
                
                # _f_score = f_score(pred, mask)

            loss.backward()
            optimizer.step()
       

            #----------------------#
            #   反向传播
            #----------------------#
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

        total_loss      += loss.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))
    return model_train
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
#         if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
#             torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))

#         if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
#             print('Save best model to best_epoch_weights.pth')
#             torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
#         torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg.crop_size
                b, _, h, w = img.shape
                final = torch.zeros(b, 2, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred, fea = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
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

                pred, fea,_,_ = model(img)
                pred=pred.argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), 2, 0)#255

            intersection = torch.from_numpy(intersection)
            union = torch.from_numpy(union)
            target = torch.from_numpy(target)

            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class