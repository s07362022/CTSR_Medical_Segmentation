from dataset.transform import crop, hflip, normalize, resize, blur, cutout

import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import imgaug.augmenters as iaa
import cv2

ALL_transform = iaa.Sequential([
                            # iaa.Fliplr(0.5),
                            # iaa.Flipud(0.5),
                            # iaa.Rot90((0, 1)),
                            # iaa.AddToHue((-1,1)),
                            # iaa.MultiplySaturation((0.9,1.1)),
                            iaa.AddToSaturation((-1,1)),
                            iaa.OneOf([iaa.LinearContrast((0.9,1.1)),
                                        iaa.GammaContrast((0.9,1.1))]),
                            iaa.AddToBrightness((-1,1)),
                            # iaa.Sometimes(0.5, iaa.OneOf([iaa.MedianBlur(k=(3,3)),
                            #                                 iaa.GaussianBlur(sigma=(0.0, 1.0))])),
                            
                            iaa.Sometimes(0.5, iaa.OneOf([iaa.AdditiveGaussianNoise(scale=(0, 0.01*255), per_channel=True),
                                                            iaa.AdditiveLaplaceNoise(scale=(0, 0.01*255), per_channel=True)])),
#                             iaa.Affine(scale=(0.875, 1.125), translate_percent=(-0.0625, 0.0625), rotate=(-10,10),mode='reflect'),
#                             iaa.Crop(percent= 0.25, keep_size=False, sample_independently=True),
                            ])

transform = iaa.Sequential([
                            iaa.Fliplr(0.5),
                            iaa.Flipud(0.5),
                            iaa.Rot90((0, 1)),
                            iaa.AddToHue((-1,1)),
                            # iaa.Resize({"height": 512, "width": 512})
                            # iaa.MultiplySaturation((0.9,1.1)),
                            # iaa.AddToSaturation((-10,10)),
                            # iaa.OneOf([iaa.LinearContrast((0.9,1.1)),
                            #             iaa.GammaContrast((0.9,1.1))]),
#                             iaa.AddToBrightness((-10,10)),
#                             iaa.Sometimes(0.5, iaa.OneOf([iaa.MedianBlur(k=(3,5)),
#                                                             iaa.GaussianBlur(sigma=(0.0, 1.0))])),
                            
#                             iaa.Sometimes(0.5, iaa.OneOf([iaa.AdditiveGaussianNoise(scale=(0, 0.05*255), per_channel=True),
#                                                             iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255), per_channel=True)])),
#                             iaa.Affine(scale=(0.875, 1.125), translate_percent=(-0.0625, 0.0625), rotate=(-10,10),mode='reflect'),
#                             iaa.Crop(percent= 0.25, keep_size=False, sample_independently=True),
                            ])

train_seq = iaa.Sequential([iaa.Fliplr(0.5),
                            iaa.Flipud(0.5),
                            iaa.Rot90((0, 1)),
                            iaa.AddToHue((-3,3)),
                            iaa.MultiplySaturation((0.9,1.1)),
                            iaa.AddToSaturation((-10,10)),
                            iaa.OneOf([iaa.LinearContrast((0.9,1.1)),
                                        iaa.GammaContrast((0.9,1.1))]),
                            iaa.AddToBrightness((-10,10)),
                            iaa.Sometimes(0.5, iaa.OneOf([iaa.MedianBlur(k=(3,5)),
                                                            iaa.GaussianBlur(sigma=(0.0, 1.0))])),
                            
                            iaa.Sometimes(0.5, iaa.OneOf([iaa.AdditiveGaussianNoise(scale=(0, 0.05*255), per_channel=True),
                                                            iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255), per_channel=True)])),
                            iaa.Affine(scale=(0.875, 1.125), translate_percent=(-0.0625, 0.0625), rotate=(-10,10),mode='reflect'),
                            iaa.Crop(percent=0.125, keep_size=False, sample_independently=True),
                            ])


cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
def _border_get(mask):
    dilate_mask = cv2.dilate(mask, cross_kernel,iterations = 2)
    erode_mask = cv2.erode(mask, cross_kernel,iterations = 2)
    dilate_mask = np.squeeze(dilate_mask)
    erode_mask = np.squeeze(erode_mask)
    border = dilate_mask - erode_mask
    if len(border.shape) < 3:
        border = border[:,:,np.newaxis]
    return border

def count_nuclei(mask):
    mask = thresh = (mask * (255 / 1)).astype(np.uint8) 
    # _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours)>0):
        # 选择第一个找到的核
        # print('contours',contours)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # 添加padding
        padding = 20
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        x2 = min(x + w + 2*padding, mask.shape[1])
        y2 = min(y + h + 2*padding, mask.shape[0])
        return len(contours), x, y, x2, y2
    else:
        return len(contours), 0, 0, 0, 0

def merge_images(img1, mask1, img2, mask2):
    # 确保掩膜为单通道
    if len(mask1.shape) == 3:
        mask1 = mask1[:, :, 0]
    if len(mask2.shape) == 3:
        mask2 = mask2[:, :, 0]
    # 在mask2中找到一个核的区域进行裁剪
    # mask2 = (mask2 * (255 / 2)).astype(np.uint8) 
    nucleus_area = count_nuclei(mask2)
    # print('nucleus_area',nucleus_area)
    c,x, y, x2, y2 = nucleus_area
    crop_img2 = img2[y:y2, x:x2]
    crop_mask2 = mask2[y:y2, x:x2]

    # 将裁剪的图像和掩膜粘贴到img1和mask1上
    img1[y:y2, x:x2] = crop_img2
    mask1[y:y2, x:x2] = np.maximum(mask1[y:y2, x:x2], crop_mask2)
    mask1 = mask1[:, :, np.newaxis]

    return img1, mask1


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train' or (self.mode == 'final_train'):
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids

        else:
            if mode == 'test':
                id_path = 'dataset/splits/%s/test/test.txt' % name
            elif mode == 'val':
                id_path = 'dataset/splits/%s/ex21/val.txt' % name
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'inf':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
        print('use mode :',mode)
        print('use data path :',len(self.ids))

    def __getitem__(self, item):
        id = self.ids[item]
        # img = Image.open(os.path.join(self.root, id.split(' ')[0]))
        
        img = cv2.imread(os.path.join(self.root, id.split(' ')[0]))
#         print('img path:',os.path.join(self.root, id.split(' ')[0]))
        
        
        
        if  self.mode == 'label':
            mask = cv2.imread(os.path.join(self.root, id.split(' ')[1]),0)
#             mask[mask >=1] = 1
#             mask = cv2.resize(mask,(512,512))
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            # mask = Image.open(os.path.join(self.root, id.split(' ')[1])).convert('L') 
            # mask_array = np.array(mask)
            # mask[mask >=1] = 1
            
            # mask = Image.fromarray(mask_array)
            img, mask = normalize(img, mask)
            # print(np.unique(mask))
            # print(id.split(' ')[1])
            return img, mask, id
        elif  self.mode == 'val':
#             mask = Image.open(os.path.join(self.root, id.split(' ')[1])).convert('L') 
            mask = cv2.imread(os.path.join(self.root, id.split(' ')[1]),0)
#             img = cv2.imread( id.split(' ')[0])
#             mask = cv2.imread( id.split(' ')[1],0)
#             mask[mask >=1] = 1
#             mask = cv2.resize(mask,(512,512))
#             img = cv2.resize(img,(512,512))
            # border = cv2.imread(os.path.join(self.root, 'boundary/'+file_border),0)
            mask[mask >=3] = 0
            if self.name == 'live_fibrosis':
                newsize = (3072,3072)
                img = img.resize(newsize)
#             cv2.imwrite(r'/work/u5914116/ki67/val_test/{}'.format(id.split(' ')[0]),img)
#             print(r'/work/u5914116/ki67/val_test/{}'.format(id.split(' ')[0]))
#             cv2.imwrite(r'/work/u5914116/ki67/val_test/{}'.format(id.split(' ')[1]),mask)
            # mask_array = np.array(mask)
            # mask_array[mask_array >= 1] = 1
            # mask = Image.fromarray(mask_array)
            img, mask = normalize(img, mask)

            return img, mask, id
        elif self.mode == 'test':
            mask = cv2.imread( id.split(' ')[1],0)
#             mask = cv2.imread(os.path.join(self.root, id.split(' ')[1]),0)
            mask2 = np.zeros_like(mask)
#             mask2[mask ==2] = 1
#             mask2[mask ==1] = 1
            # border = _border_get(mask2)
            if img is None or mask is None:
                print(id)
 
                print(f"Error loading image or mask for index {id}")

                raise ValueError(f"Image or mask could not be loaded for index {id}")
            # print("test")
            
            # mask[mask >=1] = 1
            img, mask = normalize(img, mask)
            return img, mask, id
        
        elif  self.mode == 'inf':
            # print(img.shape,os.path.join(self.root, id.split(' ')[0]))
            img = cv2.resize(img,(512,512))
            # mask = Image.fromarray(mask_array)
            img= normalize(img, None)

            return img, id
        
        elif self.mode == 'train' :
#             file_border = id.split(' ')[1].split('/')[1]
            # print(file_border)
            mask = cv2.imread(os.path.join(self.root, id.split(' ')[1]),0)
#             mask[mask >=1] = 1
#             print(np.unique(mask))
#             mask = cv2.resize(mask,(512,512))
#             print(np.unique(mask))
#             img = cv2.resize(img,(512,512))
            mask_name = id.split(' ')[1]
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask[mask >=3] = 0

            
            mask = np.reshape(mask,(mask.shape[0],mask.shape[1],1))

            
            mask = mask[np.newaxis,:,:,:]
            img = np.array(img)
            # if random.random() < 0.5:
            #     blur_img = cv2.GaussianBlur(img, (0, 0), 25)
            #     img = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
            img,mask = transform(image=img,segmentation_maps=mask)#ALL_transform(image=img,segmentation_maps=mask)
            img3, mask3 = img.copy(),mask.copy()
            img2,mask2 =img.copy(),mask.copy()
            img2,mask2 = ALL_transform(image=img3,segmentation_maps=mask3)

            mask = np.squeeze(mask)
            mask2 = np.squeeze(mask2)
            
            # mask_b2[mask2 ==1] = 1
            # mask_b2[mask2 ==2] = 1
            # border2 = _border_get(mask_b2)
            
            # print(np.unique(mask))
            img, mask = normalize(img, mask)
            img2, mask2 = normalize(img2, mask2)
            
            
            

            
            
            return img, mask, img2, mask2#, border, border2
            
        # else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            

       
        # print('using transform')

        # strong augmentation on unlabeled images
        if (self.mode == 'final_train' and id in self.labeled_ids ) or (self.mode == 'semi_train' and id in self.labeled_ids):
            # print(id in self.labeled_ids)
            # mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
            mask = cv2.imread(os.path.join(self.root, id.split(' ')[1]),0)
#             mask[mask >=1] = 1
#             print(np.unique(mask))
#             mask = cv2.resize(mask,(512,512))
#             print(np.unique(mask))
#             img = cv2.resize(img,(512,512))
            
            # mask[mask >=1] = 1
            mask[mask ==255] = 0
            # mask[mask ==127] = 1
            # print(np.unique(mask))
            # mask2 = np.zeros_like(mask)
            # mask2[mask ==2] = 1
            # mask2[mask ==1] = 2
            # mask = mask2
            
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            # print(id in self.unlabeled_ids)
            fname = os.path.basename(id.split(' ')[1])
            # print(os.path.join(self.pseudo_mask_path, fname))
            mask = cv2.imread(os.path.join(self.pseudo_mask_path, fname),0)
#             mask[mask >=1] = 1
#             print(np.unique(mask))
#             mask = cv2.resize(mask,(512,512))
#             print(np.unique(mask))
#             img = cv2.resize(img,(512,512))
            
            # mask = cv2.imread('%s/%s' % (self.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))
            mask[mask ==255] = 0
#             mask[mask ==127] = 2



        mask = np.reshape(mask,(mask.shape[0],mask.shape[1],1))
        # border = _border_get(mask) ##


        mask = mask[np.newaxis,:,:,:]
        img = np.array(img)
        img,mask = transform(image=img,segmentation_maps=mask)#ALL_transform(image=img,segmentation_maps=mask)
        img3, mask3 = img.copy(),mask.copy()
        img2,mask2 =img.copy(),mask.copy()
        img2,mask2 = ALL_transform(image=img3,segmentation_maps=mask3)

        mask = np.squeeze(mask)
        mask2 = np.squeeze(mask2)

        # print(np.unique(mask))
        img, mask = normalize(img, mask)
        img2, mask2 = normalize(img2, mask2)

        return img, mask, img2, mask2#,border
        
    def __len__(self):
        return len(self.ids)