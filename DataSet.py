import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
def random_perspective(combination,  degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    img, gray, line = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
            line = cv2.warpPerspective(line, M, dsize=(width, height), borderValue=0)
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)
            line = cv2.warpAffine(line, M[:2], dsize=(width, height), borderValue=0)



    combination = (img, gray, line)
    return combination

def random_perspective2(combination,  degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    img = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))

        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))




    combination = img
    return combination


class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, transform=None,valid=False,engin='kaggle',data='bdd'):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''

        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.valid = valid
        self.engin = engin
        self.data = data

        if self.data == 'bdd':
            if self.engin == 'kaggle': #bdd dataset on kaggle engine
                if valid:
                    self.root = '/kaggle/input/bdd100k-dataset/bdd100k/bdd100k/images/100k/val'
                    self.names = os.listdir(self.root)
                else:
                    self.root = '/kaggle/input/bdd100k-dataset/bdd100k/bdd100k/images/100k/train'
                    self.names = os.listdir(self.root)#[:1500]  # [:1000]
            else:                       #bdd dataset on colab engine
                if valid:
                    self.root = '/content/data/bdd100k/bdd100k/images/100k/val'
                    self.names = os.listdir(self.root)
                else:
                    self.root = '/content/data/bdd100k/bdd100k/images/100k/train'
                    self.names = os.listdir(self.root)#[:1500]
        elif self.data == 'IADD':
            if self.engin == 'kaggle':  #IADD dataset on kaggle engine
                if valid:
                    self.root = '/kaggle/working/IADD/IADDv6/val/img'
                    self.names = os.listdir(self.root)
                else:
                    self.root = '/kaggle/working/iadd/img/content/train_p1_unlabeled'
                    self.names = os.listdir(self.root)#[:1000]
            else:                        #IADD dataset on colab engine
                if valid:
                    self.root = '/content/IADD/IADDv6/val/img'
                    self.names = os.listdir(self.root)
                else:
                    self.root = '/content/iadd/img/content/train_p1_unlabeled'
                    self.names = os.listdir(self.root)#[:1000]


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_=512
        H_=512
        image_name=os.path.join(self.root,self.names[idx])

        image = cv2.imread(image_name)
        if self.data == 'bdd':
            if self.engin == 'kaggle':
                label1 = cv2.imread(image_name.replace("input/bdd100k-dataset/bdd100k/bdd100k/images/100k","working/labels/bdd_seg_gt").replace("jpg","png"), 0)
                label2 = cv2.imread(image_name.replace("input/bdd100k-dataset/bdd100k/bdd100k/images/100k","working/labels/bdd_lane_gt").replace("jpg","png"), 0)
            else:
                label1 = cv2.imread(image_name.replace("bdd100k/bdd100k/images/100k", "labels/bdd_seg_gt").replace("jpg", "png"), 0)
                label2 = cv2.imread(image_name.replace("bdd100k/bdd100k/images/100k", "labels/bdd_lane_gt").replace("jpg", "png"), 0)
        elif self.data == 'IADD':
            if self.valid:
                label1 = cv2.imread(image_name.replace("img", "drivable").replace(".jpg", ".png"), 0)
                label2 = cv2.imread(image_name.replace("img", "lane").replace(".jpg", ".png"), 0)
            else:
                label1 = cv2.imread(image_name.replace("img/content/train_p1_unlabeled", "da").replace(".jpg", ".png"), 0)
                label2 = cv2.imread(image_name.replace("img/content/train_p1_unlabeled", "ll").replace(".jpg", ".png"), 0)

        if not self.valid:
            if random.random()<0.5:
                combination = (image, label1, label2)
                (image, label1, label2)= random_perspective(
                    combination=combination,
                    degrees=10,
                    translate=0.1,
                    scale=0.25,
                    shear=0.0
                )
            if random.random()<0.5:
                augment_hsv(image)
            if random.random() < 0.5:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)

        label1 = cv2.resize(label1, (W_, H_))
        label2 = cv2.resize(label2, (W_, H_))
        image = cv2.resize(image, (W_, H_))

        _,seg_b1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY_INV)
        _,seg_b2 = cv2.threshold(label2,1,255,cv2.THRESH_BINARY_INV)
        _,seg1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY)
        _,seg2 = cv2.threshold(label2,1,255,cv2.THRESH_BINARY)

        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)
        seg_da = torch.stack((seg_b1[0], seg1[0]),0)
        seg_ll = torch.stack((seg_b2[0], seg2[0]),0)
        # image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        if self.transform is not None :
            image = self.transform(image)

        return image_name,image,(seg_da,seg_ll)


class UlabeledDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, transform=None, engin='kaggle', data='IADD'):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''

        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.data=data
        self.engin=engin
        if self.engin == 'kaggle':
            self.root='/kaggle/working/iadd/img/content/train_p1_unlabeled'
            self.names=os.listdir(self.root)
        else:
            self.root='/content/iadd/img/content/train_p1_unlabeled'
            self.names=os.listdir(self.root)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_ = 512
        H_ = 512
        image_name = os.path.join(self.root,self.names[idx])

        image = cv2.imread(image_name)
        shape = image.shape
        image = cv2.resize(image, (W_, H_))

        # image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        if self.transform is not None :
            image = self.transform(image)

        return image_name, image, shape