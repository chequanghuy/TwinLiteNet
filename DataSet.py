import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math
from PIL import Image
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
import albumentations as A
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def RandomBilateralBlur(img, sigma_bila_low = 0.05, sigma_bila_high=1.0):
    """
    Apply Bilateral Filtering

    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sigma = random.uniform(sigma_bila_low, sigma_bila_high)
    blurred_img = denoise_bilateral(np.array(img_rgb), sigma_spatial=sigma, channel_axis = 2)
    blurred_img *= 255
    blurred_img_rgb = Image.fromarray(blurred_img.astype(np.uint8))
    blurred_img_bgr = cv2.cvtColor(np.array(blurred_img_rgb), cv2.COLOR_RGB2BGR)
    return blurred_img_bgr


    
def RandomGaussianBlur(img, sigma_gaus_a = 1.15, sigma_gaus_b=0.15):
    """
    Apply Gaussian Blur
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sigma = sigma_gaus_b + random.random() * sigma_gaus_a
    blurred_img = gaussian(np.array(img_rgb), sigma=sigma, channel_axis = 2)
    blurred_img *= 255
    blurred_img_rgb = Image.fromarray(blurred_img.astype(np.uint8))
    blurred_img_bgr = cv2.cvtColor(np.array(blurred_img_rgb), cv2.COLOR_RGB2BGR)
    return blurred_img_bgr


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
    s = random.uniform(1 - scale, 1.5 + scale)
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


class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, hyp, valid=False, transform = None):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.transform = transform
        self.degrees = hyp["degrees"]
        self.translate = hyp["translate"]
        self.scale = hyp["scale"]
        self.shear = hyp["shear"]
        self.hgain = hyp["hgain"]
        self.sgain = hyp["sgain"]
        self.vgain = hyp["vgain"]
        self.Random_Crop = A.RandomCrop(width=hyp["width_crop"], height=hyp["height_crop"])

        self.prob_perspective = hyp["prob_perspective"]
        self.prob_flip = hyp["prob_flip"]
        self.prob_hsv = hyp["prob_hsv"]
        self.prob_bilateral = hyp["prob_bilateral"]
        self.prob_gaussian = hyp["prob_gaussian"]
        self.prob_crop = hyp["prob_crop"]
        
        self.Tensor = transforms.ToTensor()
        self.valid=valid
        if valid:
            self.root='../bdd100k/images/val'
            self.names=os.listdir(self.root)
        else:
            self.root='../bdd100k/images/train'
            self.names=os.listdir(self.root)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_=640
        H_=384
        image_name=os.path.join(self.root,self.names[idx])

        if not self.valid:
            image = cv2.imread(os.path.join(self.root,self.names[idx]))
        else:
            image = cv2.imread(os.path.join(self.root,self.names[idx]))
        
        label1 = cv2.imread(image_name.replace("images","segments").replace("jpg","png"), 0)
        label2 = cv2.imread(image_name.replace("images","lane").replace("jpg","png"), 0)
        


        if not self.valid:
            if random.random() < self.prob_perspective:
                combination = (image, label1, label2)
                (image, label1, label2)= random_perspective(
                    combination=combination,
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear
                )
            if random.random() < self.prob_hsv:
                augment_hsv(image, self.hgain, self.sgain, self.vgain)
            if random.random() < self.prob_flip:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)
            
            if random.random() < self.prob_bilateral:
                image = RandomBilateralBlur(image)
            if random.random() < self.prob_gaussian:
                image = RandomGaussianBlur(image)
            if random.random() < self.prob_crop:
                masks = np.stack([label1, label2],axis=2)
                transformed = self.Random_Crop(image=image, mask=masks)
                image = transformed['image']
                labels = transformed['mask']
                label1 = labels[:,:,0]
                label2 = labels[:,:,1]


            image = letterbox(image, (H_, W_))
        else:
            image = letterbox(image, (H_, W_))

        label1 = cv2.resize(label1, (W_, 360))
        label2 = cv2.resize(label2, (W_, 360))
        


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
        image = np.array(image)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)


       
        return image_name,torch.from_numpy(image),(seg_da,seg_ll)

class Dataset320(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, hyp, valid=False, transform = None):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.transform = transform
        self.degrees = hyp["degrees"]
        self.translate = hyp["translate"]
        self.scale = hyp["scale"]
        self.shear = hyp["shear"]
        self.hgain = hyp["hgain"]
        self.sgain = hyp["sgain"]
        self.vgain = hyp["vgain"]
        self.Random_Crop = A.RandomCrop(width=hyp["width_crop"], height=hyp["height_crop"])

        self.prob_perspective = hyp["prob_perspective"]
        self.prob_flip = hyp["prob_flip"]
        self.prob_bilateral = hyp["prob_bilateral"]
        self.prob_gaussian = hyp["prob_gaussian"]
        self.prob_crop = hyp["prob_crop"]
        
        self.Tensor = transforms.ToTensor()
        self.valid=valid
        if valid:
            self.root='../bdd100k/images/val'
            self.names=os.listdir(self.root)
        else:
            self.root='../bdd100k/images/train'
            self.names=os.listdir(self.root)


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_=320
        H_=192
        image_name=os.path.join(self.root,self.names[idx])

        if not self.valid:
            image = cv2.imread(os.path.join(self.root,self.names[idx]))
        else:
            image = cv2.imread(os.path.join(self.root,self.names[idx]))
        
        label1 = cv2.imread(image_name.replace("images","segments").replace("jpg","png"), 0)
        label2 = cv2.imread(image_name.replace("images","lane").replace("jpg","png"), 0)
        


        if not self.valid:
            if random.random() < self.prob_perspective:
                combination = (image, label1, label2)
                (image, label1, label2)= random_perspective(
                    combination=combination,
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear
                )
            if random.random() < self.prob_hsv:
                augment_hsv(image, self.hgain, self.sgain, self.vgain)
            if random.random() < self.prob_flip:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)
            
            if random.random() < self.prob_bilateral:
                image = RandomBilateralBlur(image)
            if random.random() < self.prob_gaussian:
                image = RandomGaussianBlur(image)
            if random.random() < self.prob_crop:
                masks = np.stack([label1, label2],axis=2)
                transformed = self.Random_Crop(image=image, mask=masks)
                image = transformed['image']
                labels = transformed['mask']
                label1 = labels[:,:,0]
                label2 = labels[:,:,1]


            image = letterbox(image, (H_, W_))

            label1 = cv2.resize(label1, (W_, 180))
            label2 = cv2.resize(label2, (W_, 180))

        else:

            image = letterbox(image, (H_, W_))

            label1 = cv2.resize(label1, (W_*2, 360))
            label2 = cv2.resize(label2, (W_*2, 360))
        

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
        image = np.array(image)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)


       
        return image_name,torch.from_numpy(image),(seg_da,seg_ll)
    



