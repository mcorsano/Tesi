import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import tifffile
import cv2


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
        pr = torch.clamp(tensors, 0, 255)
    return tensors


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        # self.both_transform = transforms.RandomCrop((64, 64*3), pad_if_needed=True, padding_mode="edge")
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))


    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        #img = self.both_transform(img)
        img_hr = self.hr_transform(img)
        img_lr = self.lr_transform(img)






        # test con uso di self.transform
        # img = Image.open(self.files[index % len(self.files)])
        # img = (np.asarray(img))

        # img = np.repeat(img[..., np.newaxis], 3, -1)
        # img = torch.permute(torch.from_numpy(img), (2,0,1))
        
        # toPIL = transforms.ToPILImage()
        # img = toPIL(img)

        # img_lr = self.lr_transform(img)
        # img_hr = self.hr_transform(img)




        # test senza uso di self.trasnform
        # img = Image.open(self.files[index % len(self.files)])
        # img = (np.asarray(img))/2.062036   # max among all the dataset

        # img_lr = cv2.resize(img, (32,32), interpolation = cv2.INTER_NEAREST)
        # img_hr = cv2.resize(img, (128,128), interpolation = cv2.INTER_NEAREST)
        
        # img_lr = np.repeat(img_lr[..., np.newaxis], 3, -1)
        # img_hr = np.repeat(img_hr[..., np.newaxis], 3, -1)

        # img_lr = torch.permute(torch.from_numpy(img_lr), (2,0,1))
        # img_hr = torch.permute(torch.from_numpy(img_hr), (2,0,1))

        # tifffile.imsave("images/inizio/%d.tiff" % index, img_hr.numpy())
        
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)