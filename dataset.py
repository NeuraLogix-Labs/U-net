from torch.utils.data import Dataset 
import cv2
import os
#òimport torch , numpy to make things esay 📽️
import torch
import numpy as np


# 🎯
class Dataset_Train (Dataset) : 
    def __init__(self , data_dir_path=None , transform = None , label_transform = None):
        super().__init__()
        self.data_dir_path = data_dir_path
        self.transform = transform
        self.label_transform = label_transform
        self.data_dir = os.listdir(data_dir_path)


    def __len__(self) : 
        return len(self.data_dir)
    # 🎉
    def __getitem__(self, index):
        img_dir = os.path.join(self.data_dir_path , self.data_dir[index] , 'images')
        img_path = os.path.join(img_dir , os.listdir(img_dir)[0])
        img = cv2.imread(img_path) # 🦠
        masks_dir = os.path.join(self.data_dir_path , self.data_dir[index] , 'masks')
        maskes_list = os.listdir(masks_dir)
        mask = Dataset_Train.get_maske(masks_dir = masks_dir , list_maskes = maskes_list )



        if self.transform : 
            img = self.transform(img)
        if self.label_transform :
          mask = self.label_transform(mask)

        return img /255., mask
    

    @staticmethod # 🌠🌠 🌠🌠🌠 🌠
    def get_maske(list_maskes , masks_dir) :
        mask = np.zeros((128, 128, 1), dtype=np.bool)
        IMG_WIDTH, IMG_HEIGHT = 128  , 128
        for part_mask in list_maskes : 
            mask_ = cv2.imread(os.path.join(masks_dir , part_mask))
            mask_= cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
            mask_ = cv2.resize(mask_, (IMG_WIDTH, IMG_HEIGHT))
            mask_ = np.expand_dims(mask_ , 2)
            #mask = np.maximum(mask , mask_)
            mask = np.where(mask_ > mask, 1, mask) # to get 1 or 0 

        return mask


class Dataset_Test (Dataset) : 
    def __init__(self , data_dir_path , transform = None ):
        super().__init__()
        self.data_dir_path = data_dir_path
        self.transform = transform
        self.data_dir = os.listdir(data_dir_path)


    def __len__(self) : 
        return len(self.data_dir)
    # 🎉
    def __getitem__(self, index):
        img_dir = os.path.join(self.data_dir_path , self.data_dir[index] , 'images')
        img_path = os.path.join(img_dir , os.listdir(img_dir)[0])
        img = cv2.imread(img_path) # 🦠


        if self.transform : 
            img = self.transform(img)


        return img / 255.







