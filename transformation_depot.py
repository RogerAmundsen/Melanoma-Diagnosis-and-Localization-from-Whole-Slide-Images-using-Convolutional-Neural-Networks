# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:32:01 2022

@author: Amund
"""

from torchvision import transforms

def get_transformation(name, input_size, image_net_normalization=True):
    '''Premade transformations put in a function in a seperate file to clean up the main code.
    inputs:
        name: name of the transformation type:
                'val': Only apply centercrop to input size and transform to tensors
                'train': Apply centercrop, augmentations, and transform ot tensors. Augmentations as rotation and flipping might make the overlapping patches less similar to eachother and help prevent overfitting.
                '40x': The 40x magnification tiles do not overlap with others. No augmentation added. Image is resized to "input size".
        input_size: the image size to be used in training.
        image_net_normalization: If the model is pretrained on ImageNet, add normalization using ImageNet mean and std
    return:
        transformation object'''
    
    normalization_mean_IN = [0.485, 0.456, 0.406] #ImageNet normalization
    normalization_std_IN = [0.229, 0.224, 0.225] #ImageNet normalization
    
    #normalization_mean_trainset = [0.7980,0.6558,0.7681] #complete train set from unix 04042022
    #normalization_std_trainset = [0.1429,0.1773,0.1283] #complete train set from unix 040420224
    
    normalization_mean_trainset = [0.7980,0.6558,0.7681] #complete train set from unix Lesion benign and lesion malignant classes
    normalization_std_trainset = [0.1429,0.1773,0.1283] #complete train set from unix Lesion benign and lesion malignant classes
    
    if name == 'val':
        trans = transforms.Compose([transforms.CenterCrop(input_size),
                                          transforms.ToTensor()])
    elif name == 'aug':
        trans = transforms.Compose([
                                    transforms.RandomRotation(20, fill=255),
                                    transforms.RandomVerticalFlip(p=0.4),
                                    transforms.RandomHorizontalFlip(p=0.4),
                                    #transforms.CenterCrop(input_size),
                                    transforms.RandomCrop(input_size),
                                    transforms.ToTensor()])
    elif name == 'train':
        trans = transforms.Compose([transforms.RandomCrop(input_size),
                                          transforms.ToTensor()])
        
    elif name == '40x':
        trans = transforms.Compose([transforms.Resize(input_size),
                                          transforms.ToTensor()])
    elif name == 'only_tensor':
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        print('Not a valid transformation name')
          
    if image_net_normalization:
    	trans.transforms.insert(99, transforms.Normalize(normalization_mean_IN, normalization_std_IN))
    else:
    	trans.transforms.insert(99, transforms.Normalize(normalization_mean_trainset, normalization_std_trainset))
        
    return trans
