# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:38:58 2022

@author: Amund
"""

import numpy as np
import os
import pickle
import random
random.seed(42)
#path = 'coordinates/'

#coordinate_dictionary_names = os.listdir(path)

#val_set_ratio = 0.2
#test_set_ratio = 0.2
#train_set_ratio = 0.6



def train_val_sets(val_set_ratio, path):
    
    filenames = os.listdir(path)
    filenames = np.array(filenames)
    
    malign= []
    benign = []
    
    for idx, name in enumerate(filenames):
        name = int(name.split(' ')[0][6:])
        if name >= 50:
            malign.append(idx)
        else:
            benign.append(idx)
        
    train_set_ratio = 1.0 - val_set_ratio
    length_dataset = len(filenames)
    train_length = int(length_dataset*train_set_ratio)
    val_length = int(length_dataset*val_set_ratio)
    #test_length = int(length_dataset*test_set_ratio)
    print(length_dataset, train_length, val_length)#, test_length)
    
    train_indicies = []
    for _ in range(int(train_length/2)):
        element_malign = random.choice(malign)
        element_benign = random.choice(benign)
        train_indicies.append(element_malign)
        train_indicies.append(element_benign)
        malign.remove(element_malign)
        benign.remove(element_benign)
        
    train_set = filenames[train_indicies]

    val_indicies = []
    for _ in range(int(val_length/2)):
        element_malign = random.choice(malign)
        element_benign = random.choice(benign)
        val_indicies.append(element_malign)
        val_indicies.append(element_benign)
        malign.remove(element_malign)
        benign.remove(element_benign)
    
    val_set = filenames[val_indicies] 
    

def unpack_sets(data_set, path, images_pr_wsi = 100, shuffle=False):
    
    list_of_dictionaries = []
    
    for wsi in data_set:
        with open(path+wsi, 'rb') as handle:
            coordinate_dict = pickle.load(handle)
            #print(coordinate_dict)
            coordinate_dict_samples = random.sample(list(coordinate_dict),
                                                    min(images_pr_wsi, len(coordinate_dict)))
            #print(coordinate_dict_samples)
            for sample in coordinate_dict_samples:
                list_of_dictionaries.append(coordinate_dict[sample])
                
    if shuffle:
        random.shuffle(list_of_dictionaries)
    return list_of_dictionaries

def unpack_single_wsi(path_name, classes, images_pr_wsi=np.inf, shuffle=False):
    '''unpacks a single wsi'''
    list_of_dictionaries = []
    valid_patches = []
    with open(path_name, 'rb') as handle:
        coordinate_dict = pickle.load(handle)

    for id_, patch in coordinate_dict.items():
        if patch['tissue_type'] in classes:
            valid_patches.append(patch)
    
    if shuffle:
        coordinate_dict_samples = random.sample(list(valid_patches),
                                                min(images_pr_wsi, len(valid_patches)))
    else:
        coordinate_dict_samples = range(min(images_pr_wsi, len(valid_patches)))
    
    for sample in coordinate_dict_samples:
        list_of_dictionaries.append(valid_patches[sample])

    return list_of_dictionaries

def unpack_single_wsi_inference(path_name):
    '''unpacks a single wsi'''
    list_of_dictionaries = []
    with open(path_name, 'rb') as handle:
        coordinate_dict = pickle.load(handle)
    for id_, patch in coordinate_dict.items():
        list_of_dictionaries.append(patch)
    return list_of_dictionaries
        
          
def unpack_all_into_single_batch(path, classes, tiles_pr_class_pr_wsi = None, even_classes = False):
    '''Unpack all WSI patches from a folder into one single batch
    input:
        path: path to folder containing coordinate dictionaries. e.g. train/, val/ or test/
        classes: list of classes to extract coordinates from e.g. ['lesion benign', 'lesion malignant', 'normal tissue']
        tiles_pr_class_pr_wsi: the maximum amount of tiles to extract from each class of each WSI. This will level the amount of tiles
        between the WSIs and classes
        even_classes: If True, the classes with lower amount of tiles will be given more tiles in an attempt to even out the amount between the classes.
    return:
        list_of_dictionaries: list containing dictionaries containing top left tile coordinates'''
    
    if not tiles_pr_class_pr_wsi: #if all possible coordinates are used then even classes is set to False
        even_classes = False
    
    #initiate lists and dictionaries
    list_of_dictionaries = []
    filenames = os.listdir(path)
    class_counts = {class_:0 for class_ in classes}
    unused_patches = {class_: [] for class_ in classes}
    
    #iterate through each WSI's coordinate dictionary in path
    for wsi_name in filenames:
        with open(path+wsi_name, 'rb') as handle:
            coordinate_dict = pickle.load(handle)
        
        #Collect all coordinates that belong to a class in classes
        individual_wsi_class_dict = {tissue_type: [] for tissue_type in classes}
        for values in coordinate_dict.values():
            if values['tissue_type'] in classes:
                values['augmentation'] = None
                individual_wsi_class_dict[values['tissue_type']].append(values)
        
        #Add coordinates to the output list_of_dictionaries
        for class_, values in individual_wsi_class_dict.items():
            if len(values) == 0:
                continue
            
            if tiles_pr_class_pr_wsi is not None:
                indexes = np.array(list(range(len(values))))
                random.shuffle(indexes)
                values = np.array(values)
                samples = values[indexes[0:min(tiles_pr_class_pr_wsi,len(values))]]
                unused_samples = values[indexes[tiles_pr_class_pr_wsi:]]
                list_of_dictionaries += list(samples)
                class_counts[class_]+=len(samples)
                unused_patches[class_] += list(unused_samples)
            else:
                list_of_dictionaries += values
                
    if tiles_pr_class_pr_wsi is not None:
        print('class count before even classes')
        print(class_counts)
    
    #add tiles/patches to the classes with less to even out the amount between classes
    if even_classes:
        max_length_idx = list(class_counts.values()).index(max(class_counts.values()))
        max_length_class = list(class_counts.keys())[max_length_idx]
        max_length = class_counts[max_length_class]
        for class_, count in class_counts.items():
            if class_== max_length_class:
                continue
            difference = max_length - count
            samples = random.sample(unused_patches[class_], min(difference, len(unused_patches[class_])))
            list_of_dictionaries += samples
            class_counts[class_]+=len(samples)
    
    if tiles_pr_class_pr_wsi is not None:
        print('class count after even classes')
        print(class_counts)
                
    return list_of_dictionaries

def add_augmented_patches(data_set, classes, augmentations, shuffle=False):
    '''adds augmentation to the classes that have less samples than the class with most samples
    inputs:
        data_set: list of patches. Each patch is a dictionary.
        classes: dictionary of classes
        augmentation: list of augmentation types to add
        shuffle: if True the data_set will be shuffled prior to adding copied patches with new augmentations. This will be best if only one magnification is to be used in trainng. But if several magninfications are to be used in one training the order of corresponding patches between magnifications might be lost
    outputs:
        list of new patches to be added to the existing dataset'''
    
    if shuffle:
        random.shuffle(data_set)
    
    
    classes = {class_:[] for class_ in classes}
    class_count = {class_:0 for class_ in classes} #To count and show class count before and after augmentation
    augmentation_count = {aug: 0 for aug in augmentations}
    for patch in data_set:
        class_ = patch['tissue_type']
        classes[class_].append(patch)
    
  
    max_patches = 0
    for class_, values in classes.items():
        patches = len(values)
        class_count[class_]+=patches
        if patches > max_patches:
            max_patches = patches
            max_class = class_
    
    print('class count before augmentation')
    print(class_count)
    
    augmented_patches = []
    for class_, values in classes.items():
        if class_ != max_class:
            augmented_pr_class =[]
            patches = len(values)
            difference = max_patches - patches
            amount_of_copies_to_make_pr_patch = min(difference, patches*5)
            j = 0
            for i in range(amount_of_copies_to_make_pr_patch):
                new_patch = values[i%len(values)].copy()
                new_patch['augmentation'] = augmentations[j%len(augmentations)]
                augmentation_count[augmentations[j%len(augmentations)]] += 1
                augmented_pr_class.append(new_patch)
                if i%len(values) == len(values)-1:
                    j+=1
            class_count[class_]+= len(augmented_pr_class)
            augmented_patches += augmented_pr_class
    
        
    print('class count after augmentation')
    print(class_count)
    print('augmentation count')
    print(augmentation_count)
    return augmented_patches

    

    
    
