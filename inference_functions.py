# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:42:17 2022

@author: Amund
"""


from __future__ import print_function
from __future__ import division

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os
#import pickle
#import json
import cv2
from PIL import Image

#import homemade python scripts
import create_sets
import transformation_depot
import pretrained_spyder_main

#from matplotlib.patches import Patch

from skimage.measure import label, regionprops

torch.cuda.empty_cache()

vipshome = 'C:\\vips-dev-8.10\\bin'

# set PATH
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

string = os.environ['PATH']
new_string=string.split(';')[0]
a=string.split(';')[0]

for path in string.split(';')[1:]:
    if a != path:
        new_string=new_string+';'+path
    a = path

os.environ['PATH']=new_string

import pyvips

class Inference_dataset(Dataset):
    def __init__(self, 
                 transform,
                 data_set,
                 tile_size=256,
                 magnification='10x',
                 index = None):
        
        self.transform = transform
        self.index = index
        self.tile_size = tile_size
        self.data_set = data_set
        self.magnification = magnification
        self.mag_lvl = {'40x':0, '20x':1, '10x':2, '5x':3,
                        '2.5x':4, '1.25x':5, '0.625x':6}

    def __len__(self):
         return len(self.data_set)
    def __getitem__(self, idx):
        path = self.data_set[idx]['path']
        #Convert top left coordinates to center coordinates
        x_coord = self.data_set[idx][self.magnification][0]
        y_coord = self.data_set[idx][self.magnification][1]
        #Extract image from top left coordinates and tile size
        vips_object = pyvips.Image.new_from_file(path, level=self.mag_lvl[self.magnification], autocrop=True).flatten()
        tile_object = vips_object.extract_area(x_coord,y_coord, self.tile_size, self.tile_size)
        tile_image = np.ndarray(buffer=tile_object.write_to_memory(),
                            dtype='uint8',
                            shape=[tile_object.height, tile_object.width, tile_object.bands])
                 
        tile_image = Image.fromarray(tile_image)
        tile_image = self.transform(tile_image)        
        return tile_image
        
    def display_tile(self, idx):
        '''Display the original tile without augmentation'''
        path = self.data_set[idx]['path']
        x_coord = self.data_set[idx][self.magnification][0]
        y_coord = self.data_set[idx][self.magnification][1]
        vips_object = pyvips.Image.new_from_file(path, level=self.mag_lvl[self.magnification], autocrop=True).flatten()
        tile_object = vips_object.extract_area(x_coord,y_coord, self.tile_size, self.tile_size)
        tile_array = np.ndarray(buffer=tile_object.write_to_memory(),
                             dtype='uint8',
                             shape=[tile_object.height, tile_object.width, tile_object.bands])
        plt.imshow(tile_array)
        plt.title(path[8:].split(' ')[0]+' ('+str(x_coord)+' '+str(y_coord)+')')
        plt.show()

class Run_inference:
    '''A class with methods to run inference (using Pytorch) on a single wsi.
    It can:
        Initialize a model, and load its trained weights.
        Create a dataset and apply transformations.
        Return predicted labels, either as probabilities or classified based on given probability threshold'''
    def __init__(self, params, batch_size):
        self.model_name = params["model"]
        self.weight_name = params["weight_name"]
        self.class_dict = params["class_dict"]
        self.ImageNet_normalization = params["ImageNet_normalization"]
        self.magnification = params['magnification']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tissue_label = 20
        
    
    def create_model(self, model_path):
        '''Initializes model, loads its trained weights and creates the corresponding image transformations to be used on the inference set.
        Argument takes the path of the model folder that contains all models and search for the correct model weight to load.
        inputs:
            model_path: path to the folder that contains all the model folders'''
        num_classes = len(self.class_dict)
        print('MODEL NAME:', self.model_name)
        self.model, self.input_size = pretrained_spyder_main.initialize_model(self.model_name,
                                                                              num_classes, 
                                                                              False)
        self.model = self.model.to(self.device)
        for folder in os.listdir(model_path):
            for filename in os.listdir(os.path.join(model_path,folder)):
                if filename == self.weight_name:
                    model_weight_path = os.path.join(model_path, folder, filename)
                    try:
                        self.model.load_state_dict(torch.load(model_weight_path))
                        print('model state dict loaded')
                    except:
                        print('Model state dict NOT loaded')
                    

        self.transformation = transformation_depot.get_transformation('val', self.input_size, self.ImageNet_normalization)
                    
    def create_dataset(self, wsi_name_and_path, data_set_class):
        '''creates dataset given the wsi_name and its coordinate_dictionary path. This method uses a function from the create_sets.py file.
        inputs:
            data_set_path: directory path of the folder containing all the WSI coordinate dictionaries
            wsi_name: name of the specific WSI to be run infrence on,
            data_set_class: class that creates pytorch dataset using Datasets'''
        data_set_dictionary = create_sets.unpack_single_wsi_inference(wsi_name_and_path)
        self.data_set = data_set_class(self.transformation, data_set_dictionary, magnification = self.magnification)
        
    def inference(self, threshold = None):
        '''Runs inference
        return: list of probabilities for each label'''
        
        testloader = torch.utils.data.DataLoader(self.data_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.model.eval()
        predicted_y = []
        
        with torch.no_grad():
            for patches in testloader:
                #for patch in patches:
                    #plt.imshow(patch.permute(1, 2, 0))
                    #plt.show()
                patches = patches.to(self.device)
                outputs = self.model(patches)
                predicted = list(F.softmax(outputs, dim=1).cpu().numpy())
                
                if threshold:
                    predicted_y += [np.argmax(y_hat) if np.max(y_hat) >= threshold else self.tissue_label for y_hat in predicted]
                else:
                    predicted_y += predicted
        
        return predicted_y
    
class Return_model_metrics:
    def __init__(self, mask_path, class_dict, return_prediction_mask, weights_name, inference_path):
        self.mask_path = mask_path
        self.tissue_label = 20
        self.class_dict = class_dict
        self.return_prediction_mask = return_prediction_mask
        self.weights_name = weights_name[:-4]
        self.inference_path = inference_path
    
    def return_metrics_from_threshold(self, wsi, pred_y, data_set_path, ignore_regions=False):
        '''returns metrics from a single wsi based on probabilies and threshold.
        This method use binary masks to calculate metrics.
        These metrics are a models ability to seperate non-annotated tissue from annotated Lesion Benign and Lesion Malignant.
        This method treats a multi-class problem as a binary-class problem. It merge Lesion Benign and Lesion Malignant into a "positive" class,
        while the non-annotated Tissue is the negative class. Hence, a true positive counts as a Lesion Benign or Lesion Malignant prediction within
        an area annotated by the pathologist as either of the two. E.g. a Lesion Benign patch classified as Lesion Malignant does not count as a false positive, but rather a true positive.
        input:
            wsi: name of the WSI
            pred_y: a list of predicted classes of each patch, e.g. [0,1,1,2,20]. self.class_dict tells which tissue classes these labels belong to (20 is usually the "Tissue" label (aka negative label aka non-annotated tissue))
            data_set_path: directory path of the folder containing all the WSI coordinate dictionaries
            ignore_regions:  Many WSIs contain several thin slices of tissue. Ignore the slices that have no annotations. i.e., remove them from the masks (give them the value 0).
        output:
            TP: True positives counts as the intersection of a binary mask containing the predicted Lesion Benign and Lesion Malignant pixels and a binary mask containing the annotated lesion beningn and lesion malignant pixels.
            TN: True negatives counts as the intersection of a binary mask containing the predicted Tissue pixels and a binary mask containing non-annotated tissue
            FP: False positives counts as pixels predicted as either Lesion Benign or Lesion Malignant withouth having been annotated as either by the pathologist
            FN: False negatives counts as pixels predicted as Tissue, but are either Lesion Benign or Lesion Malignant.
            pred_mask: a prediction mask that illustrates the predictions made. (Green = Lesion Benign, Red = Lesion Malignant, Blue = Tissue, Black = Background (not a prediction))
            
            '''
        
        #initiate dataset: a list of top left coordinates
        data_set_list = create_sets.unpack_single_wsi_inference(os.path.join(data_set_path,wsi))
        
        #load masks
        tissue_mask = cv2.imread(self.mask_path+'{}/tissue_mask.png'.format(wsi[:-18]),0)
        try:
            lesion_benign_mask = cv2.imread(self.mask_path+'{}/lesion benign.png'.format(wsi[:-18]),0)
            lesion_benign_mask[tissue_mask==0]=0
            lesion_malignant_mask = cv2.imread(self.mask_path+'{}/lesion malignant.png'.format(wsi[:-18]),0)
            lesion_malignant_mask[tissue_mask==0]=0
        
            #create new masks
            true_mask = lesion_benign_mask+lesion_malignant_mask
            
        except:
            print('No masks from annotations found. No metrics will be returned. Only predicted image will be returned.')
            true_mask = np.zeros((tissue_mask.shape), dtype='uint8')
            lesion_benign_mask = np.zeros((tissue_mask.shape), dtype='uint8')
            lesion_malignant_mask = np.zeros((tissue_mask.shape), dtype='uint8')
            
        mask_to_overlook = np.zeros((tissue_mask.shape), dtype='uint8')
        
        if ignore_regions:
            label_image_tissue = label(tissue_mask)
            props_tissue = regionprops(label_image_tissue)
            for region in props_tissue:
                region_mask = np.zeros((true_mask.shape), dtype='uint8')
                y = region.coords[:,0]
                x = region.coords[:,1]
                region_mask[y,x]=255
                overlap = region_mask*true_mask
                if len(np.unique(overlap))==1:
                    mask_to_overlook[y,x]=255
        
 
        tissue_mask[true_mask!=0]=0
        tissue_mask = tissue_mask - mask_to_overlook
        tissue_mask[tissue_mask!=255]=0
        #if len(np.unique(true_mask))==1:
        #    print('WARNING! NO MASK FOUND FOR '+wsi)

        #create empty prediction masks
        pred_mask_lesion_benign = np.zeros((tissue_mask.shape), dtype='uint8')
        pred_mask_lesion_malignant = np.zeros((tissue_mask.shape), dtype='uint8')
        pred_mask_tissue = np.zeros((tissue_mask.shape), dtype='uint8')
        if self.return_prediction_mask:
            pred_mask = np.zeros((tissue_mask.shape[0],tissue_mask.shape[1],3), dtype='uint8')
        for coordinates, pred_y in zip(data_set_list, pred_y):
            top_left_coords_2_5x = coordinates['2.5x']
            x1_0_625x, y1_0_625x = int(top_left_coords_2_5x[0]/4), int(top_left_coords_2_5x[1]/4)
            x1_center, y1_center = x1_0_625x+32, y1_0_625x+32
            x1, y1 = x1_center-2, y1_center-2
            x2, y2 = int(x1+4), int(y1+4)
            if pred_y == self.class_dict['lesion benign']:
                rgb_channel = 1
                pred_mask_lesion_benign[y1:y2,x1:x2]=255
            elif pred_y == self.class_dict['lesion malignant']:
                rgb_channel = 0
                pred_mask_lesion_malignant[y1:y2,x1:x2]=255
            elif pred_y == 2:
                rgb_channel = [0,1]
            elif pred_y == self.tissue_label:
                rgb_channel = 2
                pred_mask_tissue[y1:y2,x1:x2]=255
            if self.return_prediction_mask:
                pred_mask[y1:y2,x1:x2,rgb_channel]=255
        
        
        pred_mask_lesion_benign = pred_mask_lesion_benign - mask_to_overlook
        pred_mask_lesion_benign[pred_mask_lesion_benign!=255]=0
        
        pred_mask_lesion_malignant = pred_mask_lesion_malignant - mask_to_overlook
        pred_mask_lesion_malignant[pred_mask_lesion_malignant!=255]=0
        
        
        predicted_mask_combined = pred_mask_lesion_benign+pred_mask_lesion_malignant
        #predicted_mask_combined[predicted_mask_combined!=0]=255
        
        predicted_mask_combined = predicted_mask_combined - mask_to_overlook
        predicted_mask_combined[predicted_mask_combined!=255]=0
             
        pred_mask_tissue = pred_mask_tissue - mask_to_overlook
        pred_mask_tissue[pred_mask_tissue!=255]=0
        
        intersection = true_mask*predicted_mask_combined*255
        union = true_mask+predicted_mask_combined
        union[union!=0]=255
        
        pred_mask[:,:,0] = pred_mask[:,:,0]-mask_to_overlook
        pred_mask[:,:,1] = pred_mask[:,:,1]-mask_to_overlook
        pred_mask[:,:,2] = pred_mask[:,:,2]-mask_to_overlook
        pred_mask[pred_mask!=255]=0
        
        
        tp = np.count_nonzero(intersection)
        fp = np.count_nonzero(union-intersection)
        tn = np.count_nonzero(pred_mask_tissue*tissue_mask)
        fn = np.count_nonzero(pred_mask_tissue*true_mask)
        
        #Metrics inside the annotations:
        tp_benign_inside_annotation = np.count_nonzero(pred_mask_lesion_benign*lesion_benign_mask)
        false_malignant_inside_benign_anno = np.count_nonzero(pred_mask_lesion_malignant*lesion_benign_mask)
        false_tissue_inside_benign_anno = np.count_nonzero(pred_mask_tissue*lesion_benign_mask)
        
        tp_malignant_inside_annotation = np.count_nonzero(pred_mask_lesion_malignant*lesion_malignant_mask)
        false_benign_inside_malignant_anno = np.count_nonzero(pred_mask_lesion_benign*lesion_malignant_mask)
        false_tissue_inside_malignant_anno = np.count_nonzero(pred_mask_tissue*lesion_malignant_mask)
        
        annotation_metrics_dict = {'tp benign': tp_benign_inside_annotation,
                                   'false malignant in benign': false_malignant_inside_benign_anno ,
                                   'false tissue in benign': false_tissue_inside_benign_anno,
                                   'tp malignant': tp_malignant_inside_annotation,
                                   'false benign in malignant': false_benign_inside_malignant_anno,
                                   'false tissue in malignant': false_tissue_inside_malignant_anno,
                                   'all benign pixels': np.count_nonzero(lesion_benign_mask),
                                   'all malignant pixels': np.count_nonzero(lesion_malignant_mask)}
        
        if not self.return_prediction_mask:
            pred_mask = None
        return tp, tn, fp, fn, pred_mask, annotation_metrics_dict
            