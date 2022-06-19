# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:57:36 2022

@author: Roger Amundsen
"""

from __future__ import print_function
from __future__ import division

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
import cv2
from PIL import Image

#import homemade python scripts
import create_sets
import transformation_depot
import pretrained_spyder_main
import preprocessing_main
from inference_functions import Run_inference, Inference_dataset, Return_model_metrics
from matplotlib.patches import Patch

torch.cuda.empty_cache()

# vipshome = 'C:\\vips-dev-8.10\\bin'

# # set PATH
# os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

# string = os.environ['PATH']
# new_string=string.split(';')[0]
# a=string.split(';')[0]

# for path in string.split(';')[1:]:
#     if a != path:
#         new_string=new_string+';'+path
#     a = path

# os.environ['PATH']=new_string

# import pyvips


def calculate_metrics(tp, tn, fp, fn):
    recall = tp/(tp+fn) if tp>0 else 0.0
    precision = tp/(tp+fp) if tp>0 else 0.0
    specificity = tn/(tn+fp) if tn > 0 else 0.0
    false_positive_rate = 1 - specificity
    f1 = (2*recall*precision)/(recall+precision) if recall > 0.0 else 0.0
    return recall, precision, specificity, false_positive_rate, f1

def classify(predicts, threshold, tissue_label):
    classified_y = []
    for prob_array in predicts:
          max_prob = np.max(prob_array)
          if max_prob < threshold:
              classified_y.append(tissue_label)
          else:
              classified_y.append(np.argmax(prob_array).item())
    return classified_y

def get_true_label(wsi):
    wsi = wsi.split(' ')[0]
    wsi = int(wsi[6:])
    return 0 if wsi < 51 else 1

def classify_wsi(pred_image, threshold):
    malignant, benign = pred_image[:,:,0], pred_image[:,:,1]
    malignant_pixels = np.count_nonzero(malignant)
    benign_pixels = np.count_nonzero(benign)
    sum_pixels = benign_pixels + malignant_pixels
    if sum_pixels == 0:
        print('No lesion found')
        return None
    rate = malignant_pixels/sum_pixels
    if rate >= threshold:
        return 1
    else:
        return 0


class Inference:
    def __init__(self,
                 inference_metrics_path,
                 model_path,
                 data_set_path,
                 probability_save_path,
                 batch_size,
                 wsi_mask_path,
                 weights_name,
                 wsi_path):
        self.inference_metrics_path = inference_metrics_path
        self.model_path = model_path
        self.data_set_path = data_set_path
        self.probability_save_path = probability_save_path
        self.batch_size = batch_size
        self.wsi_mask_path = wsi_mask_path
        self.weights_name = weights_name
        self.wsi_path = wsi_path
        with open(self.inference_metrics_path+'best_weights.json', 'r') as file:
            self.weight_list = json.load(file)
            
        with open(self.inference_metrics_path+'best_threshold.json', 'r') as file:
            threshold_dict = json.load(file)
            
        for weights, values in threshold_dict.items():
            if weights == self.weights_name:
                self.threshold = values['best threshold']
          
    def find_weights_dict(self):
        self.weights = {}
        for model in self.weight_list:
            if model['weight_name']==self.weights_name:
                self.weights = model
                print('weights_dict found')

    def store_probabilities_from_model(self):
        '''Runs inference on and stores probabilities of a model
        This function will iterate through all WSIs in the WSI scan file folder and check if they have corresponding coordinates in the coordinates folder.
        If cooresponding coordinates are found, inference will run.
        The predicted probabilities are saved in a file that will be used further to create masks/overview images and slide-based predictions in other functions'''
            
        inference_dictionary = {}
        instance = Run_inference(self.weights, batch_size = self.batch_size)
        instance.create_model(self.model_path)
        
        for wsi_scan_file_name in os.listdir(self.wsi_path):
            for wsi_coordinates_name in os.listdir(self.data_set_path):
                if wsi_coordinates_name[:-18] == wsi_scan_file_name[:-5]:
                    
                    instance.create_dataset(os.path.join(self.data_set_path,wsi_coordinates_name), Inference_dataset)
                    pred_y = instance.inference()
                    inference_dictionary[wsi_coordinates_name]=pred_y
                    
                    with open(self.probability_save_path+self.weights['weight_name'][:-3]+'.obj', 'wb') as handle:
                        pickle.dump(inference_dictionary, handle)
                    
    def store_prediction_images(self, path):
        '''Creates prediction_images and stores them in a given path'''
        
        with open(self.probability_save_path+self.weights_name[:-3]+'.obj', 'rb') as handle:
            prob_dict = pickle.load(handle)
        
        #initiate metrics from inside annotations
        tp_benign = 0
        false_mal_in_ben_anno = 0
        false_tissue_in_ben_anno = 0
        tp_malignant = 0
        false_ben_in_mal_anno = 0
        false_tissue_in_mal_anno = 0
        all_benign_pixels = 0
        all_malignant_pixels = 0
        
        
        for wsi, probabilities in prob_dict.items():
            
            pred_y = classify(probabilities, self.threshold, 20)
            return_model_metrics = Return_model_metrics(self.wsi_mask_path, self.weights['class_dict'], True, self.weights_name, self.inference_metrics_path)
            tp, tn, fp, fn, predicted_mask, annotation_metrics_dict = return_model_metrics.return_metrics_from_threshold(wsi, pred_y, self.data_set_path, ignore_regions=False)
            #print(annotation_metrics_dict)
            path_wsi = path+self.weights['weight_name'][:-3]+'/'+wsi[:-4]+'/'
            os.makedirs(path_wsi, exist_ok=True)
            cv2.imwrite(path_wsi+'threshold_{}.png'.format(self.threshold), cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2BGR))
            
            #metrics from inside annotations
            tp_benign += annotation_metrics_dict['tp benign']
            false_mal_in_ben_anno += annotation_metrics_dict['false malignant in benign']
            false_tissue_in_ben_anno += annotation_metrics_dict['false tissue in benign']
            tp_malignant  += annotation_metrics_dict['tp malignant']
            false_ben_in_mal_anno += annotation_metrics_dict['false benign in malignant']
            false_tissue_in_mal_anno += annotation_metrics_dict['false tissue in malignant']
            all_benign_pixels += annotation_metrics_dict['all benign pixels']
            all_malignant_pixels += annotation_metrics_dict['all malignant pixels']
            
        
            
    def get_best_mel_ben_threshold(self):
        with open(self.inference_metrics_path+'mal-ben_ratio_threshold.json', 'r') as file:
            mel_ben_threshold_dict = json.load(file)
        for threshold, values in mel_ben_threshold_dict.items():
            if values['recall']==1.0 and values['fpr']==0:
                self.threshold = float(threshold)
                print('Ratio threshold: {}'.format(self.threshold))
                break

    def classify_wsi_from_pred_images(self, path):
        for wsi in os.listdir(path+self.weights_name[:-3]):
            path_wsi = path+self.weights['weight_name'][:-3]+'/'+wsi+'/'
            image_name = os.listdir(path_wsi)[0]
            prediction_image = cv2.imread(path_wsi+image_name)
            prediction_image = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2RGB)
            y = classify_wsi(prediction_image, self.threshold)
            #y_true = get_true_label(wsi)
            plt.figure(figsize=(10,10))
            plt.imshow(prediction_image)
            #plt.title('predicted: '+str(y)+'  true: '+str(y_true))
            os.makedirs(self.inference_metrics_path+'test/classified/', exist_ok=True)
            plt.title('predicted: '+str(y))
            plt.savefig(self.inference_metrics_path+'test/classified/'+wsi[:8]+'.png', bbox_inches='tight')
            plt.show()
            
    def save_true_mask(self):
        '''Save mask with colored annotations
        Blue: Tissue (not annotated)
        Red: Lesion Malignant
        Green: Lesion Benign
        Yellow: Normal Tissue'''
        
        for wsi in os.listdir(self.data_set_path):
            #load masks
            tissue_mask = cv2.imread(self.wsi_mask_path+'{}/tissue_mask.png'.format(wsi[:-18]),0)
            lesion_benign_mask = cv2.imread(self.wsi_mask_path+'{}/lesion benign.png'.format(wsi[:-18]),0)
            lesion_malignant_mask = cv2.imread(self.wsi_mask_path+'{}/lesion malignant.png'.format(wsi[:-18]),0)
            lesion_malignant_mask[tissue_mask==0]=0
            normal_tissue_mask = cv2.imread(self.wsi_mask_path+'{}/normal tissue.png'.format(wsi[:-18]),0)
            normal_tissue_mask[tissue_mask==0]=0
            tissue_mask = tissue_mask - lesion_malignant_mask - lesion_benign_mask - normal_tissue_mask
            true_mask = np.zeros((tissue_mask.shape[0], tissue_mask.shape[1], 3), dtype='uint8')
            true_mask[:,:,0] = lesion_malignant_mask
            true_mask[:,:,1] = lesion_benign_mask
            true_mask[:,:,2] = tissue_mask
            true_mask[:,:,0][normal_tissue_mask==255] = 255
            true_mask[:,:,1][normal_tissue_mask==255] = 255
            #blue_patch = Patch(color='blue', label='Tissue (not annotated)')
            #red_patch = Patch(color='red', label='Lesion Malignant')
            #green_patch = Patch(color='green', label='Lesion Benign')
            #white_patch = Patch(color='yellow', label='Normal Tissue')
            plt.imshow(true_mask)
            #plt.legend(handles=[blue_patch, red_patch, green_patch, white_patch], bbox_to_anchor=(0.5, 0.0), borderpad=2)
            cv2.imwrite(self.wsi_mask_path+'{}/annotated_mask.png'.format(wsi[:-18]), cv2.cvtColor(true_mask, cv2.COLOR_RGB2BGR))
            plt.show()
            
            
            
def main(preprocess,
         wsi_path,
         wsi_mask_path,
         xml_path,
         data_set_path,
         model_path,
         inference_metrics_path,
         probability_save_path,
         weights_name,
         batch_size):
    inference = Inference(inference_metrics_path, model_path, data_set_path, probability_save_path, batch_size, wsi_mask_path, weights_name, wsi_path)
    if preprocess:
        preprocessing_main.main(background_segmentation=True,
                                create_masks_from_annotations = False, #If the WSI have noe xml-file of annotations, use False here.
                                extract_patches = True,
                                save_tiles_as_jpeg = False,
                                xml_path = xml_path,
                                wsi_path = wsi_path)
    inference.find_weights_dict()
    inference.store_probabilities_from_model()
    inference.store_prediction_images(inference_metrics_path+'test/')
    inference.get_best_mel_ben_threshold()
    inference.classify_wsi_from_pred_images(inference_metrics_path+'test/')
    #inference.save_true_mask()
    
    
                                

    
if __name__ == '__main__':
    
    main(preprocess = True,
         #wsi_path = 'WSI_all/',
         wsi_path = 'WSI_folder/',
         #wsi_path = '/home/prosjekt/Histology/Melanoma_SUS/MSc_Benign_Malign_HE/',
         wsi_mask_path = 'WSIs/',
         xml_path = 'xml/', #Path to xml-files of annotations
         #xml_path = '/home/prosjekt/Histology/.xmlStorage/',
         data_set_path = 'coordinates/',
         model_path = 'Models/',
         inference_metrics_path = 'inference/',
         probability_save_path = 'Models/inference/test_set/',
         weights_name = 'Model_17.pt',
         batch_size = 512)

