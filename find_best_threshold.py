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
from inference_functions import Run_inference, Inference_dataset, Return_model_metrics

from matplotlib.patches import Patch


torch.cuda.empty_cache()


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
    print(np.unique(pred_image))
    malignant, benign = pred_image[:,:,0], pred_image[:,:,1]
    #print(np.unique(malignant), np.unique(benign))
    malignant_pixels = np.count_nonzero(malignant)
    benign_pixels = np.count_nonzero(benign)
    sum_pixels = benign_pixels + malignant_pixels
    #if sum_pixels == 0:
    #    return 20
    rate = malignant_pixels/sum_pixels
        
       
    if rate >= threshold:
        return 1
    else:
        return 0

                        
class Main:
    '''Main class containing methods that can start inference or to get metrics from previously run and stored inference'''
    def __init__(self,
                 threshold_list, data_set_path,
                 probability_save_path,
                 mask_path,
                 model_path,
                 inference_metrics_path,
                 batch_size,
                 store_prediction_mask):
        self.threshold_list = threshold_list
        self.data_set_path = data_set_path
        self.probability_save_path = probability_save_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.model_path = model_path
        self.inference_metrics_path = inference_metrics_path
        self.store_prediction_mask = store_prediction_mask
        with open(self.inference_metrics_path+'best_weights.json', 'r') as file:
            self.weight_list = json.load(file)
        
    def store_probabilities_from_models(self):
        '''Runs inference on and stores probabilities of all models in the 'best_weights.json list.'''
        for weights in self.weight_list:
            print('Weights:', weights)
            inference_dictionary = {}
            instance = Run_inference(weights, batch_size = self.batch_size)
            instance.create_model(self.model_path)
            for wsi in os.listdir(self.data_set_path):
                instance.create_dataset(os.path.join(self.data_set_path,wsi), Inference_dataset)
                pred_y = instance.inference()
                inference_dictionary[wsi]=pred_y
                with open(self.probability_save_path+weights['weight_name'][:-3]+'.obj', 'wb') as handle:
                    pickle.dump(inference_dictionary, handle)
                    
    
    def get_model_metrics(self):
        '''creates a dictionary conatining recall, precision, and false positive rate, for all thresholds for all models'''
        self.metrics_dict = {}
        for weight in self.weight_list:
            
            inf_dict_name = weight['weight_name'][:-3]+'.obj'
            class_dict = weight['class_dict']
            print('Weights name:', inf_dict_name)
            
            instance = Return_model_metrics(self.mask_path, class_dict, self.store_prediction_mask, inf_dict_name, self.inference_metrics_path)
            
            with open(self.probability_save_path+inf_dict_name,'rb') as handle:
                inf_dict = pickle.load(handle)
            
            recall_list = []
            precision_list = []
            f1_list = []
            fpr_list = []
            specificity_list = []
            for threshold in self.threshold_list:
                tp_sum, tn_sum, fp_sum, fn_sum = 0, 0, 0, 0
                for wsi, wsi_probs in inf_dict.items():
                    if wsi == 'class_dict':
                        continue
                    y_pred = classify(wsi_probs, threshold, 20)
                    tp, tn, fp, fn, predicted_mask = instance.return_metrics_from_threshold(wsi, y_pred, self.data_set_path, ignore_regions=True)
                    if self.store_prediction_mask:
                        path = self.inference_metrics_path+weight['weight_name'][:-3]+'/'+wsi[:-4]+'/'
                        os.makedirs(path, exist_ok=True)
                        cv2.imwrite(path+'threshold_{}.png'.format(threshold), cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2BGR))
                    tp_sum += tp
                    tn_sum+= tn
                    fp_sum+=fp
                    fn_sum += fn
                recall, precision, specificity, false_positive_rate, f1 = calculate_metrics(tp_sum, tn_sum, fp_sum, fn_sum)
                #print(recall)
                
                recall_list.append(recall)
                precision_list.append(precision)
                specificity_list.append(specificity)
                f1_list.append(f1)
                fpr_list.append(false_positive_rate)
            self.metrics_dict[weight['weight_name']]= {'recall': recall_list, 'precision': precision_list, 'fpr': fpr_list, 'f1': f1_list, 'specificity': specificity_list}
            with open(self.inference_metrics_path+'metrics_dict.json', 'w', encoding='utf-8') as f:
                 json.dump(self.metrics_dict, f, ensure_ascii=False, indent=4)
                 
    def save_true_mask(self):
        '''Save mask with colored annotations
        Blue: Tissue (not annotated)
        Red: Lesion Malignant
        Green: Lesion Benign
        Yellow: Normal Tissue'''
        
        for wsi in os.listdir(self.data_set_path):
            #load masks
            tissue_mask = cv2.imread(self.mask_path+'{}/tissue_mask.png'.format(wsi[:-18]),0)
            lesion_benign_mask = cv2.imread(self.mask_path+'{}/lesion benign.png'.format(wsi[:-18]),0)
            lesion_malignant_mask = cv2.imread(self.mask_path+'{}/lesion malignant.png'.format(wsi[:-18]),0)
            lesion_malignant_mask[tissue_mask==0]=0
            normal_tissue_mask = cv2.imread(self.mask_path+'{}/normal tissue.png'.format(wsi[:-18]),0)
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
            cv2.imwrite(self.mask_path+'{}/annotated_mask.png'.format(wsi[:-18]), cv2.cvtColor(true_mask, cv2.COLOR_RGB2BGR))
            plt.show()
            
            

    def make_plots(self, save):
        
        try:
            self.metrics_dict
        except:
            with open(self.inference_metrics_path+'metrics_dict.json', 'r') as file:
                self.metrics_dict = json.load(file)

        '''plots and stores ROC and recall/precision curves for each model'''
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize = (15,8))
        for model, values in self.metrics_dict.items():
            ax1.plot(values['fpr'], values['recall'], '-x', label=model)
            ax1.legend()
            ax2.plot(values['recall'], values['precision'], '-x', label=model)
            ax2.legend()
        #ax1.plot([0,1],[0,1], '--', color='gray')
        ax1.set_title('ROC')
        ax1.set_ylabel('True Positive Rate (recall)')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_xlim([0,1])
        ax1.set_ylim(0,1)
        ax1.grid()
            
        ax2.set_title('Recall/precision')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')    
        ax2.set_xlim([0,1])
        ax2.set_ylim([0,1])  
        ax2.grid()
        if save:
            plt.savefig(self.inference_metrics_path+'ROC.png')
        plt.show()
        
        fig = plt.figure(figsize=(10, 10))
        for model, values in self.metrics_dict.items():
            plt.plot(self.threshold_list, values['f1'], '-x', label=model)
            plt.legend()
        plt.grid()
        plt.xlabel('Threshold')
        plt.ylabel('F1 score')
        plt.xlim([0.9875,1.0025])
        if save:
            plt.savefig(self.inference_metrics_path+'F1.png')
        plt.show()
        
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize = (15,8))
        for model, values in self.metrics_dict.items():
            ax1.plot(values['recall'], values['specificity'], '-x', label=model)
            ax1.legend()
            ax2.plot(self.threshold_list, values['f1'], '-x', label=model)
            ax2.legend()
        #ax1.plot([0,1],[0,1], '--', color='gray')
        ax1.set_title('Recall - Specificity')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Specificity')
        ax1.set_xlim([0,1])
        ax1.set_ylim(0,1)
        ax1.grid()
            
        ax2.set_title('F1-score')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('F1')    
        plt.xlim([0.9875,1.0025])
        ax2.grid()
        if save:
            plt.savefig(self.inference_metrics_path+'Specificity_recall.png')
        plt.show()
        
    
    def find_best_threshold(self):
        '''Finds and stores the best threshold of all models, based on their F1 scores'''
        self.best_threshold_dict = {}
        for model, values in self.metrics_dict.items():
            f1_max = np.max(values['f1'])
            f1_max_idx = np.argmax(values['f1'])
            best_threshold = self.threshold_list[f1_max_idx]
            self.best_threshold_dict[model] = {'best threshold': best_threshold, 'f1': f1_max}
        
        with open(self.inference_metrics_path+'best_threshold.json', 'w', encoding='utf-8') as f:
             json.dump(self.best_threshold_dict, f, ensure_ascii=False, indent=4)
             
    def find_best_ratio_threshold(self, weights_name):
        
        with open(self.inference_metrics_path+'best_threshold.json', 'r') as file:
            threshold_dict = json.load(file)
            
        for weights, values in threshold_dict.items():
            if weights == weights_name:
                threshold = values['best threshold']
                print(threshold)
        
                path = self.inference_metrics_path+weights_name[:-3]+'/'
                
                recall_list = []
                fpr_list = []
                ratio_threshold_list = [0.0,0.01,0.02,0.03,0.04, 0.1,0.5,0.6,0.7,0.8,0.9,1.0]
                for ratio_threshold in ratio_threshold_list:
                    tp, tn, fp, fn = 0,0,0,0
                
                    for wsi in os.listdir(path):
                        true_y = get_true_label(wsi)
                        #path_image = path+wsi+'/'+image_name
                        for image_name in os.listdir(path+wsi+'/'):
                            thres = float(image_name.split('_')[-1][:-4])
                            if thres == threshold:
                                prediction_image = cv2.imread(path+wsi+'/'+image_name)
                                prediction_image = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2RGB)
                                y = classify_wsi(prediction_image, ratio_threshold)
                                if true_y != y:
                                    plt.imshow(prediction_image)
                                    plt.title('t: {}, true: {}, pred: {}'.format(ratio_threshold,true_y,y))
                                    plt.show()
                                if y == true_y and true_y==1:
                                    tp+=1
                                elif y == true_y and true_y==0:
                                    tn+=1
                                elif y != true_y and true_y ==0:
                                    fp+=1
                                elif y != true_y and true_y == 1:
                                    fn+=1
                    recall, precision, specificity, false_positive_rate, f1 = calculate_metrics(tp, tn, fp, fn)
                    recall_list.append(recall)
                    fpr_list.append(false_positive_rate)
                    
                    #plt.scatter(ratio_threshold, fp)
                    
                    
                    plt.scatter(false_positive_rate, recall, label = ratio_threshold)
                    plt.legend(title='Threshold')
                    #print('ratio tp tn fp fn')
                    #print(str(ratio_threshold)+'    '+ str(tp)+'  '+str(tn)+'  '+str(fp)+'  '+str(fn))
                    
                    
                plt.plot(fpr_list,recall_list, '--')
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate (recall)')
                plt.grid()
                plt.plot([0,1],[0,1], '--',color='gray')
                plt.xlim([-0.1,1.1])
                plt.ylim([-0.1,1.1])
                plt.savefig(self.inference_metrics_path+'{}.png'.format(weights_name))
                plt.show()
                
                ratio_threshold_dict = {}
                for idx, ratio in enumerate(ratio_threshold_list):
                    ratio_threshold_dict[ratio]= {'recall': recall_list[idx], 'fpr': fpr_list[idx]}
                with open(self.inference_metrics_path+'mal-ben_ratio_threshold.json', 'w', encoding='utf-8') as f:
                     json.dump(ratio_threshold_dict, f, ensure_ascii=False, indent=4)
        
        
                        


if __name__ == '__main__':
    
    main = Main(threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999997, 1],
                data_set_path = 'coordinates/inference_val/',
                probability_save_path = 'Models/inference/',
                mask_path='WSIs/', 
                model_path = 'Models/',
                inference_metrics_path = 'inference/',
                batch_size=512,
                store_prediction_mask = True)
                #batch_size=512)
    
    #main.store_probabilities_from_models()
    #main.get_model_metrics()
    #main.make_plots(save=True)
    #main.find_best_threshold()
    #main.save_true_mask()
    main.find_best_ratio_threshold('Model_17.pt')
    
