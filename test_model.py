# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:06:28 2022

@author: Amund
"""
import torch

import torch.nn.functional as F

def run_test(model, device, test_set, output_probabilities=False):
    '''classify the testset
    input:
        model
        device 'cpu or cpu'
        test_set: test set
        output_probabilities: if True, return prediction of all classes as probabilities
        
    return:
        true labels
        predicted labels: as either list of integers of 0 or 1, or as list of arrays with probabilities of each class'''
    
    print('inside test function')
    model.eval()
    true_y = []
    predicted_y = []
    with torch.no_grad():
        for data in test_set:
            images, labels = data
            true_y += list(labels.numpy())
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if output_probabilities:
                predicted = F.softmax(outputs, dim=1)
                predicted_y += list(predicted.cpu().numpy())
            else:
                _, predicted = torch.max(outputs.data, 1)
                predicted_y += [i.item() for i in list(predicted.cpu().numpy())]
            
    return true_y, predicted_y