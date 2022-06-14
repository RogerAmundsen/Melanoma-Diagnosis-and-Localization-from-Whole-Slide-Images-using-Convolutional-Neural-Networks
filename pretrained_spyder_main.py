# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:13:29 2022

@author: Amund
"""

from __future__ import print_function
from __future__ import division
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import datetime
import numpy as np
import torchvision
#from torchvision import datasets, models, transforms
from torchvision import  models
import matplotlib.pyplot as plt
import time
import os
#os.environ['CUDA_LAUNCH_BLOCKING']="1"
import sys
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#import homemade python scripts
import create_sets
import test_model
import report_functions
import transformation_depot

import pickle
import random
#from torchsummary import summary
from torchinfo import summary 

from PIL import Image

#import math
import json

#from sklearn.metrics import multilabel_confusion_matrix,classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

# PyTorch TensorBoard support
#from torch.utils.tensorboard import SummaryWriter

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
#print(os.environ['PATH'])

# and now pyvips will pick up the DLLs in the vips area
import pyvips


def early_stopping(parameter, start_from, patience, min_delta=0.001):
    '''Apply early stopping to avoid overfitting.
    input:
        parameter: the parameter to be monitored while training, e.g. a list of validation accuracies.
        start_from: the minimum amount of epochs before early stopping can occur.
        patience: the amount of epochs to check the gradient between.
        min_delta: the minimum amount of gradient between the last element in parameter and the parameter[-patience-1]'th element
    output:
        True if stop, else False'''
    
    if len(parameter)<(start_from+patience+1):
        return False
    
    else:
        elements = [i.cpu().numpy() for i in parameter[-patience-1:-1]]
        #print('last {} epochs validation accuracy'.format(patience), elements)
        #gradient = (parameter[-1].cpu().numpy()-parameter[-patience-1].cpu().numpy())/patience
        gradient = np.mean(np.gradient(elements))
        print('Mean gradient last {} epochs'.format(patience), gradient)
        print(elements)
        if gradient < min_delta:
            return True
        else:
            return False


def train_model(model,
                classes,
                dataloaders,
                criterion,
                optimizer,
                model_path,
                device,
                patience,
                summary_file,
                report_batch = np.inf,
                num_epochs=25,
                is_inception=False):
    
    weight_path = model_path+'weights.pt'
    since = time.time()    
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    running_loss = 0.0
    running_corrects = 0.0  
    
    v_true, v_pred = test_model.run_test(model, device, dataloaders['val'])
    v_acc_before_training = report_functions.return_metrics(v_true, v_pred, classes)[0]['global']['acc']
    
    stat_dict = {'val_accuracy': [], 'val_loss': [], 'train_accuracy': [], 'train_loss': []}
    inter_epoch_metric = {'train_loss': [0], 'train_x_axis': [0],
                          #'val_loss': [0], 'val_x_axis': [0],
                          'train_acc': [0]}
                          #'val_acc': [v_acc_before_training]}
    
    for epoch in range(num_epochs):
        
              
        status = 'Epoch {}/{}'.format(epoch+1, num_epochs)
        print(status)
        print('-' * 10)
        
        with open(model_path+summary_file, 'a', encoding='utf-8') as f:
            f.write(status)
            f.write('\n'+'-'*10+'\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                #y_true, y_pred = test_model.run_test(model, device, dataloaders['val'])
                #test_metr = report_functions.return_metrics(y_true, y_pred)
                #print('val metrics: ')
                #print(test_metr)

            running_loss = 0.0
            running_corrects = 0
            inter_epoch_loss = 0.0
            
            # Iterate over data.
            number_of_batches = len(dataloaders[phase])
            inputs_so_far_in_current_epoch = 0
            
            for idx, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                #for tensor in inputs:
                #    plt.imshow(tensor.permute(1, 2, 0))
                #    plt.show()
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #print('inputs.size(0)', inputs.size(0))
                inputs_so_far_in_current_epoch += inputs.size(0)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        #print('outputs',outputs)
                        #print(outputs, labels)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    #print('preds',preds)
                    #print(labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #print(loss.item(), inputs.size(0))
                #print(loss.item()*inputs.size(0))
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if epoch == 0:
                    if idx % report_batch == report_batch-1:
                        x = epoch+(idx+1)/number_of_batches
                        inter_epoch_loss = running_loss/inputs_so_far_in_current_epoch
                        inter_epoch_acc = (running_corrects.double() / inputs_so_far_in_current_epoch).cpu().numpy()
                        if phase == 'train':
                            inter_epoch_metric['train_loss'].append(inter_epoch_loss)
                            inter_epoch_metric['train_x_axis'].append(x)
                            inter_epoch_metric['train_acc'].append(inter_epoch_acc)
                
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'train' and report_batch != np.inf and epoch == 0:
                inter_epoch_metric['train_loss'].append(epoch_loss)
                inter_epoch_metric['train_acc'].append(epoch_acc.cpu().numpy())
                inter_epoch_metric['train_x_axis'].append(epoch+1)
          
            epoch_score_string = '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
            print(epoch_score_string)
            
            with open(model_path+summary_file, 'a', encoding='utf-8') as f:
                f.write(epoch_score_string)
                f.write('\n'+'-'*10+'\n')
                
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), weight_path)
                print('saved model weights')
                with open(model_path+summary_file, 'a', encoding='utf-8') as f:
                    f.write('Saved model weights')
                    f.write('\n'+'-'*10+'\n')
            if phase == 'val':
                stat_dict['val_accuracy'].append(epoch_acc)
                stat_dict['val_loss'].append(epoch_loss)
            else:
                stat_dict['train_accuracy'].append(epoch_acc)
                stat_dict['train_loss'].append(epoch_loss)
                
            stop = early_stopping(stat_dict['val_accuracy'], 0, patience)
        
        val_acc = [v_acc_before_training] + [round(i.to('cpu').numpy().item(),4) for i in stat_dict['val_accuracy']]
        train_acc = [round(i.to('cpu').numpy().item(),4) for i in stat_dict['train_accuracy']]
        
        if stop:
            print('Early stopping')
            with open(model_path+summary_file, 'a', encoding='utf-8') as f:
                f.write('EARLY STOPPING due to stagnated validation accuracy')
                f.write('\n'+'-'*10+'\n')
            break
    
    time_elapsed = time.time() - since
    status = 'TRAINING COMPLETE in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    print(status)
    print('Best val Acc: {:4f}'.format(best_acc))
    
    with open(model_path+summary_file, 'a', encoding='utf-8') as f:
        f.write(status)
        f.write('\n'+'-'*10+'\n')
        f.write('Best val Acc: {:4f}'.format(best_acc))
        f.write('\n'+'-'*10+'\n')
        
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    stat_dict['val_accuracy'] = val_acc
    stat_dict['train_accuracy'] = train_acc
    
    return model, stat_dict, time_elapsed, inter_epoch_metric


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg11_bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif model_name == "vgg11":
        """ VGG11
        """
        model_ft = models.vgg11(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif model_name == "vgg16":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif model_name == "vgg16_bn":
        """ VGG16 with batch normalization
        """
        model_ft = models.vgg16_bn(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg16_nc":
        """ VGG16 new classifier
        """
        model_ft = models.vgg16(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[0].in_features
        model_ft.classifier[0] = nn.Linear(num_ftrs, 256)
        model_ft.classifier[3] = nn.Linear(256, 512)
        #num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(512,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


class dataset(Dataset):
    def __init__(self, 
                 transform,
                 data_set,
                 label_names,
                 augmentations = None,
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
        self.labels = label_names
        self.augmentations = augmentations
        
    def __len__(self):
         return len(self.data_set)
    def __getitem__(self, idx):
        path = self.data_set[idx]['path']
        #Convert top left coordinates to center coordinates
        x_coord = self.data_set[idx][self.magnification][0]
        y_coord = self.data_set[idx][self.magnification][1]
        #x_center, y_center = x_coord+256/2, y_coord+256/2
        #create new, cropped, top left coordinates
        #x_coord, y_coord = x_center-int(self.tile_size/2), y_center-int(self.tile_size/2)
        #Extract image from top left coordinates and tile size
        vips_object = pyvips.Image.new_from_file(path, level=self.mag_lvl[self.magnification], autocrop=True).flatten()
        tile_object = vips_object.extract_area(x_coord,y_coord, self.tile_size, self.tile_size)
        tile_image = np.ndarray(buffer=tile_object.write_to_memory(),
                            dtype='uint8',
                            shape=[tile_object.height, tile_object.width, tile_object.bands])
        
        if self.augmentations:
            if self.data_set[idx]['augmentation'] in self.augmentations:
                aug = self.data_set[idx]['augmentation']
                # plt.imshow(tile_image)
                # plt.show()
                if aug == 'flip':
                    tile_image = np.flipud(tile_image)
                elif aug == 'flop':
                    tile_image = np.fliplr(tile_image)
                elif aug == 'rot90':
                    tile_image = np.rot90(tile_image)
                elif aug == 'rot180':
                    tile_image = np.rot90(tile_image, 2)
                elif aug == 'rot270':
                    tile_image = np.rot90(tile_image, 3)
                # plt.imshow(tile_image)
                # plt.title(aug)
                # plt.show()
                    
        tile_image = Image.fromarray(tile_image)
        tile_image = self.transform(tile_image)
        
        # if self.augmentations:
        #     if self.data_set[idx]['augmentation'] in self.augmentations:
        #         plt.imshow(tile_image.permute(1, 2, 0))
        #         plt.title('transformed')
        #         plt.show()
        
        # plt.imshow(tile_image.permute(1, 2, 0))
        # plt.title('transformed')
        # plt.show()
        
        label = self.labels[self.data_set[idx]['tissue_type']]
        
        return tile_image, label
        
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
        
    
    def label_fast(self, idx):
        return self.labels[self.data_set[idx]['tissue_type']]


def main(parameters):
    '''Main function to run training on premade models
    input:
        parameters: dictionary containing all parameters. Dictionary is located in the same folder as the script.
    '''
    
    
    #unpack parameters
    # Models to choose from [resnet, alexnet, vgg11, vgg16, squeezenet, densenet, inception]
    model_architecture = parameters['network']
    batch = int(parameters['batch'])
    print('batch', batch)
    batch_val = int(parameters['batch_val'])
    print('batch val', batch_val)
    tiles_pr_class_pr_wsi = int(parameters['tiles']) if parameters['tiles'] else None #tiles pr class shuffled from all WSIs within a set
    val_tiles = int(parameters['val_tiles']) if parameters['val_tiles'] else None
    even_classes = parameters['even_classes']
    test = bool(int(parameters['test']))
    train = bool(int(parameters['train']))
    initial_learning_rate = float(parameters['lr'])
    num_epochs = int(parameters['epochs'])
    magnification_level = parameters['magnification']
    classes = parameters['classes']
    patience = parameters['patience']
    report_batch = parameters['report_batch'] if parameters['report_batch'] else np.inf
    train_path = parameters['train_path']
    val_path = parameters['val_path']
    test_path = parameters['test_path']
    feature_extract = parameters['feature_extract'] # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
    image_net_normalization = feature_extract #Normalize images to ImageNet mean and std if using models pretrained on ImageNet
    train_transformation_name = parameters['transformation_type_train']
    val_transformation_name = parameters['transformation_type_val']
    only_stats = parameters['only get dataset stats do not run']
    get_mean_std = parameters['get_mean_std'] if only_stats else None
    return_probabilities = parameters['return_probabilities']
    threshold = parameters['threshold']
    weight_name = parameters['weights_name']
    if only_stats:
        train = False
        test = False
        
    augmentations = parameters['augmentations']
    if feature_extract:
        print('feature extract TRUE')

    #get the current date and time
    date_time = datetime.datetime.now()
    date_time = date_time.strftime("%c")
    date_time_keep = date_time
    date_time = date_time.replace(' ', '_').replace(':', '_')
    
    #Create name for summary txt file
    summary_name = 'summary_{}.txt'.format(date_time)
    
    # Number of classes in the dataset
    num_classes = len(classes)
    
    # Initialize the model for this run
    model, input_size = initialize_model(model_architecture, num_classes, feature_extract, use_pretrained=True)
    
    summary_string = str(summary(model, (batch, 3,input_size,input_size)))
    print(summary_string)
    
    print("Initializing Datasets and Dataloaders...")
    
    random.seed(42)
    classes_string = ""
    for class_ in classes:
        classes_string += class_+'_' 
    
    model_name = '{}_{}_imgs_pr_wsi_bs_{}_mag_{}_classes_{}_freeze_{}'.format(model_architecture,
                                                                              tiles_pr_class_pr_wsi,
                                                                              batch,
                                                                              magnification_level,
                                                                              classes_string, 
                                                                              str(feature_extract))
    model_path = 'Models/'+model_name+'/'

    os.makedirs(model_path, exist_ok=True)

    # fix torch random seed
    torch.manual_seed(0)
    
    # set number of workers
    num_workers = 0

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device='cpu'
    # Send the model to GPU
    model = model.to(device)
    
    with open(model_path+summary_name, 'w', encoding='utf-8') as f:
        json.dump(parameters, f, ensure_ascii=False, indent=4)
        f.write('\n'+ '-'*20+ '\n')
        f.write('Model Name: '+ model_name+' Model path: '+model_path)
        f.write('\n'+ '-'*20+ '\n')
        f.write('Time: '+ date_time_keep)
        f.write('\n'+ '-'*20+ '\n')
        f.write('Train: '+str(train)+' Test: '+str(test))
        f.write('\n'+ '-'*20+ '\n')
        f.write('Model: ' +model_architecture+'\n'+summary_string)
        f.write('\n'+ '-'*20+ '\n')
        f.write('Feature Extract: '+ str(feature_extract))
        f.write('\n'+ '-'*20+ '\n')
        f.write('Device: '+str(device)+' Num workers: '+str(num_workers))
        f.write('\n'+ '-'*20+ '\n')
    
    val_set = create_sets.unpack_all_into_single_batch(val_path, classes, val_tiles, even_classes=even_classes)
    val_transformation = transformation_depot.get_transformation(val_transformation_name, input_size, image_net_normalization)
    validationset = dataset(val_transformation, val_set, classes, magnification=magnification_level)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_val, shuffle=True, num_workers=num_workers)
    if only_stats:
        train_set = create_sets.unpack_all_into_single_batch(train_path, classes, tiles_pr_class_pr_wsi, even_classes=even_classes)    
        train_transformation = transformation_depot.get_transformation(train_transformation_name, input_size, image_net_normalization)
        trainset = dataset(train_transformation, train_set, classes, augmentations=augmentations, magnification=magnification_level)
        
        #Report statistics from the train and validation set
        train_stats = report_functions.set_stats(trainset, 'train', batch, classes)
        with open(model_path+'train_stats_{}.json'.format(date_time), 'w', encoding='utf-8') as f:
            json.dump(train_stats, f, ensure_ascii=False, indent=4)
        print('Trainset:',train_stats)
        val_stats = report_functions.set_stats(validationset, 'val', batch_val, classes)
        with open(model_path+'val_stats_{}.json'.format(date_time), 'w', encoding='utf-8') as f:
            json.dump(val_stats, f, ensure_ascii=False, indent=4)
        print('Validation set:',val_stats)
        
        # load test data
        test_set = create_sets.unpack_all_into_single_batch(test_path, classes)
        
        #Get transformation object
        test_transformation = transformation_depot.get_transformation(val_transformation_name, input_size, image_net_normalization)
        
        #Create Pytorch dataset object
        testset = dataset(test_transformation, test_set, classes, magnification=magnification_level)
        
        
        if get_mean_std:
            all_images = train_set#+val_set+test_set
            mean_all,std_all = report_functions.get_mean_and_std_of_dataset(all_images, '10x', 256)
            print('mean', mean_all, 'std', std_all)
            with open(model_path+summary_name, 'a', encoding='utf-8') as f:
                f.write('Mean and std of {} images of type Tensor: {}, {}'.format(len(all_images), mean_all, std_all))
                f.write('\n'+ '-'*20+ '\n')
        
        
        #Repoort test stats
        test_stats = report_functions.set_stats(testset, 'test', batch_val, classes)
        with open(model_path+'test_stats_{}.json'.format(date_time), 'w', encoding='utf-8') as f:
            json.dump(test_stats, f, ensure_ascii=False, indent=4)
        print('Test set:',test_stats)
        
    
    if train: 
        # load train data
        train_set = create_sets.unpack_all_into_single_batch(train_path, classes, tiles_pr_class_pr_wsi, even_classes=even_classes)
        #add augmented copies to increase the trainset
        if augmentations:
            train_set += create_sets.add_augmented_patches(train_set, classes, augmentations, shuffle=True)
        #val_set = create_sets.unpack_all_into_single_batch(val_path, classes, val_tiles, even_classes=even_classes)
        
        #get transformation objects
        train_transformation = transformation_depot.get_transformation(train_transformation_name, input_size, image_net_normalization)
        #val_transformation = transformation_depot.get_transformation(val_transformation_name, input_size, image_net_normalization)
        
        
        #Create Pytorch dataset objects
        trainset = dataset(train_transformation, train_set, classes, augmentations=augmentations, magnification=magnification_level)
        
        #validationset = dataset(val_transformation, val_set, classes, magnification=magnification_level)
        
        #Report statistics from the train and validation set
        train_stats = report_functions.set_stats(trainset, 'train', batch, classes)
        with open(model_path+'train_stats_{}.json'.format(date_time), 'w', encoding='utf-8') as f:
            json.dump(train_stats, f, ensure_ascii=False, indent=4)
        print('Trainset:',train_stats)
        val_stats = report_functions.set_stats(validationset, 'val', batch_val, classes)
        with open(model_path+'val_stats_{}.json'.format(date_time), 'w', encoding='utf-8') as f:
            json.dump(val_stats, f, ensure_ascii=False, indent=4)
        print('Validation set:',val_stats)
        
        #Create Pytorch dataloaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=num_workers)
        #validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_val, shuffle=True, num_workers=num_workers)
        dataloaders_dict = {'train': trainloader, 'val': validationloader}
        
        #Train model
        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model.parameters()
        param_names = []
        print("Parameters to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    param_names.append(name)
                    print("\t",name)
        else:
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    param_names.append(name)
                    print("\t",name)
    
        # Observe that all parameters are being optimized

        #optimizer_ft = optim.SGD(params_to_update, lr=initial_learning_rate, momentum=0.9) #orginal
        optimizer_ft = optim.Adam(params_to_update, lr=initial_learning_rate)
        
        # Setup the loss fxn
        #orginal
        #criterion = nn.NLLLoss(reduction="sum")
        criterion = nn.CrossEntropyLoss()
        
        with open(model_path+summary_name, 'a', encoding='utf-8') as f:
            json.dump(train_stats, f, ensure_ascii=False, indent=4)
            f.write('\n'+ '-'*20+ '\n')
            json.dump(val_stats, f, ensure_ascii=False, indent=4)
            f.write('\n'+ '-'*20+ '\n')
            f.write('Initial learning rate: '+str(initial_learning_rate))
            f.write('\n'+ '-'*20+ '\n')
            f.write('Optimizer: '+str(optimizer_ft))
            f.write('\n'+ '-'*20+ '\n')
            f.write('Loss function: '+str(criterion))
            f.write('\n'+ '-'*20+ '\n')
            f.write('Parameters to update: '+str(param_names))
            f.write('\n'+ '-'*20+ '\n')
            f.write('Train transformation:\n')
            f.write(str(train_transformation))
            f.write('\n'+ '-'*20+ '\n')
            f.write('Val transformation:\n')
            f.write(str(val_transformation))
            f.write('\n'+ '-'*20+ '\n')
            f.write('TRAINING STARTS:')
            f.write('\n'+ '-'*20+ '\n')
            
            

        model, hist, train_time, inter_epoch = train_model(model,
                                                           classes,
                                                           dataloaders_dict,
                                                           criterion,
                                                           optimizer_ft,
                                                           model_path,
                                                           device=device,
                                                           num_epochs=num_epochs,
                                                           summary_file=summary_name,
                                                           report_batch=report_batch,
                                                           is_inception=(model_architecture=="inception"),
                                                           patience=patience)
    
        
        hist['val_loss'] = [round(i,4) for i in hist['val_loss']]
        hist['train_loss'] = [round(i,4) for i in hist['train_loss']]
    
        report_functions.save_plot(hist, model_path, date_time, inter_epoch)
        with open(model_path+'train_hist_{}.json'.format(date_time), 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False, indent=4)
        
        train_parameters = {'name': model_architecture, 'pretrained': str(feature_extract), 'epochs': num_epochs, 'initial lr': initial_learning_rate, 'optimizer': optimizer_ft,
                            'criterion': criterion, 'time': date_time_keep, 'train_time': train_time}
        
        open_obj = open(model_path+'train_parameters_{}.obj'.format(date_time), 'wb')
        pickle.dump(train_parameters, open(model_path+'train_parameters_{}.obj'.format(date_time), 'wb'))
        open_obj.close()
        
        
        
        # print('before initiating writer')
        # # Default log_dir argument is "runs" - but it's good to be specific
        # # torch.utils.tensorboard.SummaryWriter is imported above
        # writer = SummaryWriter('runs/test')
        # test_image = trainset[0][0]
        # print(type(test_image))
        # plt.imshow(test_image.permute(1,2,0))
        # plt.show()
        # # Write image data to TensorBoard log dir
        # writer.add_image('Example image', test_image)
        # writer.flush()
        
        # # To view, start TensorBoard on the command line with:
        # #   tensorboard --logdir=runs
        # # ...and open a browser tab to http://localhost:6006/
        
        # print('before changing model to cpu')
        # model = model.to('cpu')
        # dataiter = iter(trainloader)
        # images, labels = dataiter.next()
        # images, labels = images.to('cpu'), labels.to('cpu')
        # # add_graph() will trace the sample input through your model,
        # # and render it as a graph.
        # writer.add_graph(model, images)
        # #writer.flush()
        # plt.imshow(images[0].permute(1,2,0))
        # plt.show()
        
        # model.to(device)
       
        
       
        
       
    
    #Try to load trained model before testing
    try:
        #load the state dict of the trained model to run for testing
        model.load_state_dict(torch.load(model_path+weight_name))
        status = 'model state dict loaded'
        print(status)
        
    except:
        status = 'No trained model found'
        print(status)
        
    with open(model_path+summary_name, 'a', encoding='utf-8') as f:
        f.write(status)
        f.write('\n'+'-'*20+'\n')
        
    
        
    #Run test on either the test set or validation set after training
    if test:
        
        print('model loaded')
        
        # load test data
        test_set = create_sets.unpack_all_into_single_batch(test_path, classes)
        
        #Get transformation object
        test_transformation = transformation_depot.get_transformation(val_transformation_name, input_size, image_net_normalization)
        
        #Create Pytorch dataset object
        testset = dataset(test_transformation, test_set, classes, magnification=magnification_level)
        
        #Repoort test stats
        test_stats = report_functions.set_stats(testset, 'test', batch_val, classes)
        with open(model_path+'test_stats_{}.json'.format(date_time), 'w', encoding='utf-8') as f:
            json.dump(test_stats, f, ensure_ascii=False, indent=4)
        print('Test set:',test_stats)
        
        #Create Pytorch dataloader
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=num_workers)
        
        test_set_type = 'Test set'
        with open(model_path+summary_name, 'a', encoding='utf-8') as f:
            f.write('Calculating metrics on {} set after training'.format(test_set_type))
            f.write('\n'+ '-'*20+ '\n')
        
        #run test
        true_y, predicted_y = test_model.run_test(model, device, testloader, return_probabilities)
        print('testing done')
        
        with open(model_path+summary_name, 'a', encoding='utf-8') as f:
            f.write('TEST START')
            f.write('\n'+ '-'*20+ '\n')
            f.write('Test transformations:\n')
            f.write(str(test_transformation))
            f.write('\n'+ '-'*20+ '\n')
            json.dump(test_stats, f, ensure_ascii=False, indent=4)
            f.write('\n'+ '-'*20+ '\n')
            #json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
    else:
        if not only_stats:
            test_set_type = 'Validation set'
            with open(model_path+summary_name, 'a', encoding='utf-8') as f:
                f.write('Calculating metrics on {} set after training'.format(test_set_type))
                f.write('\n'+ '-'*20+ '\n')
            true_y, predicted_y = test_model.run_test(model, device, validationloader, return_probabilities)
   
    if not only_stats:
        
        
        metrics_dict, predicted_y = report_functions.return_metrics(true_y, predicted_y, classes, threshold=threshold)
        target_classes = list(classes.keys())+['uncertain th={}'.format(threshold)] if return_probabilities else list(classes.keys())
        cm = confusion_matrix(true_y, predicted_y)
        report_functions.plot_confusion_matrix(cm, target_classes, model_path, date_time)
        metrics_dict['confusion_matrix'] = str(cm)
        metrics_dict['Set type']= test_set_type
        
        with open(model_path+'metrics_{}.json'.format(date_time), 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
            
        
        
        with open(model_path+summary_name, 'a', encoding='utf-8') as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
            f.write('\n'+ '-'*20+ '\n')
            
        print(classification_report(true_y, predicted_y, target_names=target_classes))
            

        
if __name__ == '__main__':
    with open('parameters.json', 'r') as file:
        list_of_parameters = json.load(file)

    
    print(torch.cuda.is_available())
    #if len(sys.argv)>1:
    #    parameter_names = sys.argv[1::2]
    #    parameter_values = sys.argv[::2][1:]
    #    for key, value in zip(parameter_names, parameter_values):
    #        parameters[key] = value
    for parameters in list_of_parameters:
        main(parameters)
        torch.cuda.empty_cache()
