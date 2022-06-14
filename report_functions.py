# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:28:46 2022

@author: Amund
"""
import pickle
import os
from collections import defaultdict
import numpy as np

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


import matplotlib.pyplot as plt

def tp_tn_fp_fn(true_y,pred_y):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i,j in zip(true_y, pred_y):
        if i == j and i == 1:
            tp +=1
        elif i == j and i == 0:
            tn += 1
        elif i != j and i == 0:
            fp += 1
        else:
            fn += 1
            
    return [tp, tn, fp, fn]
    

def return_metrics(true_y, pred_y, class_dict, threshold=0.6):
    '''Returns metrics pr. class.
    input:
        true_y: list of true labels
        pred_y: list of predicted labels
        class_dict: name of classes
    return:
        metric_dict'''
    
    #start by checking if pred_y is a list of probabilities or predictions
    if isinstance(pred_y[0], int):
        probabilities = False
    else:
        probabilities = True
    
    #Initialize
    #classes = list(np.unique(true_y))
    classes = list(class_dict.values())
    metric_dict = {i: {} for i in class_dict}
    
    #if pred_y is a list of probabilities, predict the class based on prob. assign to "other" if prob<threshold
    if probabilities:
        new_pred_y = []
        for prob_array in pred_y:
            max_prob = np.max(prob_array)
            if max_prob < threshold:
                new_pred_y.append(len(classes))
            else:
                new_pred_y.append(np.argmax(prob_array))
        pred_y = new_pred_y
        metric_dict['uncertain']= {}
        classes.append(len(classes))
        
    
    
    metric_dict['global']= {}
    
    accuracy = sum(1 for i,j in zip(true_y,pred_y) if i==j)/len(true_y)
    metric_dict['global']['acc']= round(accuracy, 4)
    
    p_micro = [[],[]]
    p_macro = 0
    p_weighted = 0
    
    r_micro = [[],[]]
    r_macro = 0
    r_weighted = 0
    
    f1_weighted = 0
    f1_macro = 0

    for label, class_name  in zip(classes,metric_dict):
        true_y_class = [1 if i ==label else 0 for i in true_y]
        pred_y_class = [1 if i ==label else 0 for i in pred_y]
        tp, tn, fp, fn = tp_tn_fp_fn(true_y_class, pred_y_class)
        support = true_y_class.count(1)
        
        precision = round(tp/(tp+fp),4) if tp >0 else 0.0 
        p_macro += precision
        p_micro[0].append(tp)
        p_micro[1].append(tp+fp)
        p_weighted += precision*support
        
        recall = round(tp/(tp+fn),4) if tp >0 else 0.0
        r_macro += recall
        r_micro[0].append(tp)
        r_micro[1].append(tp+fn)
        r_weighted += recall*support
        
        f1 = round(((2*precision*recall)/(precision+recall)), 4) if precision > 0.0 and recall > 0.0 else 0.0
        f1_weighted += f1*support
        f1_macro += f1
        
        specificity = round(tn/(tn+fp),4) if tn >0 else 0.0
        false_positive_rate = round(1 - specificity, 4)
        
        
        metric_dict[class_name]['precision']=precision
        metric_dict[class_name]['recall']=recall
        metric_dict[class_name]['f1']=f1
        metric_dict[class_name]['specificity']=specificity
        metric_dict[class_name]['false positive rate'] = false_positive_rate
        metric_dict[class_name]['support']= support
        
    p_macro = p_macro/len(class_dict)
    p_micro = sum(p_micro[0])/sum(p_micro[1])
    p_weighted = p_weighted/len(true_y)
    
    metric_dict['global']['precision macro avg'] = round(p_macro,4)
    metric_dict['global']['precision micro avg'] = round(p_micro,4)
    metric_dict['global']['precision weighted avg'] = round(p_weighted,4)
    
    r_macro = r_macro/len(class_dict)
    r_micro = sum(r_micro[0])/sum(r_micro[1])
    r_weighted = r_weighted/len(true_y)
    
    metric_dict['global']['recall macro avg'] = round(r_macro,4)
    metric_dict['global']['recall micro avg'] = round(r_micro,4)
    metric_dict['global']['recall weighted avg'] = round(r_weighted,4)
    
    f1_macro = f1_macro/len(class_dict)
    f1_micro = 2*(r_micro*p_micro)/(r_micro+p_micro)
    f1_weighted = f1_weighted/len(true_y)
    
    metric_dict['global']['f1 macro avg'] = round(f1_macro,4)
    metric_dict['global']['f1 micro avg'] = round(f1_micro,4)
    metric_dict['global']['f1 weighted avg'] = round(f1_weighted,4)
    
    
    return metric_dict, pred_y
        

def plot_confusion_matrix(cm,
                          target_names,
                          path,
                          name,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy    

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize='large')
    plt.title('Predicted label', fontsize='large')
    plt.colorbar()

    new_names = []
    if target_names is not None:
        for class_name in target_names:
            new_name = class_name.replace(' ', '\n')
            new_names.append(new_name)
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, new_names, fontsize='large')
        plt.yticks(tick_marks, new_names, rotation=45, fontsize='large')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize='large')
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label', fontsize='large')
    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize='large')
    #plt.xlabel('Predicted label', fontsize='large')
    plt.savefig(path+'cm_'+name+'.png')
    plt.show()
    


def set_stats(dataset_, data_type, batch_size, classes):
    label_list = []
    for i in range(len(dataset_)):
        label_list.append(dataset_.label_fast(i))
    
    stats_dict = {'{} patches'.format(data_type): len(dataset_),'batch size': batch_size}
    for idx, class_ in enumerate(classes):
        stats_dict[class_] = label_list.count(idx)
           
    return stats_dict

def save_report(list_of_dictionaries, path):
    
    if 'report.obj' in os.listdir(path):
        with open(path+'report.obj', 'rb') as handle:
            dict_to_save = pickle.load(handle)
            
    else:
        dict_to_save = defaultdict(list)
    
    for dictionary in list_of_dictionaries:
        for key, value in dictionary.items():
            dict_to_save[key].append(value)
    
    open_obj = open(path+'report.obj', 'wb')
    pickle.dump(dict_to_save, open(path+'report.obj', 'wb'))
    open_obj.close()
    
def save_plot(data, path, name, inter_epoch):
    
    color_train = 'tab:blue'
    color_val = 'orange'
    
    x_train = np.array(range(1,len(data['train_accuracy'])+1))
    x_val = range(len(data['val_accuracy']))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[1].plot(x_train, data['train_accuracy'], color=color_train)
    axes[1].plot(x_val, data['val_accuracy'], color=color_val)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(['Trainset', 'Valset'])
    axes[1].grid()
    axes[0].plot(x_train, data['train_loss'], color=color_train)
    axes[0].plot(x_train, data['val_loss'], color=color_val)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(['Trainset','Valset'])
    axes[0].grid()
    plt.savefig(path+name+'original'+'.png')
    plt.show()
    

    
    if len(inter_epoch['train_x_axis'])>1:

        # x_train = np.array(range(1,len(data['train_accuracy'])+1))
        # x_val = range(len(data['val_accuracy']))
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        # axes[1].plot(x_train, data['train_accuracy'])
        # axes[1].plot(inter_epoch['train_x_axis'][1:], inter_epoch['train_acc'][1:])
        # axes[1].plot(x_val, data['val_accuracy'])
        # #axes[1].plot(inter_epoch['val_x_axis'], inter_epoch['val_acc'])
        # axes[1].set_xlabel('Epoch')
        # axes[1].set_ylabel('Accuracy')
        # axes[1].legend(['Trainset', 'Inter epoch train', 'Valset'])
        # axes[1].grid()
        # axes[0].plot(x_train, data['train_loss'])
        # axes[0].plot(inter_epoch['train_x_axis'][1:], inter_epoch['train_loss'][1:], marker='x')
        # axes[0].plot(x_train, data['val_loss'])
        # #axes[0].plot(inter_epoch['val_x_axis'][1:], inter_epoch['val_loss'][1:])
        # axes[0].set_xlabel('Epoch')
        # axes[0].set_ylabel('Loss')
        # axes[0].legend(['Trainset', 'Inter epoch train', 'Valset'])
        # axes[0].grid()
        # plt.savefig(path+name+'inter_epochs'+'.png')
        # plt.show()
    
        stop_index_train = inter_epoch['train_x_axis'].index(1)+1
        x_train = np.array(range(1,len(data['train_accuracy'])+1))
        x_val = range(len(data['val_accuracy']))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        axes[1].plot(x_train, data['train_accuracy'], color=color_train)
        axes[1].plot(inter_epoch['train_x_axis'][1:stop_index_train], inter_epoch['train_acc'][1:stop_index_train],color=color_train, marker ='x')
        axes[1].plot(x_val, data['val_accuracy'], color = color_val)
        #axes[1].plot(inter_epoch['val_x_axis'][:stop_index_val], inter_epoch['val_acc'][:stop_index_val], color=color_val, marker = 'x')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend(['Trainset', 'Inter epoch train', 'Valset'])
        axes[1].grid()
        axes[0].plot(x_train, data['train_loss'], color = color_train)
        axes[0].plot(inter_epoch['train_x_axis'][1:stop_index_train], inter_epoch['train_loss'][1:stop_index_train], color=color_train, marker='x')
        axes[0].plot(x_train, data['val_loss'], color=color_val)
        #axes[0].plot(inter_epoch['val_x_axis'][1:stop_index_val], inter_epoch['val_loss'][1:stop_index_val], color=color_val, marker = 'x')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend(['Trainset', 'Inter epoch train', 'Valset'])
        axes[0].grid()
        plt.savefig(path+name+'inter_epoch0'+'.png')
        plt.show()
 
def get_mean_and_std_of_dataset(dataset, magnification, img_size):
    from torchvision import transforms
    import torch
    convert_to_tensor = transforms.ToTensor()
    #convert_to_tensor = transforms.Compose([transforms.ToTensor(),
    #                                        transforms.Normalize([0.8095,0.6654,0.7799],
    #                                                             [0.1341,0.1650,0.1139])])
        
    '''SLOW METHOD that returns mean and std for each channel of a dataset of tensors'''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    mag_lvl = {'40x':0, '20x':1, '10x':2, '5x':3,'2.5x':4, '1.25x':5, '0.625x':6}
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    #for i, data in enumerate(dataloader):
    for i, coordinate_dict in enumerate(dataset):
        path = coordinate_dict['path']
        #Convert top left coordinates to center coordinates
        x_coord = coordinate_dict[magnification][0]
        y_coord = coordinate_dict[magnification][1]
        #x_center, y_center = x_coord+256/2, y_coord+256/2
        #create new, cropped, top left coordinates
        #x_coord, y_coord = x_center-int(self.tile_size/2), y_center-int(self.tile_size/2)
        #Extract image from top left coordinates and tile size
        vips_object = pyvips.Image.new_from_file(path, level=mag_lvl[magnification], autocrop=True).flatten()
        tile_object = vips_object.extract_area(x_coord,y_coord, img_size, img_size)
        tile_image = np.ndarray(buffer=tile_object.write_to_memory(),
                            dtype='uint8',
                            shape=[tile_object.height, tile_object.width, tile_object.bands])
        data = convert_to_tensor(tile_image)

        if (i % 10000 == 0): print(i)
        data = data.squeeze(0)
        if (i == 0): size = data.size(1) * data.size(2)
        mean += data.sum((1, 2)) / size
    
    mean /= len(dataset)
    mean = mean.unsqueeze(1).unsqueeze(2)
        
    for i, coordinate_dict in enumerate(dataset):
        path = coordinate_dict['path']
        #Convert top left coordinates to center coordinates
        x_coord = coordinate_dict[magnification][0]
        y_coord = coordinate_dict[magnification][1]
        vips_object = pyvips.Image.new_from_file(path, level=mag_lvl[magnification], autocrop=True).flatten()
        tile_object = vips_object.extract_area(x_coord,y_coord, img_size, img_size)
        tile_image = np.ndarray(buffer=tile_object.write_to_memory(),
                            dtype='uint8',
                            shape=[tile_object.height, tile_object.width, tile_object.bands])
        data = convert_to_tensor(tile_image)
        
        if (i % 10000 == 0): print(i)
        data = data.squeeze(0)
        std += ((data - mean) ** 2).sum((1, 2)) / size
    
    std /= len(dataset)
    std = std.sqrt()
    return mean.squeeze(), std

#import create_sets
#dataset =   create_sets.unpack_all_into_single_batch('coordinates/val/', {'lesion benign': 0, 'lesion malignant':2,
#                                                                          'normal tissue': 2})


#mean, std = get_mean_and_std_of_dataset(dataset, '10x', 256)
            

    
        
        