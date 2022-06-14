# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:45:10 2022

@author: Roger Amundsen
"""

#pip install pyvips==2.1.12
import os
#print('path =', os.getenv('PATH

# change this for your install location and vips version, and remember to 
# use double backslashes
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
from skimage import morphology
from scipy import ndimage
import numpy as np
import pickle
import time
import cv2
import matplotlib.pyplot as plt
import math

from skimage.measure import label, regionprops
#from skimage.filters import threshold_otsu
from skimage.morphology import closing, square

import xml.etree.ElementTree as ET

def find_regions(mask, threshold, remove=False):
    '''Finds individual regions in a mask. These regions are not connected to eachother.
    input:
        mask: e.g. lesion benign mask, normal tissue mask etc.
        remove: If true remove regions that have area smaller than threshold;
        threshold: min_size of a region. If remove is true, remove regions that are smaller than threshold
    return:
        labels: each region is labeled with a different integer'''
    
    # Use a boolean condition to find where pixel values are > 0.75
    blobs = mask > 0.75
    labels, regions_found_in_wsi_before_removing_small_obj = ndimage.label(blobs)
    print('\tFound {} regions'.format(regions_found_in_wsi_before_removing_small_obj))
    
    if remove:
    # Remove all the small regions
        labels = morphology.remove_small_objects(labels, min_size=threshold)
    return labels
        

def box_around_region(region):
    '''creates a square box around a region/polygon. This box will be used to scan for valid
    tiles instead of having to scan the whole WSI
    input:
        region: skimage region coordinates
    output:
        coorner coordinates of the box'''
    y_box_min = min(region.coords[:,0])
    x_box_min = min(region.coords[:,1])
    y_box_max = max(region.coords[:,0])
    x_box_max = max(region.coords[:,1])
    return [y_box_min, y_box_max, x_box_min, x_box_max]


def mask_overlay(dict_with_masks, alpha_chan):
    '''Draw a mask overlay on top of the output image'''
    for tissue in dict_with_masks:
        if tissue == 'lesion benign':
            dict_with_masks[tissue] = cv2.merge((dict_with_masks[tissue], alpha_chan, alpha_chan, alpha_chan))
        elif tissue == 'lesion malignant':
            dict_with_masks[tissue] = cv2.merge((alpha_chan, alpha_chan, dict_with_masks[tissue], alpha_chan))
        elif tissue == 'normal tissue':
            dict_with_masks[tissue] = cv2.merge((alpha_chan, dict_with_masks[tissue], alpha_chan, alpha_chan))
        else:
            dict_with_masks[tissue] = cv2.merge((alpha_chan, alpha_chan, alpha_chan, dict_with_masks[tissue]))
            
    return dict_with_masks

       
def tile_variance(pyvips_img, threshold):
    '''Find images with low variance: background images
    input:
        pyvips_img: pyvips object representation of an WSI image patch
        threshold: minimum variance a patch can have before it is disregarded
    return:
        False if patch is rejected or True if patch is approved'''
    
    vips2np = np.ndarray(buffer=pyvips_img.write_to_memory(), dtype='uint8', shape=[pyvips_img.height, pyvips_img.width, pyvips_img.bands])
    
    r, g, b = vips2np[:,:,0], vips2np[:,:,1], vips2np[:,:,2]
    r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
    std = np.std(np.array([r_mean, g_mean, b_mean]))
    mean_of_mean = np.mean(np.array([r_mean,g_mean,b_mean]))
    
    #if mean < threshold:
        
    #    plt.subplot(2,2,1)
    #    plt.imshow(vips2np)
    #    plt.subplot(2,2,2)
    #    plt.imshow(vips2np_hsv[:,:,0])
    #    plt.subplot(2,2,3)
    #    plt.imshow(vips2np_hsv[:,:,1])
    #    plt.subplot(2,2,4)
    #    plt.imshow(vips2np_hsv[:,:,2])
    #    plt.show()
    
    return False if mean_of_mean > threshold and std < 4.5 else True
    


class Preprocess:
    def __init__(self,
                 tissue_classes_to_fit_tiles_on = ['normal tissue', 'lesion benign', 'lesion malignant'],
                 mask_hierarchy = ['normal tissue', 'lesion benign', 'lesion malignant'],
                 phi =0.7,
                 alpha = '40x',
                 tau = '2.5x',
                 tiles_to_show = ['40x'],
                 tile_size = 256,
                 save_tiles_as_jpeg=False,
                 save_binary_annotation_mask=True,
                 remove_small_regions = False,
                 small_region_remove_threshold = 1000,
                 wsi_dataset_file_path = 'WSIs/',
                 wsi_path = 'WSI_all/',
                 output_folder = 'Output/',
                 coordinate_path = 'coordinates/',
                 extracted_tiles_folder = 'Extracted_tiles/',
                 skip_list = ['SUShud24 - 2021-11-17 10.53.41.ndpi',
                              'SUShud42 - 2021-12-06 11.24.59.ndpi',
                              'SUShud47 - 2021-12-06 11.33.10.ndpi',
                              'SUShud53 - 2021-12-06 11.53.26.ndpi',
                              'SUShud77 - 2021-12-06 13.52.35.ndpi',
                              'SUShud92 - 2021-12-07 16.12.10.ndpi',
                              'SUShud93 - 2021-12-07 16.23.26.ndpi',
                              'SUShud95 - 2021-12-07 16.35.31.ndpi',
                              'Master benign-malign diagnose.xlsx'],
                 magnification_levels = {0: '40x', 1: '20x', 2: '10x', 3: '5x',
                                         4: '2.5x', 5: '1.25x', 6: '0.625x'},
                 magnification_levels_to_save = ['40x','20x','10x', '2.5x'],
                 wsi_name = None
                 ):
        
        self.tissue_classes_to_fit_tiles_on = tissue_classes_to_fit_tiles_on
        self.mask_hierarchy = mask_hierarchy
        self.phi = phi
        self.alpha = alpha
        self.tau = tau
        self.tiles_to_show = tiles_to_show
        self.tile_size = tile_size
        self.save_tiles_as_jpeg = save_tiles_as_jpeg
        self.save_binary_annotation_mask = save_binary_annotation_mask
        self.remove_small_regions = remove_small_regions
        self.small_region_remove_threshold = small_region_remove_threshold
        self.wsi_dataset_file_path = wsi_dataset_file_path
        self.wsi_path = wsi_path
        self.output_folder = output_folder
        self.coordinate_path = coordinate_path
        self.extracted_tiles_folder = extracted_tiles_folder
        self.skip_list = skip_list
        self.magnification_levels = magnification_levels
        self.magnification_levels_to_save = magnification_levels_to_save
        self.wsi_name = wsi_name


    def make_directories(self):
        '''Create directories for preprocessing. Folders to save extracted tiles, masks, and overview image with tiles on'''
        os.makedirs(self.wsi_dataset_file_path, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.coordinate_path, exist_ok=True)
        for wsi_name in os.listdir(self.wsi_path):
            if wsi_name not in self.skip_list:
                os.makedirs(self.wsi_dataset_file_path+wsi_name[:-5], exist_ok=True)
        if self.save_tiles_as_jpeg:
            os.makedirs(self.extracted_tiles_folder, exist_ok=True)
            
    def load_wsi(self, levels=list(range(7))):
        '''Load pyvips wsi
        Input:
            path_name: path and file name
            magnification_level_names: Dictionary containing the wsi levels as keys and their names as values,
            eg: {0: '40x', 1: '20x', 2: '10x', 3: '5x'}.
            levels: magnification level/levels to load, list or single integer.
        Return:
            image_dictionary: dictionary with magnification levels as keys and pyvips objects as values
            area_ratio_dictionary: dictionary with magnification levels as keys and area ratios as values
        '''
        if isinstance(levels, int):
            levels = [levels]
        self.image_dictionary={}
        for i in levels:
            name = self.magnification_levels[i]
            self.image_dictionary[name] = pyvips.Image.new_from_file(self.wsi_path+self.wsi_name, level=i, autocrop=True).flatten()
        
        highest_magnification_name = self.magnification_levels[min(levels)]
        highest_magnification_area = self.image_dictionary[highest_magnification_name].width*self.image_dictionary[highest_magnification_name].height
        
        self.area_ratio_dictionary = {}
        for magnification, image in self.image_dictionary.items():
            image_area = image.width*image.height
            area_ratio = highest_magnification_area/image_area
            self.area_ratio_dictionary[magnification]= area_ratio
            
    def load_mask_dict(self):
        try:
            with open(self.wsi_dataset_file_path+self.wsi_name[:-5]+'/masks.obj', 'rb') as f:
                dict_ = pickle.load(f)
            self.mask_dict = {}
            for requested_class in self.tissue_classes_to_fit_tiles_on:
                if requested_class in dict_:
                    self.mask_dict[requested_class] = dict_[requested_class]
        except:
            self.mask_dict = None
                
    def remove_mask_overlap(self):
        '''remove from mask if it is overlapping with another and is below it in hierarchy'''
        for class1 in [i for i in self.mask_hierarchy if i in self.mask_dict]:
            class1_mask = self.mask_dict[class1]
            #print(class1, np.unique(class1_mask))
            for class2 in self.mask_hierarchy:
                if class1!=class2:
                    class2_mask = self.mask_dict[class2]
                    sum_masks = class1_mask+class2_mask
                    #print(class1, class2)
                    #print(np.unique(sum_masks))
                    if len(np.where(sum_masks == 2)[0])>0:
                        hierarchy_class1 = self.mask_hierarchy.index(class1)
                        hierarchy_class2 = self.mask_hierarchy.index(class2)
                        if hierarchy_class1< hierarchy_class2:
                            print('removing overlap between {} and {} from {}'.format(class1,class2,class2))
                            class2_mask[sum_masks==2]=0
                        else:
                            class1_mask[sum_masks==2]=0
                            print('removing overlap between {} and {} from {}'.format(class1,class2,class1))
                            
    def dynamic_TAU_size(self, threshold):
        '''Reduces tau to a lower magnification level if the number of pixels in the image is higher than a given threshold
        input:
            threshold: the max number of pixels'''
        highest_magnification = self.magnification_levels[min(self.magnification_levels.keys())]
        height = self.image_dictionary[highest_magnification].height
        width = self.image_dictionary[highest_magnification].width
        if height*width > threshold:
            self.tau = '1.25x'
        else:
            self.tau = '2.5x'
            
    def image_pyramid_measurements(self):
        '''calculates ratios and sizes of the images in the wsi_pyramid'''
        # Ratio between alpha and tau
        self.ratio_alpha_tau = self.area_ratio_dictionary[self.alpha]/self.area_ratio_dictionary[self.tau]

        #Define the tile size of alpha on the tau image
        self.tile_size_alpha_on_tau = int(np.sqrt(self.ratio_alpha_tau)*self.tile_size)
        
        #create a subset of levels that are not tau
        self.subset_of_levels = []
        for magnification in self.magnification_levels_to_save:
            if magnification != self.tau:
                self.subset_of_levels.append(magnification)

        # Find width/heigh of tau image
        self.tau_image_width = self.image_dictionary[self.tau].width
        self.tau_image_height = self.image_dictionary[self.tau].height
        
    def patch_criteria(self):
        '''calculate pathc criteria: the amount of pixels within a patch that has to belong to a mask (have the value 1 or 255) for the patch to be valid'''
        self.criteria = pow(self.tile_size,2)*self.ratio_alpha_tau * self.phi
        
    def possible_x_and_y_coordinates(self, min_max_coordinates):
        '''returns a list of all coordinates that has the potential to become a tile
        input:
            min_max_coorinates: the coordinates of a box around the region/polygon/ROI
            ratio: the ratio between the alpha layer and tau layer
            tile size: tile size to create a pad around the box
            image_shape: to make sure the range of coordinate does not extend outside of the image
            
        return:
            top left corner coorinates of potential tiles'''
        
        #tile_size_alpha_on_tau = int(tile_size*ratio) #find tile size at tau
        y_min, y_max = min_max_coordinates[0], min_max_coordinates[1]
        x_min, x_max = min_max_coordinates[2], min_max_coordinates[3]
        pad = int(self.tile_size/2)
        range_x = range(max(x_min-pad,0), min((x_max+pad)-self.tile_size,self.tau_image_width-self.tile_size), self.tile_size_alpha_on_tau)
        range_y = range(max(y_min-pad,0), min((y_max+pad)-self.tile_size,self.tau_image_height-self.tile_size), self.tile_size_alpha_on_tau)
        return (list(range_x), list(range_y))
    
    def find_valid_tiles(self, all_x_all_y_pos, mask):
        '''Finds valid tiles from alpha level by checking that the tile area contains enough mask area within it.
        input:
            criteria: the minimum ratio of mask area within the tile area.
            tile_size: tile size aka patch size
            tile_size_alpha_on_tau: the ratio between the area of a tile at alpha level and tau level
        return:
            list_of_valid_tiles_from_current_class: list of top left coordinates of valid tiles in the current class/tissue-type'''
        
        
        all_x, all_y = all_x_all_y_pos
        list_of_valid_tiles_from_current_class = []
        for y_pos in all_y:
            y_center = y_pos + int(self.tile_size/2)
            y_top = y_center - int(self.tile_size_alpha_on_tau/2)
            y_base = y_center + int(self.tile_size_alpha_on_tau/2)
            for x_pos in all_x:
                x_center = x_pos + int(self.tile_size/2)
                x_left = x_center - int(self.tile_size_alpha_on_tau/2)
                x_right = x_center + int(self.tile_size_alpha_on_tau/2)
                    
                # Equation 1 in paper
                if int(sum(sum(mask[y_top:y_base,
                                x_left:x_right]))) >= self.criteria:
                    list_of_valid_tiles_from_current_class.append((x_pos, y_pos))
        
        return list_of_valid_tiles_from_current_class
    
    def create_coordinate_dictionary(self, 
                                     dict_of_all_predicted_coordinates,
                                     list_of_valid_tiles,
                                     current_class,
                                     remove_tiles=True,
                                     threshold=150):
        '''creates a dictionary containing the coordinates of valid tiles for each magnification level for each WSI
        input: 
            valid_tiles_from_current_wsi: list of valid tiles given by their top left x- and y-coordinates [(x1,y1),(x2,y2)...]
            
        '''

        idx_of_valid_tiles= list(range(len(list_of_valid_tiles)))
        start_index = len(dict_of_all_predicted_coordinates.keys())
        id_number = start_index
        
        #for idx, current_xy_pos in enumerate(list_of_valid_tiles[start_index:]):
        for idx, current_xy_pos in enumerate(list_of_valid_tiles):
            approved_tile = True
            tile_x_tau = current_xy_pos[0]
            tile_y_tau = current_xy_pos[1]
            
            if remove_tiles:
                ratio_tau_alpha = self.area_ratio_dictionary[self.tau]/self.area_ratio_dictionary[self.alpha]
                tile_x_alpha = (tile_x_tau+(self.tile_size/2))*np.sqrt(ratio_tau_alpha)-self.tile_size/2
                tile_y_alpha = (tile_y_tau+(self.tile_size/2))*np.sqrt(ratio_tau_alpha)-self.tile_size/2
                tile_to_save = self.image_dictionary[self.alpha].extract_area(int(tile_x_alpha),int(tile_y_alpha),self.tile_size, self.tile_size)
                approved_tile = tile_variance(tile_to_save, threshold)
                
            
            if approved_tile:
                dict_of_all_predicted_coordinates[id_number] = dict()
                dict_of_all_predicted_coordinates[id_number]['path'] = self.wsi_path+self.wsi_name
                dict_of_all_predicted_coordinates[id_number]['tissue_type'] = current_class
                
                # Equation 2 in paper.
                for beta in self.magnification_levels_to_save:
                    ratio_tau_beta = self.area_ratio_dictionary[self.tau]/self.area_ratio_dictionary[beta]
                    tile_x = (tile_x_tau+(self.tile_size/2))*np.sqrt(ratio_tau_beta)-self.tile_size/2
                    tile_y = (tile_y_tau+(self.tile_size/2))*np.sqrt(ratio_tau_beta)-self.tile_size/2
                    
                    dict_of_all_predicted_coordinates[id_number][str(beta)] = (int(tile_x), int(tile_y))
                        
                    if self.save_tiles_as_jpeg:
                        tile_to_save = self.image_dictionary[beta].extract_area(int(tile_x),int(tile_y),self.tile_size, self.tile_size)
                        tile_to_save.jpegsave(self.extracted_tiles_folder + self.wsi_name[:-5] + '/tile_{}_{}_{}_{}.jpeg'.format(id_number,beta, int(tile_x), int(tile_y)), Q=100)
                id_number += 1
            else:
                idx_of_valid_tiles.remove(idx)
                
        list_of_valid_tiles = [list_of_valid_tiles[i] for i in idx_of_valid_tiles]
        return dict_of_all_predicted_coordinates, list_of_valid_tiles
    
    def save_coordinates(self, coordinates):
        with open(self.coordinate_path+self.wsi_name[:-5]+'_tile_size_'+str(self.tile_size)+'.obj', 'wb') as handle:
            pickle.dump(coordinates, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def draw_tiles_on_image(self, image, valid_tiles):
        '''Draws the valid tiles on the output image
        input:
            image
            valid_tiles: list of top-left coordinates of the valid tiles at TAU level
            '''
        for current_xy_pos in valid_tiles:
            # Equation 3 in paper.
            for beta in self.tiles_to_show:
                ratio_beta_tau = self.area_ratio_dictionary[beta]/self.area_ratio_dictionary[self.tau]

                center_x_tau = current_xy_pos[0]+(self.tile_size/2)
                center_y_tau = current_xy_pos[1]+(self.tile_size/2)

                start_x = int(center_x_tau-(self.tile_size/2)*np.sqrt(ratio_beta_tau))
                start_y = int(center_y_tau-(self.tile_size/2)*np.sqrt(ratio_beta_tau))

                end_x = int(start_x + self.tile_size*np.sqrt(ratio_beta_tau))
                end_y = int(start_y + self.tile_size*np.sqrt(ratio_beta_tau))

                color = (0, 0, 255) if beta == self.alpha else (0, 255, 0)
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 1)
                
    def save_overview_image(self, image, num_tiles):
        # Save overview image
        cv2.imwrite(self.output_folder + self.wsi_name[:-5] + '/image_with_mask_and_tiles_alpha_{}_phi_{}_{}.jpg'.format(self.alpha, self.phi, num_tiles), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        
class Foreground_background_segmentation(Preprocess):
    def __init__(self,
                 level=6,
                 hue_min=100,
                 hue_max=180,
                 size2remove=100):
        self.level = level
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.size2remove = size2remove
        super().__init__()
        
    def remove_small_regions_from_mask(self, mask):
        '''closes holes in mask and removes regions that are smaller than a threshold named size2remove.
        input: 
            mask
        return:
            closed mask'''
        
        # Closing of the image and gather the labels of the image
        mask = closing(mask, square(3))
        label_image = label(mask)

        # Run through the image labels and set the small regions to zero
        props = regionprops(label_image)
        for region in props:
            if region.area < self.size2remove:
                minY, minX, maxY, maxX = region.bbox
                mask[minY:maxY, minX:maxX] = 0

        return mask
    
    def create_tissue_mask(self, wsi_pyramid, display_mask=False):
        '''creates a tissue mask there tissue has the value 1 and background 0, based on thresholding.
        input:
            wsi_pyramid: a dictionary containing the WSI image on all maginification levels (in pyvips object format).
        return:
            tissue_mask'''
        
        magnification = self.magnification_levels[self.level]
        img = wsi_pyramid[magnification]
        img = np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8, shape=[img.height, img.width, img.bands])
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask_HSV = cv2.inRange(img_hsv, (self.hue_min, 0, 0), (self.hue_max, 255, 255))
        mask_HSV = self.remove_small_regions_from_mask(mask_HSV)
        
        # Close the holes in the image
        maskInv = cv2.bitwise_not(mask_HSV)
        maskInv_closed = self.remove_small_regions_from_mask(maskInv)
        mask_HSV = cv2.bitwise_not(maskInv_closed)
        
        mask_HSV[mask_HSV>0]=1
        self.mask_tissue = mask_HSV
        
        if display_mask:
            plt.imshow(self.mask_tissue)
            plt.show()
            
def write_to_pickle(path, name, dictionary):
    if name in os.listdir(path):
        with open(path+name, 'rb') as f:
            dict_ = pickle.load(f)
        for key in dictionary:
            dict_[key] = dictionary[key]
        with open(path+name, 'wb') as f:
            pickle.dump(dict_, f)  
    else:
        with open(path+name, 'wb') as f:
            pickle.dump(dictionary, f)
            
            
class Mask_from_annotations(Preprocess):
    def __init__(self,
                 xml_path = 'xml/',
                 tags = {'lesion benign': ['lesion benign'],
                         'lesion malignant': ['lesion malign'],
                         'all malignant': ['ulceration', 'lesion malign', 'in situ', 'necrosis', 'til', 'mitosis'],
                         'normal tissue': ['normal epidermis','normal adnexal structure','normal adipose tissue'],
                         'ulceration': ['ulceration'],
                         'normal epidermis': ['normal epidermis']}):
        self.tags = tags
        self.xml_path = xml_path
        self.xml_files = [file for file in os.listdir(self.xml_path) if 'MSc_benign' or 'SUShud51' in file]
        super().__init__()
    
    def extract_polygons(self, xml_file, scale=1.0):
        '''Extracts polygons as coordinates from xml file.
        inputs:
            xml_file: an opened xml file, eg. file = open(xml_filename, 'r')
            scale: float that will be multiplied with the coordate in order to scale to a lower or bigger image
            
        returns:
            dictionary with tags as keys and polygon coordinates as values, eg {'Lesion malign': array([[x1,y1],[x2,y2],...])}
            '''
        i=2
        polygon_coordinates = {}
        xml_tree = ET.parse(xml_file)
        root = xml_tree.getroot()
        for child in root[0][0]:
            attributes = child.attrib
            tag = attributes['tags']
            coordinates = []
            for coord in child[0]:
                x = float(coord.attrib['X'])
                y = float(coord.attrib['Y'])
                coordinates.append(np.array([x,y]))
            if tag not in polygon_coordinates.keys():
                polygon_coordinates[tag]= (np.array(coordinates)*scale).astype(int)
            else:
                polygon_coordinates[tag+'_'+str(i)]= (np.array(coordinates)*scale).astype(int)
                i+=1
                 
        return polygon_coordinates
    
    def create_masks_from_polygons(self, polygons, size, plot=True):
        mask_dictionary = {class_: np.zeros(size) for class_ in self.tags}
        
        for tag in polygons:
            tag_edit = tag.lower().split('_')[0]
            for class_, values in self.tags.items():
                if tag_edit in values:
                    mask_dictionary[class_]=cv2.drawContours(mask_dictionary[class_], np.array([polygons[tag]]), -1, 1, -1).astype('uint8')
            
        if plot:
            rows = math.ceil(len(mask_dictionary)/2)
            plt.figure(figsize=(20,20))
            for i, items in enumerate(mask_dictionary.items()):
                class_, mask = items
                plt.subplot(rows,2,i+1)
                plt.imshow(mask,cmap='gray')
                plt.title(class_)
            plt.show()

        return mask_dictionary
    

    def store_masks_from_xml_annotations(self, wsi_pyramid, img_levels=[0,6], show_masks=False):
        '''main function to store masks based on a WSI's xml annotation file
        inputs:
            wsi_pyramid: dictionary containing pyvips objects of WSI on all magnification levls.
            img_levels: the highest magnification level 0 cooresponds to the true size and the lowest 6 corresponds desired mask size
        return:
            will store a dictionary with masks based on tags and annotations
            will also save masks as images'''
            
        for file in self.xml_files:
            if 'MSc_Benign' in file:
                if self.wsi_name[:-5] in file :
                    xml_filename = file
            elif 'SUShud51' in file:
                if self.wsi_name[:-5] in file :
                    xml_filename = file
                
        
        scale_max = self.magnification_levels[min(img_levels)]
        scale_min = self.magnification_levels[max(img_levels)]
        scale = wsi_pyramid[scale_min].width/wsi_pyramid[scale_max].width
        
        
       
        file = open(self.xml_path+xml_filename, 'r')
        polygons = self.extract_polygons(file, scale)
        file.close()
        size = (wsi_pyramid[scale_min].height,wsi_pyramid[scale_min].width)
        dictionary = self.create_masks_from_polygons(polygons, size, plot=show_masks)
        
        write_to_pickle(self.wsi_dataset_file_path+self.wsi_name[:-5]+'/', 'masks.obj', dictionary)
        for mask_name, mask in dictionary.items():
            cv2.imwrite(self.wsi_dataset_file_path+self.wsi_name[:-5]+'/'+mask_name+'.png', mask*255)

            
def main(background_segmentation,
         create_masks_from_annotations,
         extract_patches,
         save_tiles_as_jpeg,
         xml_path,
         wsi_path):
    
    preprocess = Preprocess(wsi_path=wsi_path, save_tiles_as_jpeg=save_tiles_as_jpeg)
    preprocess.make_directories()
    
    if background_segmentation:
        foreground_background_segmentation = Foreground_background_segmentation()
    
    if create_masks_from_annotations:
        mask_from_annotations = Mask_from_annotations(xml_path = xml_path)
    
    
    for wsi_name in os.listdir(preprocess.wsi_path):
        if wsi_name in preprocess.skip_list:
            continue
        
        #if 'SUShud78 -' not in wsi_name:
        #    continue
        
        print(wsi_name)
        
        #load wsi
        preprocess.wsi_name = wsi_name
        preprocess.load_wsi()
        
        if background_segmentation:
            foreground_background_segmentation.create_tissue_mask(preprocess.image_dictionary, display_mask=False)
            cv2.imwrite(preprocess.wsi_dataset_file_path+wsi_name[:-5]+'/tissue_mask.png', foreground_background_segmentation.mask_tissue*255)
            write_to_pickle(preprocess.wsi_dataset_file_path+wsi_name[:-5]+'/', 'masks.obj', {'tissue': foreground_background_segmentation.mask_tissue})
        
        if create_masks_from_annotations:
            mask_from_annotations.wsi_name = wsi_name
            mask_from_annotations.store_masks_from_xml_annotations(preprocess.image_dictionary, show_masks=True)
        
        if extract_patches:
            
            #Initiate dictionary containing top left coordinates of all valid tiles of all requested magninficaions
            coordinate_dict = dict()
            
            #Load mask dict
            preprocess.load_mask_dict()
            if not preprocess.mask_dict:
                print('no masks fround from {}'.format(preprocess.wsi_dataset_file_path+preprocess.wsi_name+'/'))
                continue
            
            if preprocess.mask_hierarchy:
                preprocess.remove_mask_overlap()
                
            # Create folder for each WSI to store output
            os.makedirs(preprocess.output_folder+preprocess.wsi_name[:-5]+'/', exist_ok=True)
            if preprocess.save_tiles_as_jpeg:
                os.makedirs(preprocess.extracted_tiles_folder + preprocess.wsi_name[:-5], exist_ok=True)
    
    
            #start timer
            current_wsi_start_time = time.time()
            
            list_of_valid_tiles_from_current_wsi = []
            i = 0 
            
            #decrease tau if the image is bigger than threshold
            preprocess.dynamic_TAU_size(threshold = 3*10**9)
            
            #calculate area ratios and other mearuerments between the different levels of the wsi_pyramid
            preprocess.image_pyramid_measurements()
            
            #calculate the patch criteria to find valid patches
            preprocess.patch_criteria()
            
            # Loop through each tissue class to fit tiles on
            class_to_remove = []
            region_masks = dict()
            tissue_classes_to_fit_tiles_on = preprocess.tissue_classes_to_fit_tiles_on.copy()
            for current_class_to_copy in tissue_classes_to_fit_tiles_on:
                print('Now processing {} regions'.format(current_class_to_copy))
                # Extract mask for current class
                current_class_mask = preprocess.mask_dict[current_class_to_copy].copy()
                
                if len(np.unique(current_class_mask))==1:
                    print('Mask for class {} is empty'.format(current_class_to_copy))
                    #tissue_classes_to_fit_tiles_on.remove(current_class_to_copy)
                    class_to_remove.append(current_class_to_copy)
                    continue
                
                # Resize colormap to the size of tau overview image
                current_class_mask = cv2.resize(current_class_mask, dsize=(preprocess.tau_image_width, preprocess.tau_image_height), interpolation=cv2.INTER_CUBIC)
                print('Loaded segmentation mask with size {} x {}'.format(current_class_mask.shape[1], current_class_mask.shape[0]))
                
                #Add mask of current class to region_masks dictionary
                region_masks[current_class_to_copy]=current_class_mask
                
                # Save the annotation mask image (If option is set to True)
                if preprocess.save_binary_annotation_mask:
                    #annotation_mask_for_saving = current_class_mask * 255
                    cv2.imwrite(preprocess.output_folder + preprocess.wsi_name[:-5] + '/mask_{}.jpg'.format(current_class_to_copy), current_class_mask * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    
                labels = find_regions(current_class_mask, preprocess.small_region_remove_threshold, preprocess.remove_small_regions)
                # Get region properties
                list_of_regions = regionprops(labels)
                
                if preprocess.remove_small_regions:
                    print('\tFound {} regions (after removal of small regions)'.format(len(list_of_regions)))
                    # Create a new binary map after removing small objects
                    region_masks[current_class_to_copy] = np.zeros(shape=current_class_mask.shape)
                    
                #Extract all coordinates (to draw region on overview image)
                list_of_valid_tiles_from_current_class = []
                for current_region in list_of_regions:
                    #list_of_valid_tiles_from_current_class = []
                    #Draw region on mask
                    if preprocess.remove_small_regions:
                        region_masks[current_class_to_copy][current_region.coords[:,0],current_region.coords[:,1]]=1
                    #Create a box around region
                    box_coorinates = box_around_region(current_region)
                    #Find potential tile coordinates
                    all_coordinates = preprocess.possible_x_and_y_coordinates(box_coorinates)
                    
                    # image = np.zeros((region_masks[current_class_to_copy].shape[0],region_masks[current_class_to_copy].shape[1],3))
                    # image[:,:,0][region_masks[current_class_to_copy]==1.0]=255
                    # image[:,:,1][region_masks[current_class_to_copy]==1.0]=255
                    # image[:,:,2][region_masks[current_class_to_copy]==1.0]=255
                    # for x in all_x_pos:
                    #     for y in all_y_pos:
                    #         cv2.rectangle(image, (x, y), (x+preprocess.tile_size, y+preprocess.tile_size), (255, 0, 0), 1)
                            
                    # plt.imshow(image)
                    # plt.show()
                    
                    #Find accepted tile coordinates based on tile criteria
                    list_of_valid_tiles_from_current_class += preprocess.find_valid_tiles(all_coordinates, region_masks[current_class_to_copy])
                    
                    
                plt.imshow(region_masks[current_class_to_copy], cmap='gray')
                plt.title('{} mask after removal of small areas'.format(current_class_to_copy))
                plt.show()
                
                tiles_before_removal = len(list_of_valid_tiles_from_current_class)
                #remove tiles that only contain background. This is not needed for the tissue mask, as it has already had background segmentation
                if current_class_to_copy == 'tissue':
                    remove_tiles = False
                else:
                    remove_tiles = True
                    
                coordinate_dict, list_of_valid_tiles_from_current_class = preprocess.create_coordinate_dictionary(coordinate_dict,
                                                                                                     list_of_valid_tiles_from_current_class,
                                                                                                     current_class_to_copy,
                                                                                                     remove_tiles=remove_tiles, 
                                                                                                     threshold = 205)
                tiles_after_removal = len(list_of_valid_tiles_from_current_class)
                print(tiles_before_removal-tiles_after_removal, ' tiles removed due to low variance')
                
                # Add the tiles to the list of tiles of current wsi
                list_of_valid_tiles_from_current_wsi.extend(list_of_valid_tiles_from_current_class)
                
            for i in class_to_remove:
                tissue_classes_to_fit_tiles_on.remove(i)
                
            #Move on to next WSI if there are no masks
            if len(tissue_classes_to_fit_tiles_on)==0:
                print('No masks found for WSI {}'.format(wsi_name))
                continue
            
            # Save predicted coordinates dict as pickle
            preprocess.save_coordinates(coordinate_dict)
            
            # Save overview image
            filename = preprocess.output_folder + preprocess.wsi_name[:-5] + '/image_clean.jpeg'
            preprocess.image_dictionary[preprocess.tau].jpegsave(filename, Q=100)
            
            # Read overview image again using cv2, and add alpha channel to overview image.
            overview_jpeg_file = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            
            #overview_jpeg_file = mask_overlay(overview_jpeg_file, region_masks['benign'])
            overview_jpeg_file = np.dstack([overview_jpeg_file, np.ones((overview_jpeg_file.shape[0], overview_jpeg_file.shape[1]), dtype="uint8") * 255])
            
            #Convert masks from 0-1 -> 0-255 (can also be used to set the color)
            for n in tissue_classes_to_fit_tiles_on:
                region_masks[n] *= 255
                
            # Create a empty alpha channel
            shape_for_alpha_channel = region_masks[tissue_classes_to_fit_tiles_on[0]].shape
            alpha_channel = np.zeros(shape=shape_for_alpha_channel).astype(region_masks[tissue_classes_to_fit_tiles_on[0]].dtype)
            
            #n_tissue_classes = len(tissue_classes_to_fit_tiles_on)
            # Each mask is 1-channel, merge them to create a 3-channel image (RGB), the order is used to set the color for each mask. Add a alpha-channel.
            
            #Draw transparent masks over the overview image
            region_masks = mask_overlay(region_masks, alpha_channel)
            # Draw the selected regions on the overview image
            for _, current_tissue_mask in region_masks.items():
                overview_jpeg_file = cv2.addWeighted(current_tissue_mask, 1, overview_jpeg_file, 1.0, 0, dtype=cv2.CV_64F)
                
            # Draw tiles on the overview image
            preprocess.draw_tiles_on_image(overview_jpeg_file, list_of_valid_tiles_from_current_wsi)
            
            #save overview image
            preprocess.save_overview_image(overview_jpeg_file, len(list_of_valid_tiles_from_current_wsi))
                    
            # Calculate elapse time for current run
            elapse_time = time.time() - current_wsi_start_time
            m, s = divmod(elapse_time, 60)
            h, m = divmod(m, 60)
            model_time = '%02d:%02d:%02d' % (h, m, s)

            # Print out results
            print('Found {} tiles in image'.format(len(list_of_valid_tiles_from_current_wsi)))
            print('Finished. Duration: {}'.format(model_time))
            
            
            
            

    
if __name__ == '__main__':
    main(background_segmentation=False,
         create_masks_from_annotations = False,
         extract_patches = True,
         save_tiles_as_jpeg = False,
         #xml_path ='xml/',
         xml_path = '/home/prosjekt/Histology/.xmlStorage/',
         #wsi_path = 'WSI_all/')
         wsi_path = '/home/prosjekt/Histology/Melanoma_SUS/MSc_Benign_Malign_HE/')
