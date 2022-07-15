# Melanoma-Diagnosis-and-Localization-from-Whole-Slide-Images-using-Convolutional-Neural-Networks

This repository requires access to the following:

1) WSI files: /home/prosjekt/Histology/Melanoma_SUS/MSc_Benign_Malign_HE/
2) More WSI files: /home/prosjekt/Histology/Melanoma_SUS/MSc_Good_Bad_prognosis/
3) XML files (annotations): /home/prosjekt/Histology/.xmlStorage/

If running inference with a trained model, the user can download the weights of the best model (Model 17 in the report) from here:

https://www.dropbox.com/s/8tyxoybmd41mnf2/Model_17.pt?dl=0

The model weights must be placed in the folder Models/Weights/

For inference, run inference.py. inference.py takes a folder of WSIs as input and output predicted images and a slide based diagnosis. 1 = Melanoma. 0 = Benign nevus.


When running inference.py, functions are called in the main-function:

1) inference.find_weights_dict() finds the weights of the pre-trained model.
2) inference.store_probabilities_from_model() runs inference on the WSIs that are located in wsi_path. For each WSI it stores the probabilites of each patch beloning to the either lesion bening, lesion malignant and in the multi class case normal tissue. The probabilities are saved as objects in the path "probability_save_path"
3) inference.store_prediction_images() stores one prediction image for each WSI in the directory "probability_save_path".
4) inference.get_best_mal_ben_threshold() collects the stored best ratio-threshold between malignant and benign pixels.
5) inference.classify_wsi_from_pred_images() counts the number of malingnant and benign pixels and classify the WSI based on the ratio between them.
6) inference.save_tru_mask() if annotations exist, this function will save a image with annotations that can be used as comparision with the predicted image.

The following paths of inference.py:

1) wsi_path: This is the folder where the whole slide image scan files are located
2) wsi_mask_path: when running preprocessing, the masks created for each WSI for each class are stored here. Both as an object containing the top left coordinates of each patch and as image files.
3) xml_path: If available, this is where the annotations from the pathologist are stored as xml files
4) data_set_path: The folder where the top left coordinates of the extracted patches of each WSI are collected. During preprocessing, the the coordinates are stored by default in coordinates/
5) model_path: The folder where the weights of the different models are stored
6) probability_save_path: The folder where the output from the softmax-function is stored during inference. This is an object containing all WSIs used in inference and the probabilites of corresponing patches beloning to all possible classes.
7) inference_metrics_path: the folder with textfiles that lists the best models, and the best thresholds. Also the folder where the resulting predictions and predicion images are stored

