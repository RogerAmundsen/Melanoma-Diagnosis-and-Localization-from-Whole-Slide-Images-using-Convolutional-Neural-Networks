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
