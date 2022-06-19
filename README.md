# Melanoma-Diagnosis-and-Localization-from-Whole-Slide-Images-using-Convolutional-Neural-Networks

This repository requires access to the following:

1) WSI files: /home/prosjekt/Histology/Melanoma_SUS/MSc_Benign_Malign_HE/
2) More WSI files: /home/prosjekt/Histology/Melanoma_SUS/MSc_Good_Bad_prognosis/
3) XML files (annotations): /home/prosjekt/Histology/.xmlStorage/

If running inference with a trained model, the user can download the weights of the best model (Model 17 in the report) from here:

https://www.dropbox.com/s/8tyxoybmd41mnf2/Model_17.pt?dl=0

The model weights must be placed in the folder Models/Weights/

For inference, run inference.py. inference.py takes a folder of WSIs as input and output predicted images and a slide based diagnosis. 1 = Melanoma. 0 = Benign nevus.
