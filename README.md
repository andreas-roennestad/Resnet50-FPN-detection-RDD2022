## Project (TDT17): Predicting road damage on the RDD2022 dataset using a fine-tuned Resnet-50 FPN backbone with pre-trained weights in PyTorch

### Overview
This project aims to fine-tune a PyTorch object detector for the purpose of detecting road damage in the RDD2022 dataset, with the goal of participating in the [Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)](https://crddc2022.sekilab.global/). The RDD2022 dataset contains images of roads captured from vehicles, with labels for various types of road damage including longitudinal cracks (D00), transverse cracks (D10), alligator cracks (D20), and potholes (D40). The model used for this project is a fine-tuned [Resnet-50 FPN backbone with pre-trained weights](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html). This project was completed as a basis for evaluation in the course TDT17 - Visual Intelligence at the Norwegian University of Science and Technology by Andreas Rønnestad.


### Process
For the project, students were open to select any object detector for the purpose of detection on RDD2022. The dataset contains labelled images of roads captured from vehicles, with labels generally categorized into one of four categories, namely:
* D00 - longitudinal cracks
* D10 - transverse cracks
* D20 - alligator cracks
* D40 - potholes


Upon inspection of representative images from the dataset, I observed that road damage is often difficult to detect, as it can be obscure and with large variation in size and lighting. Furthermore it can be hard to distinguish road cracks from for example shadows, and to predict its exact location.
This motivated my choice to extract features with the pretrained weights of the architecture referenced, but to attempt to fine-tune also layers of the backbone feature extraction network so as to adapt features extracted to the specific scenario. As the damage to detect in this task is much less distinguished than objects in the object-classes pre-trained on, this adaptation is required.


Fine-tuning was performed on the Norway training subset of the data with 1, 2, 3 trainable backbone layers, and results on the test dataset of the challenge were 0.18, 0.21, and 0.31, respectively. Some exemplary samples are presented in the next subsection.

The code is documented and contains implementation of dataloaders, training and inference methods, etc. 

### Samples
The following samples were predicted by the model after training for 40 epochs on the Norway train-dataset. The labels and their corresponding bounding box colors are: D00(longitudinal cracks) - green, D10(transverse cracks) - red, D20(alligator cracks) - orange, D40(potholes) - pink.

In the following images the model demonstrates learning the features of the different classes:

<img src="https://github.com/andreas-roennestad/miniProject-TDT17/blob/master/predicted_image_samples/Norway_008442.jpg" width="1280" height="720" />
<img src="https://github.com/andreas-roennestad/miniProject-TDT17/blob/master/predicted_image_samples/Norway_008340.jpg" width="1280" height="720" />

<img src="https://github.com/andreas-roennestad/miniProject-TDT17/blob/master/predicted_image_samples/Norway_008512.jpg" width="1280" height="720" />

Alligator cracks are detected even when not highly represented in dataset:
<img src="https://github.com/andreas-roennestad/miniProject-TDT17/blob/master/predicted_image_samples/Norway_008528.jpg" width="1280" height="720" />

Shadows are sometimes hard to distinguish from road-damage in visual images:
<img src="https://github.com/andreas-roennestad/miniProject-TDT17/blob/master/predicted_image_samples/Norway_008181.jpg" width="1280" height="720" />

