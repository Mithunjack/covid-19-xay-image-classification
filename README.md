# COVID-19-Xray-Image-Classification

## Introduction
This project aims to experiment with the automated detection of COVID-19 using X-ray images, employing Convolutional Neural Networks (CNN) for image classification. We focus on diagnosing COVID-19, typically associated with pneumonia symptoms, through X-ray imaging.

## Problem Statement
Since the outbreak of COVID-19 in December 2019 in Wuhan, China, the disease has spread globally. Deep learning, particularly CNN, has shown great promise in classifying medical images. This project explores the potential of CNN in identifying COVID-19 infections in chest X-ray images.

## Presentation Slide [here](https://github.com/Mithunjack/covid-19-xay-image-classification/blob/master/Presentaion_Slides.pdf)

## Goals
- To apply CNN in classifying chest X-ray images for COVID-19 detection.
- To evaluate the performance of our CNN model against pre-trained models like VGG16, ResNet, and DenseNet.
- To contribute to AI research in understanding COVID-19 infections.
- Note: This project is experimental and not intended for clinical use.

## Datasets
1. **COVID-Chestxray-Dataset**: A quality annotated dataset referenced in multiple papers. [GitHub Link](https://github.com/ieee8023/covid-chestxray-dataset)
2. **Kaggle CoronaHack - Chest X-Ray-Dataset**: Includes various categories including COVID-19, SARS, Streptococcus, and ARDS. [Kaggle Link](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset)
3. **COVIDx CT-2 Dataset**: A collection of CT-SCAN data of 2837 patients. [Kaggle Link](https://www.kaggle.com/hgunraj/covidxct)
4. **COVID-XRay-5K Dataset**: Approximately 5000 images, curated with the assistance of a board-certified radiologist. [GitHub Link](https://github.com/shervinmin/DeepCovid)

### Data Preprocessing
- Augmentation techniques used to expand the limited dataset of 84 covid-19 positive X-ray images.
- Post-augmentation, we obtained 242 new augmented images.

#### Augmentation Code
```python
ImageDataGenerator(rotation_range=5,    
                   rescale=1./255, 
                   shear_range=0.2, 
                   zoom_range=0.3, 
                   horizontal_flip=True, 
                   fill_mode='nearest', 
                   data_format='channels_last', 
                   brightness_range=[0.2,1.0])


Model: "sequential_43"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_90 (Conv2D)           (None, 85, 85, 64)        1792      
_________________________________________________________________
max_pooling2d_80 (MaxPooling (None, 42, 42, 64)        0         
_________________________________________________________________
dropout_52 (Dropout)         (None, 42, 42, 64)        0         
_________________________________________________________________
conv2d_91 (Conv2D)           (None, 14, 14, 32)        18464     
_________________________________________________________________
max_pooling2d_81 (MaxPooling (None, 7, 7, 32)          0         
_________________________________________________________________
dropout_53 (Dropout)         (None, 7, 7, 32)          0         
_________________________________________________________________
flatten_33 (Flatten)         (None, 1568)              0         
_________________________________________________________________
dense_68 (Dense)             (None, 256)               401664    
_________________________________________________________________
dense_69 (Dense)             (None, 1)                 257       
=================================================================
Total params: 422,177
Trainable params: 422,177
Non-trainable params: 0
_________________________________________________________________

```

Pre-Trained Model Performance
We also evaluated the performance using pre-trained models:

MobileNet
Loss: 0.74
Accuracy: 0.69


ResNet
Loss: 0.11
Accuracy: 0.98


DenseNet
Loss: 0.07
Accuracy: 0.96

