# README: Classification of Organic and Non-Organic Apples and Mushrooms Based on Visible and Thermal Imaging Using Traditional and Deep Methods   

## Overview  
This project aims to classify organic apples and mushrooms from non-organic counterparts using both traditional feature extraction methods and advanced deep learning techniques. The methodologies presented in this project are based on the findings of our research article and have been implemented to facilitate reproducibility and further exploration.  

## Dataset  
We have generated a custom dataset in a controlled environment that includes both thermal and visible images of organic and non-organic apples and mushrooms. This dataset is made freely available to encourage research and development in this field. 
You can download it using the following link:
[Organic and Non-Organic Dataset](https://data.mendeley.com/datasets/pwzk7dj5wf/1)  
The imaging environment which is designed to control environmental conditions is shown in the figure.
![image](https://github.com/user-attachments/assets/92a490ab-a3c8-4882-a291-fdee17793603)
<p align="center">
    <em> (a) NIR camera. (b) Experimental imaging setting
</p>

## Project Structure  
The project is organized into two main methodologies: Traditional Method and Deep Learning Method. Each methodology comprises specific code files explained below. 
The general view of our work in this paper is represented in the next figure.
![Screenshot (253)](https://github.com/user-attachments/assets/9520e98d-21ba-4740-b880-182a95f5bed1)
<p align="center">
    <em> The general structure of the proposed method</em>
</p>

### Pre-processing
1. **Color Image Segmentation**:  
   - **File**: `color_feature_extraction.py`  
   - **Description**: Thresholding on color images involves setting boundaries for each channel to create masks for segmenting fruits from the background. Converting images from RGB to different color spaces, particularly HSV, yields the best results for apple and mushroom segmentation.

1. **Thermal Image Segmentation**:  
   - **File**: `color_feature_extraction.py`  
   - **Description**: Active contour segmentation iteratively refines the boundary of a target object based on initial points placed around it. A total of 250 iterations were performed, along with morphological opening and closing operations to enhance segmentation accuracy.

### Traditional Method  
Upon employing the image segmentation approach, different methods of extracting features exhibit varying results. Methods like GLCM, Gabor filters, HOG descriptors, and CM for visible images provide distinct sets of features. When these features are used with classifiers like MLP, LDA, RF, and SVM, their ability to distinguish between organic and non-organic fruits differs. Additionally, GA is established to enhance the results.

1. **Color Feature Extraction**:  
   - **File**: `color_feature_extraction.py`  
   - **Description**: Three color spaces—RGB, HSV, and Lab—are utilized, with images converted to HSV and Lab. From these color spaces, 27 statistical features, including mean, standard deviation, and skewness, are extracted from each channel. 

2. **Classical Feature Extraction**:  
   - **File**: `classical_feature_extraction.py`  
   - **Description**: This script builds upon the color features by extracting additional features, including shape, texture (GLCM), and statistical features. This comprehensive feature set enhances the classification performance.  

3. **Feature Evaluation**:  
   - **File**: `evaluate_features.py`  
   - **Description**: In this script, we evaluate the performance of the extracted features using multiple classifiers, including Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), Random Forest (RF), and Multi-Layer Perceptron (MLP). The performance metrics will demonstrate the effectiveness of the traditional classification approach.  

### Deep Learning Method  

1. **Pre-trained CNN**:  
   - **File**: `pretrained_cnn.py`  
   - **Description**: This script implements and compares 13 pre-trained Convolutional Neural Networks (CNNs) to determine the best-performing model for classification tasks. A customized classification head is appended to prevent overfitting and improve accuracy.  

2. **CNNs as Feature Extractor**:  
   - **File**: `cnn_feature_extractor.py`  
   - **Description**: This script utilizes selected pre-trained CNNs as feature extractors. It combines features from the last layers of MobileNetV2, DenseNet169, and ResNet152 to enhance model performance and achieve high classification accuracy.  

---  

## Results  
The methodologies developed in this project yield the following results:  

### Traditional Method:  
- The traditional approach achieved a maximum accuracy of 94.89% for classifying apples in thermal images using LDA.  

### Deep Learning Method:  
- The best accuracy obtained for apples in visible images was 100% using LDA.  
- The RF classifier achieved 92.35% accuracy for apples in thermal images.  
- For mushrooms, the SVM reached 97.30% accuracy in visible images and 100% accuracy in thermal images.  

These findings validate the methodologies and demonstrate the potential of using both traditional and deep learning approaches for non-destructive classification of organic produce.  

---  

## Installation  
To execute the project, ensure that you have the following prerequisites installed:  
- Python version 3.x  

You can install all necessary libraries by running:  

```bash  
pip install -r requirements.txt
