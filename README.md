# README: Organic vs Non-Organic: Accurate Fruit Classification Leveraging Visible and Thermal Imaging with Deep and Traditional Methods

## Overview  
This project aims to classify organic apples and mushrooms from non-organic counterparts using both traditional feature extraction methods and deep learning techniques. The methodologies presented in this project are based on the findings of our research article and have been implemented to facilitate reproducibility and further exploration.  

## Dataset  
We have generated a custom dataset in a controlled environment that includes both thermal and visible images of organic and non-organic apples and mushrooms. This dataset is made freely available to encourage research and development in this field.  
You can download it using the following link:  
[Organic and Non-Organic Dataset](https://data.mendeley.com/datasets/pwzk7dj5wf/1)  
The imaging system which is designed to control environmental conditions is shown in the figure 1. 
![Screenshot (255)](https://github.com/user-attachments/assets/c125ec4f-a90e-4b4a-8a64-f0613fd14a9c)  

<p align="center">  
    <em> Figure1: (a) NIR camera. (b) Experimental imaging setting</em>  
</p>  

## Project Structure  
The project is organized into two main methodologies: the Traditional Method and Deep Learning Method. Each methodology comprises specific code files explained below.  
The general view of our work in this paper is represented in the figure 2.  
![Screenshot (253)](https://github.com/user-attachments/assets/9520e98d-21ba-4740-b880-182a95f5bed1)  
<p align="center">  
    <em> Figure2: The general structure of the proposed method</em>  
</p>  

### Pre-processing  
1. **Color Image Segmentation**:  
   - **File**: [Color Image Segmentation](https://github.com/Zahramashayekhpour/organic-and-nonorganic-fruits-classification-based-on-traditional-machine-learning-and-deep-learning/blob/master/Color%20Image%20Segmentation%20(HSV%20color%20space).m)  
   - **Description**: Thresholding on color images involves setting boundaries for each channel to create masks for segmenting fruits from the background. Converting images from RGB to different color spaces, particularly HSV, yields the best results for apple and mushroom segmentation.  

2. **Thermal Image Segmentation**:  
   - **File**: [Thermal Image Segmentation](https://github.com/Zahramashayekhpour/organic-and-nonorganic-fruits-classification-based-on-traditional-machine-learning-and-deep-learning/blob/master/Thermal%20Image%20Segmentation%20(Active%20Contour).m) 
   - **Description**: Active contour segmentation iteratively refines the boundary of a target object based on initial points placed around it. A total of 250 iterations were performed, along with morphological opening and closing operations to enhance segmentation accuracy.  

### Traditional Method  
Upon employing the image segmentation approach, different methods of extracting features exhibit various results. Methods like GLCM, Gabor filters, HOG descriptors, and CM for visible images provide distinct sets of features. When these features are used with classifiers like MLP, LDA, RF, and SVM, their ability to distinguish between organic and non-organic fruits differs. Additionally, GA is established to enhance the results.  

1. **Color Feature Extraction**:  
   - **File**: [Color Feature Extraction](https://github.com/Zahramashayekhpour/organic-and-nonorganic-fruits-classification-based-on-traditional-machine-learning-and-deep-learning/blob/master/Color_Image_Feature.ipynb)  
   - **Description**: Three color spaces—RGB, HSV, and Lab—are utilized, with images converted to HSV and Lab. From these color spaces, 27 statistical features, including mean, standard deviation, and skewness, are extracted from each channel.   

2. **Classical Feature Extraction**:  
   - **File**: [Classical Feature Extraction](https://github.com/Zahramashayekhpour/organic-and-nonorganic-fruits-classification-based-on-traditional-machine-learning-and-deep-learning/blob/master/Classical_Feature_Extraction.ipynb)
   - **Description**: This code extracts various categories of texture features, including GLCM in four directions (0, 45, 90, and 135 degrees), energy and entropy from the output of Gabor filters, statistical features, and shape features represented by HOG descriptor.  

3. **Feature Evaluation**:  
   - **File**: [Feature Evaluation](https://github.com/Zahramashayekhpour/organic-and-nonorganic-fruits-classification-based-on-traditional-machine-learning-and-deep-learning/blob/master/Evaluation_the_Features.ipynb)
   - **Description**: In this script, we evaluate the performance of the extracted features using multiple classifiers, including Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), Random Forest (RF), and Multi-Layer Perceptron (MLP). The performance metrics will demonstrate the effectiveness of the traditional classification approach.  

4. **Feature Selection**:  
   - **File**: [Feature Selection](https://github.com/Zahramashayekhpour/organic-and-nonorganic-fruits-classification-based-on-traditional-machine-learning-and-deep-learning/blob/master/Genetic_Algorithm.ipynb)  
   - **Description**: Accurate classification relies on identifying informative features, and feature selection aims to find the optimal subset for machine learning models. GA, an adaptive search algorithm, iteratively applies main operators such as populations, fitness functions, selection, crossover, and mutation to identify the optimized solution after five iterations.  

### Deep Learning Method  

1. **Pre-trained CNN**:
   
   - **File**: [Pre-trained CNN](https://github.com/Zahramashayekhpour/organic-and-nonorganic-fruits-classification-based-on-traditional-machine-learning-and-deep-learning/blob/master/Pre-trained%20CNNs%20.ipynb) 
   - **Description**: This script implements and compares 13 pre-trained Convolutional Neural Networks (CNNs), including MobileNet, MobileNetV2, ResNet101VV2, ResNet101, ResNet50, ResNet50V2, ResNet152, DenseNet201, DenseNet169, DenseNet121, Xception, InceptionV3, and InceptionResNetV2. These models are evaluated to determine which performs best for classification tasks. A customized classification head is appended to prevent overfitting and improve accuracy. The CNN model, illustrated in Figure 3, consists of two main components: the first section has all layers frozen with no optimization, while the second part involves fine-tuning the last layer with a heading model. The layer arrangement may vary based on the specific problem and dataset but generally follows a logical flow: feature extraction layers, pooling layers to reduce dimensions and capture essential features, batch normalization for input normalization and speed, a dense layer with 256 neurons for learning complex relationships, a dropout layer (0.25) to prevent overfitting, and a final dense output layer with two neurons for predictions. Images are resized to 224 × 224 and 227 × 227 pixels to match the network's input dimensions.   
![Screenshot (254)](https://github.com/user-attachments/assets/6e829dfe-54bd-41b7-82bb-8c1996754560)  
<p align="center">  
    <em> Figure3: The proposed DL system model</em>  
</p>  

2. **CNNs as Feature Extractor**:
   
   - **File**: [CNNs as Feature Extractor](https://github.com/Zahramashayekhpour/organic-and-nonorganic-fruits-classification-based-on-traditional-machine-learning-and-deep-learning/blob/master/CNNs_as_Feature_Extractors.ipynb) 
   - **Description**: Using pre-trained CNNs for feature extraction is computationally intensive, while features are typically extracted from fully connected layers before the final classification layer without additional training. These models require less computing capacity. After evaluating various pre-trained networks, Densenet169, ResNet152, and MobileNetV2 were chosen. The feature vectors from each network were combined to create a final feature vector, which was then used as input for MLP, LDA, SVM, and RF classification algorithms.  

---  

## Results  
The methodologies developed in this project yield the following results:  

 
- The traditional approach achieved a maximum accuracy of 94.89% for classifying apples in thermal images using LDA.  
- The best accuracy obtained for apples in visible images was 100% using LDA.  
- The RF classifier achieved 92.35% accuracy for apples in thermal images.  
- For mushrooms, the SVM reached 97.30% accuracy in visible images and 100% accuracy in thermal images.  

These findings validate the methodologies and demonstrate the potential of using both traditional and deep learning approaches for non-destructive classification of organic produce.  

---  
  
## Installation  

You can run the project in **Google Colab** by installing the required libraries with `!pip install -U tensorflow keras opencv-python scikit-learn matplotlib seaborn` and uploading your dataset.
For **MATLAB**, ensure you have a compatible version with the required toolboxes (Image Processing, Statistics and Machine Learning, and Deep Learning) and run the provided `.m` files.

If you have any questions, contact me by this email : Zahra_mashayekhpour@yahoo.com
