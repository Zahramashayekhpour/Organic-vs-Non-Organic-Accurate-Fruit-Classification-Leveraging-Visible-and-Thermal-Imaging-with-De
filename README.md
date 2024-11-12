README: Organic Fruits and Vegetables Classification Project
Overview
This project aims to classify organic apples and mushrooms from non-organic counterparts using both traditional feature extraction methods and advanced deep learning techniques. The methodologies presented in this project are based on the findings of our research article and have been implemented to facilitate reproducibility and further exploration.

Dataset
We have generated a custom dataset that includes both thermal and visible images of organic and non-organic apples and mushrooms. This dataset is made freely available to encourage research and development in this field. You can download it using the following link:
Download Dataset (insert actual link here)

Project Structure
The project is organized into two main methodologies: Traditional Method and Deep Learning Method. Each methodology comprises specific code files explained below.

Traditional Method
Color Feature Extraction:

File: color_feature_extraction.py
Description: This script extracts color features from the provided images using techniques such as color moments. The extracted features serve as foundational inputs for subsequent analyses.
Classical Feature Extraction:

File: classical_feature_extraction.py
Description: This script builds upon the color features by extracting additional features, including shape, texture (GLCM), and statistical features. This comprehensive feature set enhances the classification performance.
Feature Evaluation:

File: evaluate_features.py
Description: In this script, we evaluate the performance of the extracted features using multiple classifiers, including Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), Random Forest (RF), and Multi-Layer Perceptron (MLP). The performance metrics will demonstrate the effectiveness of the traditional classification approach.
Deep Learning Method
Pre-trained CNN:

File: pretrained_cnn.py
Description: This script implements and compares 13 pre-trained Convolutional Neural Networks (CNNs) to determine the best-performing model for classification tasks. A customized classification head is appended to prevent overfitting and improve accuracy.
CNNs as Feature Extractor:

File: cnn_feature_extractor.py
Description: This script utilizes selected pre-trained CNNs as feature extractors. It combines features from the last layers of MobileNetV2, DenseNet169, and ResNet152 to enhance model performance and achieve high classification accuracy.
Results
The methodologies developed in this project yield the following results:

Traditional Method:
The traditional approach achieved a maximum accuracy of 94.89% for classifying apples in thermal images using LDA.
Deep Learning Method:
The best accuracy obtained for apples in visible images was 100% using LDA.
The RF classifier achieved 92.35% accuracy for apples in thermal images.
For mushrooms, the SVM reached 97.30% accuracy in visible images and 100% accuracy in thermal images.
These findings validate the methodologies and demonstrate the potential of using both traditional and deep learning approaches for non-destructive classification of organic produce.

Installation
To execute the project, ensure that you have the following prerequisites installed:

Python version 3.x
You can install all necessary libraries by running:

bash
pip install -r requirements.txt  
Usage
To run the project, follow these steps:

Download the Dataset: Ensure you have the dataset accessible to your scripts.

Run the Traditional Method:

Execute color_feature_extraction.py to extract color features.
Run classical_feature_extraction.py to extract additional features.
Finally, execute evaluate_features.py to assess and compare the classifier performances.
Run the Deep Learning Method:

First, execute pretrained_cnn.py to evaluate the various pre-trained CNN architectures.
Next, run cnn_feature_extractor.py to extract features using the selected CNNs.
Contributing
We welcome contributions to improve and enhance the project. Feel free to fork the repository and submit a pull request with your suggestions or enhancements.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
We would like to acknowledge [insert any relevant acknowledgments here, such as institutions, individuals involved in research, or datasets used].
