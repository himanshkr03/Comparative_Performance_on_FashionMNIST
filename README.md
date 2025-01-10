# FashionClasstic-Benchmarking FashionMNIST Classifiers

## Overview

This project aims to classify images from the Fashion MNIST dataset using various machine learning and deep learning models. The Fashion MNIST dataset is a collection of 28x28 grayscale images of clothing items, with 10 different classes.

## Data

### Dataset Description

The project utilizes the Fashion MNIST dataset, which is a widely used benchmark dataset for image classification. It consists of 70,000 grayscale images, divided into 60,000 training images and 10,000 testing images. Each image is 28x28 pixels and represents a clothing item or accessory.

**Image Classes:**

The dataset includes 10 distinct classes of clothing items, as follows:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

**Data Format:**

The images are stored as NumPy arrays with shape (28, 28), representing the pixel intensities. The labels are integers ranging from 0 to 9, corresponding to the 10 classes mentioned above. The dataset is readily available through TensorFlow Datasets (`tensorflow_datasets`) and can be easily loaded and preprocessed for model training and evaluation.


### Data Preprocessing

The images are preprocessed by normalizing pixel values to the range [0, 1]. This normalization helps improve the performance and stability of the machine learning models during training.

## Methodology

### Data Exploration and Visualization

The notebook starts with data exploration and visualization techniques to understand the dataset:
1. Showing images with labels to visually inspect the data.
2. Analyzing the distribution of classes in the training data using a histogram.
3. Displaying a grid of images from different classes to showcase the variety within each category.

### Model Training and Evaluation

The project experiments with several machine learning and deep learning models for classification, including:
**Stage 1: Simple Models**
1. **Logistic Regression:** A simple linear model.
2. **Support Vector Machine (SVM):** A powerful model for classification.
3. **Decision Tree:** A tree-based model that makes decisions based on features.

**Stage 2: Complex Models**
1. **Random Forest:** An ensemble of decision trees for improved performance.
2. **XGBoost:** A gradient boosting algorithm known for its accuracy.
3. **Deep Neural Network (DNN):** A multi-layered neural network implemented using TensorFlow/Keras.
4. **CNN+RNN:** A hybrid model combining Convolutional Neural Network (CNN) for feature extraction and Recurrent Neural Network (RNN) for sequential modeling.

**Pretrained + Fine-Tuned Models**
1. **DeiT (Data-efficient Image Transformers):** A pre-trained transformer model from Hugging Face Transformers is used for image classification.
2. **MobileNetV2:** A pre-trained model fine-tuned for Fashion MNIST classification.

Each model is trained on the training data and evaluated on the test data using appropriate metrics such as accuracy, confusion matrix, and classification report.

## Results and Analysis

### Model Performance Comparison

The following table summarizes the accuracy scores achieved by different models on the Fashion MNIST test dataset:

| Model | Accuracy |
|---|---|
| Logistic Regression | ~84% |
| Support Vector Machine (SVM) | ~88% |
| Decision Tree | ~80% |
| Random Forest | ~88% |
| XGBoost | ~90% |
| Deep Neural Network (DNN) | ~90% |
| CNN+RNN | ~92% |
| DeiT (Pretrained) | ~85% |
| MobileNetV2 (Fine-tuned) | ~92% |


**Analysis:**

* **Simple Models:** Logistic Regression, SVM, and Decision Tree provide a baseline performance, with SVM achieving the highest accuracy among them.
* **Complex Models:** Random Forest, XGBoost, DNN, and CNN+RNN demonstrate significant improvements in accuracy compared to the simple models. The CNN+RNN model achieves the best accuracy among these models.
* **Pretrained Models:** The DeiT pre-trained model achieves decent accuracy, while the fine-tuned MobileNetV2 model provides comparable performance to the CNN+RNN model.


### Confusion Matrix and Classification Report

Confusion matrices and classification reports are generated for each model to provide a detailed analysis of their performance on different classes. These visualizations help identify classes that are easily confused by the models and assess the overall precision, recall, and F1-score for each class.

### Feature Importance

For tree-based models like Decision Tree, Random Forest, and XGBoost, feature importance analysis is performed to understand which features contribute most to the model's predictions. This analysis can provide insights into the patterns learned by the models and the relevant characteristics of the images for classification.

### Learning Curves

Learning curves are plotted to visualize the model's training and validation performance over epochs. These curves help identify potential issues like overfitting or underfitting. If the training accuracy is significantly higher than the validation accuracy, it might indicate overfitting, suggesting the need for regularization or more data.

## Usage

To run this project, follow these steps:
1. Open the Jupyter Notebook in Google Colab.
2. Install the required libraries using the provided code cell.
3. Execute the code cells sequentially to load the data, train the models, and evaluate their performance.
4. Review the visualizations and results to understand the performance of each model.

## Conclusion

This project provides a comprehensive analysis of various machine learning and deep learning models for classifying images from the Fashion MNIST dataset. The results show that complex models like CNN+RNN and fine-tuned MobileNetV2 achieve the highest accuracy. Additionally, the project employs several visualization techniques to gain a deeper understanding of the models' performance and identify areas for improvement. The findings can be valuable for selecting suitable models for similar image classification tasks and understanding the factors influencing model performance.

## 👋 HellO There! Let's Dive Into the World of Ideas 🚀

Hey, folks! I'm **Himanshu Rajak**, your friendly neighborhood tech enthusiast. When I'm not busy solving DSA problems or training models that make computers *a tad bit smarter*, you’ll find me diving deep into the realms of **Data Science**, **Machine Learning**, and **Artificial Intelligence**.  

Here’s the fun part: I’m totally obsessed with exploring **Large Language Models (LLMs)**, **Generative AI** (yes, those mind-blowing AI that can create art, text, and maybe even jokes one day 🤖), and **Quantum Computing** (because who doesn’t love qubits doing magical things?).  

But wait, there's more! I’m also super passionate about publishing research papers and sharing my nerdy findings with the world. If you’re a fellow explorer or just someone who loves discussing tech, memes, or AI breakthroughs, let’s connect!

- **LinkedIn**: [Himanshu Rajak](https://www.linkedin.com/in/himanshu-rajak-22b98221b/) (Professional vibes only 😉)
- **Medium**: [Himanshu Rajak](https://himanshusurendrarajak.medium.com/) (Where I pen my thoughts and experiments 🖋️)

Let’s team up and create something epic. Whether it’s about **generative algorithms** or **quantum wizardry**, I’m all ears—and ideas!  
🎯 Ping me, let’s innovate, and maybe grab some virtual coffee. ☕✨


