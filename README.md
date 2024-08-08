# Tomato Disease Detection Model

## Overview

This project focuses on developing a Convolutional Neural Network (CNN) for detecting diseases in tomato plants using image data. The model was implemented using TensorFlow and Keras, and its performance was evaluated on a dataset of tomato plant images.

## Data Preparation

### Dataset

- **Source:** Google Drive folder located at `/content/drive/My Drive/Projects/Tomato-disease-detection/Data`
- **Image Dimensions:** 256x256 pixels
- **Batch Size:** 32
- **Number of Epochs:** 50

### Data Splitting

- **Training Set:** 72% of the dataset
- **Validation Set:** 18% of the dataset
- **Test Set:** 10% of the dataset

### Data Augmentation

To enhance model generalization and prevent overfitting, the following augmentations were applied to the training images:
- Random horizontal and vertical flips
- Random rotations

## Model Architecture

### Base Model

- **Type:** Sequential Convolutional Neural Network (CNN)

### Layer Configuration

- **Convolutional Layers:**
  - 32 filters, kernel size (3x3), activation function ReLU
  - 64 filters, kernel size (3x3), activation function ReLU
  - 64 filters, kernel size (3x3), activation function ReLU
- **Pooling Layers:**
  - MaxPooling layers following each convolutional layer
- **Flattening Layer:**
  - Converts 2D feature maps into 1D feature vectors
- **Dense Layers:**
  - Fully connected layers with ReLU activation functions
  - Output Layer: Softmax activation function for multi-class classification

## Training

### Process

- **Checkpoint Callback:** Saves model weights at the end of each epoch.
- **Performance Metrics Monitored:**
  - Training Accuracy
  - Training Loss
  - Validation Accuracy
  - Validation Loss

### Training Progress

- **Initial Accuracy:** ~29% in the first epoch
- **Final Accuracy:** 97.78% on the test set
- **Initial Loss:** 1.8134 in the first epoch
- **Final Loss:** 0.0989

### Validation Performance

- **Highest Validation Accuracy:** 96.57% in the 48th epoch
- **Validation Loss:** Generally under 0.25 in later epochs

### Visualization

Curves for training and validation accuracy and loss were plotted to assess the model's performance over epochs.

## Evaluation

### Test Set Performance

- **Accuracy:** 97.92%
- **Loss:** 0.0781

### Confusion Matrix

The confusion matrix provides insights into the model's performance across different classes, helping to identify strengths and weaknesses. (Note: The detailed confusion matrix may be truncated.)

### Precision, Recall, and F1 Scores

- **Macro F1 Score:** 75.32%
- **Micro F1 Score:** 94.79%

### Distribution of Confidences

A histogram shows that most predictions are made with high confidence, which is encouraging for practical applications.

### Random Image Prediction Results

Example images with true labels, predicted labels, and confidence scores were displayed to understand model performance on individual samples.

## Model Saving

- **Location:** The final model is saved and versioned in the directory `/content/drive/My Drive/Projects/Tomato-disease-detection/Models`.

## Key Findings

1. **High Test Accuracy:** Achieved an accuracy of 97.92% on the test set.
2. **Class-Specific Performance:** Variation in precision, recall, and F1 scores across classes provides insight into areas for improvement.
3. **Confusion Matrix Analysis:** Helps in understanding the model’s performance across different disease types.
4. **Confidence in Predictions:** Most predictions are made with high confidence, beneficial for real-world applications.


